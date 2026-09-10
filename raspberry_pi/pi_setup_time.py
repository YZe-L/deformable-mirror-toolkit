# SPDX-License-Identifier: GPL-3.0-or-later

"""Trigger mirror setup-time steps and report results from the PC."""

import argparse
import sys
import time

import pi_link
from pi_sweep_link import build_pwm, _load_cached_host, _save_cached_host


# Fixed PC address on a direct Ethernet link (PC <-> Pi cable, no router). Set
# the Windows adapter to this IP with a blank gateway so it never steals the
# default route -- both machines keep internet on their other NIC.
WIRED_HOST = "192.168.77.1"


def _reachable(host, port, timeout=1.5):
    """Quick TCP probe: is a server actually listening at host:port now?

    Lets the wired link be preferred only when the cable is really up, so a
    missing cable falls straight through to the normal discovery path.

    Args:
        host: Remote host name or address.
        port: TCP port number.
        timeout: Maximum wait time, in seconds.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_host(args):
    if args.host:
        _save_cached_host(args.host, args.port)
        return args.host, args.port
    # Direct-Ethernet path: prefer the fixed wired IP when the cable is up,
    # probed first so an unplugged link falls through to discovery below.
    if args.wired:
        if _reachable(args.wired, args.port):
            print("Wired direct link: PC reachable at %s:%d"
                  % (args.wired, args.port))
            return args.wired, args.port
        print("Wired link %s:%d not answering (cable? PC firewall on the "
              "direct NIC?) -- falling back to UDP discovery"
              % (args.wired, args.port))
    print("No -H given, discovering the PC over UDP (5 s)...")
    found = pi_link.discover(timeout=5.0)
    if found:
        print("Discovered PC at %s:%d" % found)
        _save_cached_host(*found)
        return found
    cached = _load_cached_host()
    if cached:
        print("Discovery failed -> using last host %s:%d" % cached)
        return cached
    print("Discovery failed and no cached host. Run with -H <pc-ip>.")
    return None


def wait_result(link, trial, timeout=20.0):
    """Block for Windows' result for this trial (or timeout).

    None on timeout.

    Args:
        link: Active Pi communication link.
        trial: Trial identifier or trial record.
        timeout: Maximum wait time, in seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = link.get(timeout=1.0)  # Raises LinkClosed if peer gone.
        if msg is None:
            continue
        if (msg.get("t") == pi_link.T_RESULT
                and int(msg.get("trial", -1)) == trial):
            return msg
    return None


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="DM setup-time trigger over pi_link.",
        epilog="Press Enter to step (toggles 0 <-> --bit); Ctrl+C to quit.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-H", "--host", default=None,
                    help="PC IP; omit to auto-discover / use the cached one")
    ap.add_argument("--wired", nargs="?", const=WIRED_HOST, default=None,
                    metavar="IP",
                    help="prefer a direct Ethernet link to the PC. Bare "
                         "--wired uses %s; --wired <IP> overrides. Probed "
                         "first, then falls back to UDP discovery, so it is "
                         "safe to leave on." % WIRED_HOST)
    ap.add_argument("-p", "--port", type=int, default=65432)
    ap.add_argument("-c", "--channel", type=int, default=1)
    ap.add_argument("-b", "--bit", type=int, default=2000,
                    help="step target (Enter toggles 0 <-> bit)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print("This Pi IP: %s" % pi_link.local_ip())
    resolved = resolve_host(args)
    if resolved is None:
        return 2
    host, port = resolved
    print("Connecting to %s:%d ..." % (host, port))
    link = pi_link.connect(host, port, name="pi-setup", retries=-1)
    print("Linked to Windows (%s)." % link.peer)

    pwm = build_pwm(args.dry_run)
    pwm.set_pwm(args.channel, 0, 0)
    state = 0  # Current DM level.
    trial = 0
    print("\nReady. On the PC: connect and arm the recorder. Each Enter toggles\n"
          "the DM between 0 and %d (a step up or down); it does NOT auto-reset,\n"
          "so press Enter again to step back. Ctrl+C quits." % args.bit)
    rc = 0
    try:
        while True:
            up = (state == 0)
            target = args.bit if up else 0
            input("  >>> Enter to step %s (Ctrl+C quit) "
                  % ("UP 0->%d" % args.bit if up else "DOWN %d->0" % args.bit))
            trial += 1
            # Send the trigger essentially together with the step so Windows'
            # t0 lines up with the move (the ~ms between them is negligible).
            link.send(pi_link.T_TRIGGER, trial=trial, bit=target, up=up)
            pwm.set_pwm(args.channel, 0, target)
            state = target
            print("    trial %d: stepped %s to %d. Waiting for Windows..."
                  % (trial, "UP" if up else "DOWN", target), flush=True)
            res = wait_result(link, trial)
            if res is None:
                print("    (no result -- is the test armed on Windows?)")
            else:
                print("    delay to first %s = %.1f ms   (amplitude %.1f nm)\n"
                      % ("rise" if up else "fall",
                         float(res.get("setup_s", 0)) * 1000.0,
                         float(res.get("amp_nm", 0))), flush=True)
    except KeyboardInterrupt:
        print("\nCtrl+C -> quitting.")
    except pi_link.LinkClosed as e:
        print("\nLink lost: %s" % e)
        rc = 1
    finally:
        try:
            pwm.set_pwm(args.channel, 0, 0)
        except Exception:
            pass
        link.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
