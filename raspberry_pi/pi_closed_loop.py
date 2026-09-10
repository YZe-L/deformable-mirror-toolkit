# SPDX-License-Identifier: GPL-3.0-or-later

"""Apply PC-issued single-channel commands on the Raspberry Pi."""

import argparse
import os
import sys
import time

import pi_link


def build_pwm(dry):
    if dry:
        class DryPWM:
            def set_pwm(self, ch, on, off):
                print("    [dry] set_pwm(ch=%d, 0, %d)" % (ch, off))
        print("DRY RUN: no hardware, set_pwm calls are printed only.")
        return DryPWM()
    from ServoPi import PWM
    pwm = PWM(0x40)
    pwm.set_pwm_freq(1526)
    pwm.output_enable()
    print("ServoPi ready @0x40, 1526 Hz.")
    return pwm


def serve(link, pwm, channel, bias):
    """Apply the bias, then apply each T_GOTO the PC sends, acking T_SETTLED.

    The ack carries the Pi-side set duration (ms); the PC times the link RTT
    and the post-settle capture on its own clock, so the full send->set->
    observe chain is logged end to end.

    Args:
        link: Active Pi communication link.
        pwm: PWM controller used to issue commands.
        channel: Actuator channel identifier.
        bias: Baseline PWM command applied to the actuator.
    """
    bias = max(0, min(4095, int(bias)))
    pwm.set_pwm(channel, 0, bias)
    link.send(pi_link.T_SETTLED, bit=bias, seq=-1, set_ms=0.0)  # Ready at bias.
    print("ch%d biased to %d. Waiting for PC setpoints (Ctrl+C to stop)..."
          % (channel, bias), flush=True)

    while True:
        msg = link.get(timeout=1.0)
        if msg is None:
            continue
        t = msg.get("t")
        if t == pi_link.T_GOTO:
            bit = max(0, min(4095, int(msg.get("bit", bias))))
            seq = int(msg.get("seq", -1))
            t0 = time.monotonic()
            pwm.set_pwm(channel, 0, bit)
            set_ms = (time.monotonic() - t0) * 1000.0
            link.send(pi_link.T_SETTLED, bit=bit, seq=seq, set_ms=set_ms)
            print("  goto %d (seq %d)  set in %.2f ms" % (bit, seq, set_ms),
                  flush=True)
        elif t == pi_link.T_DONE:
            print("PC ended the session.", flush=True)
            return


HOST_CACHE = os.path.expanduser("~/.pi_sweep_link_host")


def _load_cached_host():
    try:
        with open(HOST_CACHE) as fh:
            host, port = fh.read().strip().split(":")
            return host, int(port)
    except (OSError, ValueError):
        return None


def _save_cached_host(host, port):
    try:
        with open(HOST_CACHE, "w") as fh:
            fh.write("%s:%d" % (host, port))
    except OSError:
        pass


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
    """Resolve host.

    Explicit -H (cached) > wired direct link > UDP discovery > last cached.
    """
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
    print("Discovery failed and no cached host. Read the IP from the app's\n"
          "'Connect to Pi' dialog and run:  python3 pi_closed_loop.py -H <ip>")
    return None


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="PC-led closed-loop DM driver over pi_link.")
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
    ap.add_argument("-b", "--bias", type=int, default=2000,
                    help="initial DM bit (mid-stroke operating point)")
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
    link = pi_link.connect(host, port, name="pi-loop", retries=-1)
    print("Linked to Windows (%s)." % link.peer)

    pwm = build_pwm(args.dry_run)
    rc = 0
    try:
        serve(link, pwm, args.channel, args.bias)
    except KeyboardInterrupt:
        print("\nCtrl+C -> aborting.")
        link.close(reason="pi user abort")
        rc = 130
    except pi_link.LinkClosed as e:
        print("\nLink lost: %s" % e)
        rc = 1
    finally:
        try:
            pwm.set_pwm(args.channel, 0, 0)
            print("ch%d reset to 0." % args.channel)
        except Exception:
            pass
        link.close(notify=(rc == 0))
    return rc


if __name__ == "__main__":
    sys.exit(main())
