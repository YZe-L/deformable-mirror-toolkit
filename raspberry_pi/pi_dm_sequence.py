# SPDX-License-Identifier: GPL-3.0-or-later

"""Apply PC-issued multi-channel mirror commands on the Raspberry Pi."""

import argparse
import os
import sys
import time

import pi_link

# Every DM piezo channel this Pi can drive: 1..5 is the five-element mirror,
# 6..14 the nine-element one. The exit path zeroes all of them.
CHANNELS = tuple(range(1, 15))  # 1..5 = DM5, 6..14 = DM9.
BIT_MIN, BIT_MAX = 0, 4095


def build_pwm(dry):
    if dry:
        class DryPWM:
            def set_pwm(self, ch, on, off):
                print("    [dry] set_pwm(ch=%d, 0, %d)" % (ch, off), flush=True)
        print("DRY RUN: no hardware, set_pwm calls are printed only.")
        return DryPWM()
    from ServoPi import PWM
    pwm = PWM(0x40)
    pwm.set_pwm_freq(1526)
    pwm.output_enable()
    print("ServoPi ready @0x40, 1526 Hz.")
    return pwm


def _clamp(b):
    return max(BIT_MIN, min(BIT_MAX, int(b)))


def apply_channels(pwm, channels, state):
    """Set each channel to its bit; remember the applied state.

    Args:
        pwm: PWM controller used to issue commands.
        channels: Actuator channel identifiers.
        state: Current actuator or workflow state.
    """
    for c, b in channels.items():
        c, b = int(c), _clamp(b)
        pwm.set_pwm(c, 0, b)
        state[c] = b


def serve(link, pwm, default_channel, bias, verbose=False, state=None):
    """Bias the default channel, then apply each T_GOTO the PC sends.

    Args:
        link: Active Pi communication link.
        pwm: PWM controller used to issue commands.
        default_channel: Actuator channel used when a command omits one.
        bias: Baseline PWM command applied to the actuator.
        verbose: Whether to emit detailed progress messages.
        state: Caller-owned {channel: bit} of what has been driven. Passed in
            rather than kept local so the exit path can zero exactly the
            channels this session touched, including any outside `CHANNELS`.
            A piezo left under drive because nobody was tracking it is a
            hardware risk, not a logging gap.
    """
    if state is None:
        state = {c: 0 for c in CHANNELS}
    apply_channels(pwm, {default_channel: bias}, state)
    link.send(pi_link.T_SETTLED, seq=-1, set_ms=0.0)  # Ready
    print("Ready. default ch%d biased to %d. Waiting for PC setpoints..."
          % (default_channel, bias), flush=True)

    n = 0
    while True:
        msg = link.get(timeout=1.0)
        if msg is None:
            continue
        t = msg.get("t")
        if t == pi_link.T_GOTO:
            chans = msg.get("channels")
            if not chans:  # Single-channel fallback
                chans = {default_channel: msg.get("bit", bias)}
            seq = int(msg.get("seq", -1))
            t0 = time.monotonic()
            apply_channels(pwm, chans, state)
            set_ms = (time.monotonic() - t0) * 1000.0
            link.send(pi_link.T_SETTLED, seq=seq, set_ms=set_ms)  # Ack FIRST
            n += 1
            # Throttle the console: printing every setpoint can stall this
            # loop over a slow SSH terminal. The ack is never throttled.
            if verbose or n % 50 == 0:
                print("  applied %d setpoints (last seq %d, %.2f ms)"
                      % (n, seq, set_ms), flush=True)
        elif t == pi_link.T_DONE:
            print("PC ended the session (%d setpoints applied)." % n,
                  flush=True)
            return


HOST_CACHE = os.path.expanduser("~/.pi_sweep_link_host")


# Fixed PC address on a direct Ethernet link (PC <-> Pi crossover cable, no
# router). Set the Windows adapter to this IP with a blank gateway so it never
# steals the default route -- both machines keep internet on their other NIC.
WIRED_HOST = "192.168.77.1"


def _reachable(host, port, timeout=1.5):
    """Check whether the server is reachable.

    Quick TCP probe: is a server actually listening at host:port right now?
    Used to prefer the wired link only when it is really up, so a missing cable
    falls straight through to the normal discovery path.

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
    # Explicit -H always wins, wired or not.
    if args.host:
        return args.host, args.port
    # NEW: direct-Ethernet path. Prefer the fixed wired IP when the cable is up,
    # but probe first so an unplugged link falls through to discovery below --
    # the original logic is untouched, this only prepends a preferred route.
    if args.wired:
        if _reachable(args.wired, args.port):
            print("Wired direct link: PC reachable at %s:%d" % (args.wired,
                                                                args.port))
            return args.wired, args.port
        print("Wired link %s:%d not answering (cable? Windows firewall on the "
              "direct NIC?) -- falling back to UDP discovery" % (args.wired,
                                                                 args.port))
    print("No -H given, discovering the PC over UDP (5 s)...")
    found = pi_link.discover(timeout=5.0)
    if found:
        print("Discovered PC at %s:%d" % found)
        return found
    try:
        with open(HOST_CACHE) as fh:
            host, port = fh.read().strip().split(":")
            print("Discovery failed -> cached host %s:%s" % (host, port))
            return host, int(port)
    except (OSError, ValueError):
        print("No PC found. Read the IP from the app's Connect dialog and run:\n"
              "    python3 pi_dm_sequence.py -H <ip>")
        return None


def parse_args(argv):
    ap = argparse.ArgumentParser(description="PC-led multi-channel DM driver.")
    ap.add_argument("-H", "--host", default=None)
    ap.add_argument("-p", "--port", type=int, default=65432)
    ap.add_argument("--wired", nargs="?", const=WIRED_HOST, default=None,
                    metavar="IP",
                    help="prefer a direct Ethernet link to the PC. Bare "
                         "--wired uses %s; --wired <IP> overrides. Probed "
                         "first, then falls back to UDP discovery if the cable "
                         "is down, so it is safe to leave on." % WIRED_HOST)
    ap.add_argument("-c", "--channel", type=int, default=1,
                    help="default channel for single-bit commands / bias")
    ap.add_argument("-b", "--bias", type=int, default=0)
    ap.add_argument("--zero-on-exit", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="print every setpoint (default: a line every 50). "
                         "Leave off for long automation runs to avoid terminal "
                         "back-pressure stalling the loop.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print("This Pi IP: %s" % pi_link.local_ip())
    resolved = resolve_host(args)
    if resolved is None:
        return 2
    host, port = resolved
    print("Connecting to %s:%d ..." % (host, port))
    link = pi_link.connect(host, port, name="pi-dm-seq", retries=-1)
    print("Linked to Windows (%s)." % link.peer)

    pwm = build_pwm(args.dry_run)
    rc = 0
    # Owned here, not inside serve(), so the finally below can zero whatever
    # was actually driven no matter how serve() ended.
    state = {c: 0 for c in CHANNELS}
    try:
        serve(link, pwm, args.channel, args.bias, verbose=args.verbose,
              state=state)
    except KeyboardInterrupt:
        print("\nCtrl+C -> aborting.")
        link.close(reason="pi user abort")
        rc = 130
    except pi_link.LinkClosed as e:
        print("\nLink lost: %s" % e)
        rc = 1
    finally:
        if args.zero_on_exit:
            # Every channel this session drove, plus every channel that could
            # be wired. Zeroed one at a time so one bad channel cannot leave
            # the rest of the mirror held.
            done = []
            for c in sorted(set(CHANNELS) | set(state)):
                try:
                    pwm.set_pwm(c, 0, 0)
                    done.append(c)
                except Exception as e:
                    print("WARNING ch%d NOT zeroed: %s" % (c, e))
            print("Channels reset to 0: %s"
                  % ", ".join("ch%d" % c for c in done))
        link.close(notify=(rc == 0))
    return rc


if __name__ == "__main__":
    sys.exit(main())
