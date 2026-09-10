# SPDX-License-Identifier: GPL-3.0-or-later

"""Run terminal-driven lock-step sweeps on the Raspberry Pi."""

# Example modes: `ramp --channel 1 --start 0 --end 4095 --step 100`,
# `sweep --channel 1 --ranges 4095,2000,1000 --step 100`, and
# `ramp --dry-run`. Use `--help` for host and rest options.


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


def plan_ramp(start, end, step):
    """Plan ramp.

    Same point order as measure_ramp(): up then down, peak photographed once.

    Args:
        start: Inclusive start value.
        end: Inclusive end value.
        step: Increment between consecutive values.
    """
    up = list(range(start, end + 1, step))
    if up[-1] != end:
        up.append(end)
    down = up[-2::-1]
    pts = []
    for v in up:
        pts.append(dict(bit=v, dir="up", peak=(v == end), rng=0, rest=0.0))
    for v in down:
        pts.append(dict(bit=v, dir="down", peak=False, rng=0, rest=0.0))
    return pts


def plan_sweep(ranges, step, range_rest):
    """Plan sweep.

    Same point order as sweep_ranges(): 0->r->0 per range, shared 0 not
    repeated. range_rest is attached as pre-rest on each new range's first
    point.

    Args:
        ranges: Sequence of range values.
        step: Increment between consecutive values.
        range_rest: Rest duration between sweep ranges, in seconds.
    """
    ranges = [max(0, min(4095, int(r))) for r in ranges]
    pts = []
    last = len(ranges) - 1
    for i, r in enumerate(ranges):
        up = list(range(0, r, step)) + [r]
        seq = up + up[-2::-1]
        peak_v = r
        if i > 0:
            seq = seq[1:]  # Already sitting at 0 from prev range.
        for j, v in enumerate(seq):
            first_of_range = (i > 0 and j == 0)
            going_up = v in up and (j < len(up) - (0 if i == 0 else 1))
            pts.append(dict(
                bit=v,
                dir="up" if going_up else "down",
                peak=(v == peak_v),
                rng=i,
                rest=(range_rest if first_of_range else 0.0)))
    return pts


def plan_from_pc(bits, rests):
    """Build the point list from the PC's T_PLAN.

    The Pi computes NOTHING -- every bit is the PC's final hardware value
    (already hysteresis-compensated where the PC chose to). dir/peak/rng are
    cosmetic here (the PC cross-checks bit + index only); rest drives the
    inter-range pause.

    Args:
        bits: PWM command bits.
        rests: Rest durations paired with the command sequence.
    """
    bits = [int(b) for b in bits]
    rests = [float(r) for r in (rests or [])]
    pts = []
    for i, b in enumerate(bits):
        rest = rests[i] if i < len(rests) else 0.0
        direction = "up" if i == 0 or b >= bits[i - 1] else "down"
        pts.append(dict(bit=b, dir=direction, peak=False, rng=0, rest=rest))
    return pts


def wait_for_plan(link):
    """Block until the PC ships the sweep (T_PLAN) or the link dies.

    Returns the point list, or None if the PC ended the session first.
    """
    print("Waiting for the PC to send the sweep plan "
          "(press Start in the app)...", flush=True)
    while True:
        msg = link.get(timeout=1.0)  # Raises LinkClosed if the PC leaves.
        if msg is None:
            continue
        if msg.get("t") == pi_link.T_PLAN:
            bits = msg.get("bits", [])
            print("Received plan: %d points." % len(bits), flush=True)
            return plan_from_pc(bits, msg.get("rests", []))
        if msg.get("t") == pi_link.T_DONE:
            return None


def wait_for(link, msg_type, index):
    """Block until a message of msg_type for this index arrives.

    Ignores other traffic. Raises LinkClosed if the peer is gone. Returns the
    message.

    Args:
        link: Active Pi communication link.
        msg_type: Selected kind of msg.
        index: Zero-based item index.
    """
    while True:
        msg = link.get(timeout=1.0)
        if msg is None:
            continue
        if msg.get("t") != msg_type:
            continue
        if int(msg.get("index", -1)) != index:
            continue  # Stale ack for an earlier point.
        return msg


def wait_recorded(link, index, bit):
    """Wait recorded.

    Block until Windows acks THIS point is recorded, cross-checking the bit.
    Raises on mismatch or dead link.

    Args:
        link: Active Pi communication link.
        index: Zero-based item index.
        bit: PWM command bit.
    """
    msg = wait_for(link, pi_link.T_RECORDED, index)
    got = int(msg.get("bit", -10**9))
    if got != bit:
        raise RuntimeError(
            "bit mismatch at index %d: Pi=%d, Windows recorded=%d"
            % (index, bit, got))


def run(link, pwm, channel, pts, delay):
    """Run the configured worker loop.

    Args:
        link: Active Pi communication link.
        pwm: PWM controller used to issue commands.
        channel: Actuator channel identifier.
        pts: Sequence of pt values.
        delay: Delay between operations, in seconds.
    """
    n = len(pts)
    # Reference point (no pre-delay; Windows records it when you press Start)
    p0 = pts[0]
    pwm.set_pwm(channel, 0, p0["bit"])
    print("Reference: ch%d -> %d. Waiting for Windows to press Start..."
          % (channel, p0["bit"]), flush=True)
    link.send(pi_link.T_AT, bit=p0["bit"], index=0, dir=p0["dir"],
              peak=p0["peak"], rng=p0["rng"], total=n)
    wait_recorded(link, 0, p0["bit"])
    print("  Windows recorded reference. Sweeping...", flush=True)

    rtts = []  # Pure link round-trip per point.
    for i in range(1, n):
        p = pts[i]
        if p["rest"] > 0:
            # Entering the rest between ranges: tell Windows so it can keep
            # monitoring the return point during the wait, then sleep.
            link.send(pi_link.T_GAP, secs=p["rest"], rng=p["rng"])
            print("  ...range rest %.1fs (Windows may monitor)" % p["rest"],
                  flush=True)
            time.sleep(p["rest"])
        else:
            time.sleep(delay)
        pwm.set_pwm(channel, 0, p["bit"])
        # Print the move RIGHT NOW (the mirror has just moved) so the terminal
        # tracks the real hardware state instead of lagging until Windows has
        # finished recording (which is seconds later, after the settle wait).
        print("  #%d/%d  ch%d -> %d (%s%s)  moved, waiting for Windows..."
              % (i + 1, n, channel, p["bit"], p["dir"],
                 ", peak" if p["peak"] else ""), flush=True)
        # Link latency only: 'at' sent -> immediate 'got' (excludes Settle wait)
        t0 = time.monotonic()
        link.send(pi_link.T_AT, bit=p["bit"], index=i, dir=p["dir"],
                  peak=p["peak"], rng=p["rng"], total=n)
        wait_for(link, pi_link.T_GOT, i)
        rtt_ms = (time.monotonic() - t0) * 1000.0
        rtts.append(rtt_ms)
        wait_recorded(link, i, p["bit"])  # Lock-step gate (settle+record)
        print("        recorded by Windows | link RTT %.1f ms" % rtt_ms,
              flush=True)

    link.send(pi_link.T_DONE, total=n)
    print("Sequence complete (%d points)." % n)
    if rtts:
        r = sorted(rtts)
        print("Link latency Pi->Win->Pi round-trip (settle NOT included): "
              "mean %.1f ms | min %.1f | max %.1f | median %.1f  (n=%d)"
              % (sum(r) / len(r), r[0], r[-1], r[len(r) // 2], len(r)))


HOST_CACHE = os.path.expanduser("~/.pi_sweep_link_host")

EXAMPLES = """
examples (combos):
  ramp, auto-find PC:      python3 pi_sweep_link.py ramp
  ramp on channel 2:       python3 pi_sweep_link.py ramp -c 2
  finer steps, 1s settle:  python3 pi_sweep_link.py ramp -s 50 -d 1
  point to the PC by IP:   python3 pi_sweep_link.py ramp -H 172.26.1.5
  sweep 3 ranges:          python3 pi_sweep_link.py sweep --ranges 4095,2000,1000
  sweep + 10s range rest:  python3 pi_sweep_link.py sweep --ranges 4095,2000 -R 10
  dry run (no hardware):   python3 pi_sweep_link.py ramp --dry-run

The PC's IP is shown in the app's "Connect to Pi" dialog. Once you pass it with
-H it is cached (~/.pi_sweep_link_host), so next time you can drop -H entirely.
"""


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Lock-step DM sweep over pi_link (ramp / sweep).",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ramp", "sweep", "pc"],
                    help="ramp = single 0->peak->0; sweep = several ranges; "
                         "pc = wait for the app to send the sweep (the app "
                         "builds it, incl. hysteresis compensation)")
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
    ap.add_argument("-s", "--step", type=int, default=100)
    ap.add_argument("-d", "--delay", type=float, default=2.0,
                    help="Pi-side settle before each move (s)")
    # Ramp
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=4095)
    # Sweep
    ap.add_argument("--ranges", default="4095,2000,1000")
    ap.add_argument("-R", "--range-rest", type=float, default=0.0,
                    help="rest at 0 before each new range (s)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


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

    Pick the PC to connect to: explicit -H (and cache it) > wired direct
    link > UDP discovery > last cached host. Returns (host, port) or None.
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
          "'Connect to Pi' dialog and run:  python3 pi_sweep_link.py %s -H <ip>"
          % args.mode)
    return None


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # In 'pc' mode the app sends the sweep after we connect, so the plan is
    # built later (see wait_for_plan). ramp/sweep still plan up front.
    if args.mode == "ramp":
        pts = plan_ramp(args.start, args.end, args.step)
    elif args.mode == "sweep":
        ranges = [int(x) for x in args.ranges.split(",") if x.strip()]
        pts = plan_sweep(ranges, args.step, args.range_rest)
    else:
        pts = None
    if pts is not None:
        print("Planned %d points (%s)." % (len(pts), args.mode))
    print("This Pi IP: %s" % pi_link.local_ip())

    resolved = resolve_host(args)
    if resolved is None:
        return 2
    host, port = resolved

    print("Connecting to %s:%d ..." % (host, port))
    link = pi_link.connect(host, port, name="pi-sweep", retries=-1)
    print("Linked to Windows (%s)." % link.peer)

    pwm = build_pwm(args.dry_run)
    rc = 0
    try:
        if pts is not None:  # Ramp/sweep: one fixed plan.
            run(link, pwm, args.channel, pts, args.delay)
        else:  # 'pc': serve plans until we quit
            while True:  # So RAW + COMP passes (and many.
                plan = wait_for_plan(link)  # Loop sessions) need ONE launch.
                if plan is None:
                    print("PC ended the session.")
                    break
                run(link, pwm, args.channel, plan, args.delay)
                print("Plan done; waiting for the next plan "
                      "(Ctrl+C to quit)...", flush=True)
    except KeyboardInterrupt:
        print("\nCtrl+C -> aborting.")
        link.close(reason="pi user abort")
        rc = 130
    except pi_link.LinkClosed as e:
        print("\nLink lost: %s" % e)
        rc = 1
    except RuntimeError as e:
        print("\nABORT: %s" % e)
        link.close(reason=str(e))
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
