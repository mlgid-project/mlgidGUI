#!/usr/bin/env python3
"""External memory profiler for mlgidGUI.

Samples /proc/<pid>/status at a fixed interval, records VmRSS / VmSize
to a CSV, and reads the kernel's VmHWM / VmPeak (high watermarks)
at exit. Stdlib only; Linux only.

Whole-process RSS is what answers "max RAM usage": the LRU cache, Qt
widgets, the h5py buffer pool, and numpy arrays all live inside the
same process. Run the GUI in one terminal, this tool in another, drive
the workload (open a stack, play it, run the pipeline), then Ctrl-C
to stop sampling and print the summary.

Usage:
    python mlgidGUI/tools/memprofile.py
    python mlgidGUI/tools/memprofile.py --pid 12345
    python mlgidGUI/tools/memprofile.py --interval 0.25 --out mem.csv --plot mem.png
"""
import argparse
import csv
import glob
import os
import signal
import statistics
import sys
import time

STATUS_KEYS = ("VmRSS", "VmSize", "VmHWM", "VmPeak", "VmData")


def read_status(pid: int) -> dict | None:
    """Return a dict of selected /proc/<pid>/status fields in kB, or None
    if the process is gone."""
    out: dict = {}
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                if k in STATUS_KEYS:
                    out[k] = int(v.strip().split()[0])  # value is in kB
    except FileNotFoundError:
        return None
    return out


def find_pid(needle: str) -> list[tuple[int, str]]:
    """Scan /proc for processes whose cmdline contains `needle`
    (case-insensitive). Returns [(pid, cmdline), ...]."""
    hits: list[tuple[int, str]] = []
    self_pid = os.getpid()
    needle = needle.lower()
    for entry in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(entry, "rb") as fh:
                raw = fh.read()
        except (OSError, PermissionError):
            continue
        cmd = raw.decode("utf-8", "replace").replace("\x00", " ").strip()
        if needle in cmd.lower():
            pid = int(entry.split("/")[2])
            if pid != self_pid:
                hits.append((pid, cmd))
    return hits


def fmt_mb(kB: float) -> str:
    return f"{kB / 1024.0:.1f} MiB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample mlgidGUI's RSS over time; report peak RAM at exit."
    )
    ap.add_argument("--pid", type=int,
                    help="PID to sample. If omitted, auto-discovers a process "
                         "whose cmdline contains --needle.")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="sample interval in seconds (default 0.5)")
    ap.add_argument("--out", default="memprof.csv",
                    help="CSV output path (default memprof.csv)")
    ap.add_argument("--plot",
                    help="optional PNG plot path (uses matplotlib if available)")
    ap.add_argument("--needle", default="mlgidgui",
                    help="substring to match for auto-discover (default mlgidgui)")
    args = ap.parse_args()

    if args.pid is None:
        hits = find_pid(args.needle)
        if not hits:
            sys.exit(f"no process matching '{args.needle}'; pass --pid")
        if len(hits) > 1:
            print(f"multiple processes match '{args.needle}'; pass --pid:")
            for pid, cmd in hits:
                print(f"  pid={pid}  cmd={cmd[:120]}")
            sys.exit(1)
        args.pid, cmd = hits[0]
        print(f"profiling pid={args.pid}  cmd={cmd[:120]}")

    if read_status(args.pid) is None:
        sys.exit(f"pid {args.pid} is not running")

    t0 = time.monotonic()
    samples: list[tuple[float, int, int]] = []
    fh = open(args.out, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["t_seconds", "rss_kB", "vms_kB"])

    interrupted = False

    def on_sigint(_signo, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)
    print(f"sampling every {args.interval}s. Ctrl-C to stop.")

    try:
        while not interrupted:
            s = read_status(args.pid)
            if s is None:
                print("target process exited; finalising.")
                break
            t = time.monotonic() - t0
            rss = s.get("VmRSS", 0)
            vms = s.get("VmSize", 0)
            samples.append((t, rss, vms))
            writer.writerow((f"{t:.3f}", rss, vms))
            fh.flush()
            # Sleep in small slices so Ctrl-C responds promptly.
            slept = 0.0
            while slept < args.interval and not interrupted:
                step = min(0.1, args.interval - slept)
                time.sleep(step)
                slept += step
    finally:
        fh.close()

    final = read_status(args.pid) or {}
    n = len(samples)
    if n == 0:
        print("no samples recorded.")
        return

    rss_vals = [r for _, r, _ in samples]
    duration = samples[-1][0]
    peak_rss = final.get("VmHWM", max(rss_vals))
    peak_vms = final.get("VmPeak", max(v for _, _, v in samples))
    final_rss = final.get("VmRSS", rss_vals[-1])
    mean_rss = statistics.fmean(rss_vals)

    print()
    print(f"samples           {n}  over  {duration:.1f} s")
    print(f"peak RSS (HWM)    {fmt_mb(peak_rss)}  (kernel-tracked high watermark)")
    print(f"peak VMS (Peak)   {fmt_mb(peak_vms)}")
    print(f"final RSS         {fmt_mb(final_rss)}")
    print(f"sampled mean RSS  {fmt_mb(mean_rss)}")
    if n >= 20:
        p95 = sorted(rss_vals)[int(0.95 * n) - 1]
        print(f"sampled p95 RSS   {fmt_mb(p95)}")
    print(f"csv               {os.path.abspath(args.out)}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ts = [t for t, _, _ in samples]
            rss_mb_series = [r / 1024.0 for r in rss_vals]
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(ts, rss_mb_series, lw=1.2)
            ax.axhline(peak_rss / 1024.0, color="C3", lw=0.8, ls="--",
                       label=f"peak {peak_rss / 1024.0:.0f} MiB (VmHWM)")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("RSS (MiB)")
            ax.set_title(f"mlgidGUI RSS  pid={args.pid}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            plt.close(fig)
            print(f"plot              {os.path.abspath(args.plot)}")
        except ImportError:
            print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
