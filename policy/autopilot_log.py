"""Autopilot Session logs: what the models did, and your takeovers as training data.

The robot app's Autopilot Session screen records a run with its Logging switch (or the
gamepad's x) into Documents/OpenBot/autopilot_<date>.zip: the data logger's layout plus
sensor_data/autopilotLog.txt, one row per inferred frame:

    timestamp[ns],frame,source,direction,model,pred_steering,pred_throttle,
    steering,throttle,inference[ms],memory

source is who drove that frame: model, human (you took over with the stick), pause (the
stop on a direction switch) or none (no model loaded for that direction). pred_* is the
model's raw output - while you drive too, what it would have done - and steering,throttle
what was sent. Copy the zips (or the unpacked folders) into
dataset/fwdbwd/autopilot_runs/, then:

  report   per run and per model: how long the model and you drove, takeovers per
           minute the model drove, how far the model was from you during your
           takeovers, and <run>/autopilot_report.png, prediction and command over time.
  extract  your takeovers as a normal recording, dataset/fwdbwd/inbox/<run>_takeover/,
           for split_direction.py. Only human frames, each takeover trimmed by
           --trim_start at its start (your reaction; the car still doing what the model
           did) and --trim_end before you let go (the stick on its way back to centre).
           The 0,0 the phone sends for TAKEOVER_RELEASE_MS after that goes with it.

The phone's ctrlLog.txt holds the models' commands too, and SensorService timestamps a
line when it gets to it on the main thread, which can be after the next frame. So
extract writes its own ctrlLog.txt from autopilotLog.txt: each human row's command at the
time it was sent. Matching then gives every frame the last command sent before it, the
data logger's pairing, and extract re-matches the output to check exactly that.

Usage (from policy/, conda env):
    python autopilot_log.py report                        # every run in autopilot_runs/
    python autopilot_log.py report autopilot_20261106_153000
    python autopilot_log.py extract --dry_run
    python autopilot_log.py extract                       # then split_direction.py
"""

import argparse
import csv
import os
import shutil
from collections import namedtuple

from openbot.associate_frames import match_frame_session
from split_direction import ROOT, processed_rows, read_lines, unpack

RUNS = "autopilot_runs"
LOG = "autopilotLog.txt"
PLOT = "autopilot_report.png"
SUFFIX = "_takeover"
# A row counts for the time to the next one, but no longer than this (a stall, a gap).
MAX_ROW_S = 0.5

Row = namedtuple(
    "Row", "ts frame source direction model pred_s pred_t s t infer_ms memory"
)


def optional(value):
    return float(value) if value else None


def read_log(run):
    rows = []
    with open(os.path.join(run, "sensor_data", LOG), newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for fields in reader:
            if len(fields) < 11:
                continue
            ts, frame, source, direction, model, ps, pt, s, t, ms, mem = fields[:11]
            rows.append(
                Row(
                    int(ts),
                    int(frame),
                    source,
                    direction,
                    model,
                    optional(ps),
                    optional(pt),
                    float(s),
                    float(t),
                    optional(ms),
                    mem,
                )
            )
    return rows


def find_runs(root, names):
    """Run folders for names (paths, or names in autopilot_runs/; default: all of it).
    A zip is unpacked next to itself first, unless that folder already exists."""
    runs_dir = os.path.join(root, RUNS)
    if names:
        paths = []
        for n in names:
            for p in (n, os.path.join(runs_dir, n), os.path.join(runs_dir, n + ".zip")):
                if os.path.exists(p):
                    paths.append(p)
                    break
            else:
                raise SystemExit(f"no run {n} (looked in . and {runs_dir})")
    else:
        if not os.path.isdir(runs_dir):
            raise SystemExit(f"no {runs_dir}; copy the autopilot_*.zip from the phone there")
        paths = sorted(os.path.join(runs_dir, n) for n in os.listdir(runs_dir))
    runs = []
    for p in paths:
        p = os.path.abspath(p)
        if p.endswith(".zip"):
            folder = p[: -len(".zip")]
            if not os.path.isdir(folder):
                unpack(p, folder)
            p = folder
        if p in runs or not os.path.isdir(p):
            continue
        if os.path.isfile(os.path.join(p, "sensor_data", LOG)):
            runs.append(p)
        elif names:
            raise SystemExit(f"{p} has no sensor_data/{LOG}")
    if not runs:
        raise SystemExit("no autopilot runs found")
    return runs


def takeovers(rows):
    """Runs of consecutive human rows."""
    runs, cur = [], []
    for r in rows:
        if r.source == "human":
            cur.append(r)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def row_seconds(rows):
    """Seconds each row stands for: to the next row, at most MAX_ROW_S."""
    out = [min((b.ts - a.ts) / 1e9, MAX_ROW_S) for a, b in zip(rows, rows[1:])]
    return out + [0.0] if rows else []


def fmt_s(seconds):
    m, s = divmod(round(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


# ---------------------------------------------------------------- report


def model_stats(rows, stats=None):
    """(direction, model) -> model_s, human_s, takeovers, off_s, off_t, off_n."""
    stats = {} if stats is None else stats
    for r, sec in zip(rows, row_seconds(rows)):
        if r.source not in ("model", "human"):
            continue
        st = stats.setdefault((r.direction, r.model), [0.0, 0.0, 0, 0.0, 0.0, 0])
        if r.source == "model":
            st[0] += sec
        else:
            st[1] += sec
            # Not the stick back in the centre before the models drive again.
            if r.pred_s is not None and (r.s != 0 or r.t != 0):
                st[3] += abs(r.pred_s - r.s)
                st[4] += abs(r.pred_t - r.t)
                st[5] += 1
    for seg in takeovers(rows):
        stats.setdefault((seg[0].direction, seg[0].model), [0.0, 0.0, 0, 0.0, 0.0, 0])[
            2
        ] += 1
    return stats


def print_stats(stats, indent="  "):
    for (direction, model), (model_s, human_s, n, off_s, off_t, off_n) in sorted(
        stats.items()
    ):
        per_min = f"{n / (model_s / 60):.1f}/min" if model_s else "-"
        autonomy = model_s / (model_s + human_s) if model_s + human_s else 0
        print(
            f"{indent}{direction} {model or '(none)'}: model {fmt_s(model_s)}, "
            f"you {fmt_s(human_s)} ({autonomy:.0%} autonomous), "
            f"{n} takeover{'' if n == 1 else 's'} ({per_min})"
        )
        if off_n:
            print(
                f"{indent}    while you drove, the model was off by "
                f"|steering| {off_s / off_n:.0f}, |throttle| {off_t / off_n:.0f} "
                f"(of 255, {off_n} frames)"
            )


def spans(rows, source):
    """(start_s, end_s) of the stretches of rows with this source."""
    t0 = rows[0].ts
    out, start = [], None
    for a, b in zip(rows, rows[1:] + [None]):
        if a.source == source and start is None:
            start = (a.ts - t0) / 1e9
        if start is not None and (b is None or b.source != source):
            out.append((start, ((b or a).ts - t0) / 1e9))
            start = None
    return out


def plot(run, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t0 = rows[0].ts
    t = [(r.ts - t0) / 1e9 for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, name, pred, sent in (
        (axes[0], "steering", "pred_s", "s"),
        (axes[1], "throttle", "pred_t", "t"),
    ):
        for a, b in spans(rows, "human"):
            ax.axvspan(a, b, color="tab:orange", alpha=0.25, lw=0)
        for a, b in spans(rows, "pause"):
            ax.axvspan(a, b, color="0.6", alpha=0.3, lw=0)
        p = [getattr(r, pred) for r in rows]
        ax.plot(t, [float("nan") if v is None else v for v in p], lw=0.8, color="tab:blue")
        ax.plot(t, [getattr(r, sent) for r in rows], lw=1.0, color="black")
        ax.axhline(0, lw=0.5, color="0.5")
        ax.set_ylim(-270, 270)
        ax.set_ylabel(name)
    axes[1].set_xlabel("time [s]")
    fig.legend(
        handles=[
            plt.Line2D([], [], color="tab:blue", lw=0.8, label="model prediction"),
            plt.Line2D([], [], color="black", lw=1.0, label="sent to the car"),
            Patch(color="tab:orange", alpha=0.25, label="you drove"),
            Patch(color="0.6", alpha=0.3, label="pause"),
        ],
        loc="upper right",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(os.path.basename(run), x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(run, PLOT)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def report(runs, make_plot):
    total = {}
    for run in runs:
        rows = read_log(run)
        if len(rows) < 2:
            print(f"{os.path.basename(run)}: empty log")
            continue
        secs = row_seconds(rows)
        duration = (rows[-1].ts - rows[0].ts) / 1e9
        by_source = {}
        for r, sec in zip(rows, secs):
            by_source[r.source] = by_source.get(r.source, 0.0) + sec
        ms = [r.infer_ms for r in rows if r.source == "model" and r.infer_ms is not None]
        print(
            f"{os.path.basename(run)}: {fmt_s(duration)}, {len(rows)} frames "
            f"({len(rows) / duration:.1f} fps)"
            + (f", inference {sum(ms) / len(ms):.1f} ms" if ms else "")
        )
        print_stats(model_stats(rows))
        model_stats(rows, total)
        rest = [
            f"{s} {fmt_s(by_source[s])}" for s in ("pause", "none") if by_source.get(s)
        ]
        if rest:
            print("  " + ", ".join(rest))
        if make_plot:
            print(f"  plot: {plot(run, rows)}")
    if len(runs) > 1 and total:
        print(f"all {len(runs)} runs, by model:")
        print_stats(total)


# ---------------------------------------------------------------- extract


def kept_rows(segment, trim_start, trim_end):
    """The rows of one takeover that become training frames."""
    active = [r for r in segment if r.s != 0 or r.t != 0]
    if not active:
        return []
    start = segment[0].ts + trim_start * 1e9
    end = active[-1].ts - trim_end * 1e9
    return [r for r in segment if start <= r.ts <= end]


def extracted_already(root, name):
    """Where a takeover recording of this name already is, or None."""
    places = [os.path.join(root, "inbox", name)]
    raw = os.path.join(root, "raw")
    if os.path.isdir(raw):
        places += [os.path.join(raw, s, name) for s in sorted(os.listdir(raw))]
    parked = os.path.join(root, "parked")
    if os.path.isdir(parked):
        places += [os.path.join(parked, r, name) for r in sorted(os.listdir(parked))]
    return next((p for p in places if os.path.exists(p)), None)


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def plan_extract(run, trim_start, trim_end):
    """(kept rows, label per kept frame, human rows for ctrlLog, frame timestamps)."""
    rows = read_log(run)
    _, frame_lines = read_lines(os.path.join(run, "sensor_data", "rgbFrames.txt"))
    frame_ts = {}
    for l in frame_lines:
        ts, frame = l.split(",")[:2]
        frame_ts[int(frame)] = int(ts)
    images = set(os.listdir(os.path.join(run, "images")))
    previous = {b.frame: a for a, b in zip(rows, rows[1:])}

    kept = []
    for seg in takeovers(rows):
        for r in kept_rows(seg, trim_start, trim_end):
            prev = previous.get(r.frame)
            # The label is the command sent on the frame before; it must be yours.
            if prev is None or prev.source != "human":
                continue
            if r.frame in frame_ts and f"{r.frame}_crop.jpeg" in images:
                kept.append(r)
    # As the phone's ctrlLog has it: Vehicle's floats cast to int.
    labels = {r.frame: (int(previous[r.frame].s), int(previous[r.frame].t)) for r in kept}
    human = [r for r in rows if r.source == "human"]
    return rows, kept, labels, human, frame_ts


def write_takeover(run, out, kept, labels, human, frame_ts, trim_start, trim_end):
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "sensor_data"))
    sensor = os.path.join(out, "sensor_data")
    for r in kept:
        name = f"{r.frame}_crop.jpeg"
        link_or_copy(os.path.join(run, "images", name), os.path.join(out, "images", name))
    with open(os.path.join(sensor, "rgbFrames.txt"), "w") as f:
        f.write("timestamp[ns],frame\n")
        f.writelines(f"{frame_ts[r.frame]},{r.frame}\n" for r in kept)
    with open(os.path.join(sensor, "ctrlLog.txt"), "w") as f:
        f.write("timestamp[ns],steering,throttle\n")
        f.writelines(f"{r.ts},{int(r.s)},{int(r.t)}\n" for r in human)
    shutil.copyfile(
        os.path.join(run, "sensor_data", "indicatorLog.txt"),
        os.path.join(sensor, "indicatorLog.txt"),
    )
    with open(os.path.join(sensor, "takeover.txt"), "w") as f:
        f.write(
            f"takeover frames of {os.path.basename(run)}, autopilot_log.py extract "
            f"--trim_start {trim_start} --trim_end {trim_end}\n"
        )
        f.write("models: " + ", ".join(sorted({f"{r.direction} {r.model}" for r in kept})) + "\n")

    # Re-match like split_direction does and check every frame got its label. Matching
    # drops 0,0 labels (a stopped car), so those frames aren't expected back.
    match_frame_session(out, 1e3, redo_matching=True)
    got = {int(path.split("_")[0]): (s, t) for _, path, s, t, _ in processed_rows(out)}
    want = {f: lt for f, lt in labels.items() if lt != (0, 0)}
    assert got == want, f"{out}: re-matched labels differ ({len(got)} vs {len(want)})"


def extract(root, runs, trim_start, trim_end, dry_run):
    for run in runs:
        name = os.path.basename(run) + SUFFIX
        rows, kept, labels, human, frame_ts = plan_extract(run, trim_start, trim_end)
        segs = takeovers(rows)
        frames = {r.frame for r in kept}
        seconds = sum(sec for r, sec in zip(rows, row_seconds(rows)) if r.frame in frames)
        print(
            f"{os.path.basename(run)}: {len(segs)} takeovers, {len(human)} human frames, "
            f"{len(kept)} kept after trimming ({fmt_s(seconds)})"
        )
        where = extracted_already(root, name)
        if where:
            print(f"  skipped, already extracted: {where}")
            continue
        if not kept:
            print("  nothing to extract")
            continue
        out = os.path.join(root, "inbox", name)
        if dry_run:
            print(f"  would write {out}")
            continue
        write_takeover(run, out, kept, labels, human, frame_ts, trim_start, trim_end)
        print(f"  -> {out}")
    if not dry_run:
        print('next: python split_direction.py --note "takeovers of <models>"')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="what the models did, takeovers, plots")
    rep.add_argument("runs", nargs="*", help=f"default: everything in fwdbwd/{RUNS}/")
    rep.add_argument("--no_plot", action="store_true", help="skip autopilot_report.png")
    ext = sub.add_parser("extract", help="your takeovers into the inbox")
    ext.add_argument("runs", nargs="*", help=f"default: everything in fwdbwd/{RUNS}/")
    ext.add_argument(
        "--trim_start",
        type=float,
        default=0.3,
        help="seconds dropped at the start of each takeover (default 0.3)",
    )
    ext.add_argument(
        "--trim_end",
        type=float,
        default=0.3,
        help="seconds dropped before you let go of the stick (default 0.3)",
    )
    ext.add_argument("--dry_run", action="store_true", help="print, write nothing")
    for p in (rep, ext):
        p.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    runs = find_runs(root, args.runs)
    if args.command == "report":
        report(runs, not args.no_plot)
    else:
        if args.trim_start < 0 or args.trim_end < 0:
            raise SystemExit("--trim_start and --trim_end can't be negative")
        extract(root, runs, args.trim_start, args.trim_end, args.dry_run)


if __name__ == "__main__":
    main()
