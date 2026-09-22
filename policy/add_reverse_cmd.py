"""
Re-purpose the indicator ("cmd") signal as a forward/reverse command.

The autopilot model already takes a high-level command input, normally the turn
indicator (-1 = left, 0 = none, 1 = right). Since the dataset contains no turn
signals at all (every indicatorLog.txt holds a single "0"), one of those values
can be reused to mean "drive backward", so a single pilot_net can cover both
directions of the same track instead of needing a sequence model to infer the
direction from motion.

This script reads each session's ctrlLog.txt, finds the stretches where throttle
is negative, and writes the corresponding signal events into indicatorLog.txt:

    <first negative throttle> - lead   ->  REVERSE signal
    <throttle turns positive again>    ->  0

The lead mirrors real driving: you flick the indicator first, then pull the
throttle back. It also means that at inference time the model sees the command
change a moment before it is expected to act on it, exactly as in training.

indicatorLog.txt is an event log - associate_frames.py matches each frame to the
last event at or before it (max_offset is 1 us) - so a single event stays in
effect until the next one.

The original file is preserved as indicatorLog.txt.orig on first run and is the
source for every subsequent run, so the script is idempotent and can be re-run
with a different --lead-ms, or undone with --restore.
"""

import argparse
import csv
import os
import shutil

HEADER = ["timestamp[ns]", "signal"]
INDICATOR_LOG = "indicatorLog.txt"
BACKUP_SUFFIX = ".orig"
# Derived files that embed the cmd column and must be rebuilt after a change.
DERIVED_FILES = ["matched_frame_ctrl_cmd.txt", "matched_frame_ctrl_cmd_processed.txt"]


def list_sessions(data_dir, splits):
    """Yield every session directory that has a sensor_data/ctrlLog.txt."""
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for dataset in sorted(os.listdir(split_dir)):
            dataset_dir = os.path.join(split_dir, dataset)
            if not os.path.isdir(dataset_dir) or dataset.startswith("."):
                continue
            for session in sorted(os.listdir(dataset_dir)):
                sensor_dir = os.path.join(dataset_dir, session, "sensor_data")
                if os.path.isfile(os.path.join(sensor_dir, "ctrlLog.txt")):
                    yield sensor_dir


def read_ctrl(path):
    """Read ctrlLog.txt as a timestamp-sorted list of (timestamp, throttle)."""
    samples = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            samples.append((int(row[0]), int(row[2])))
    samples.sort()
    return samples


def read_events(path):
    """Read an indicator log as a timestamp-sorted list of (timestamp, signal)."""
    events = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            events.append((int(row[0]), int(row[1])))
    events.sort()
    return events


def find_reverse_windows(samples):
    """Find the (start_ts, end_ts) of every reverse manoeuvre.

    A window starts at the first negative-throttle sample and ends at the first
    strictly positive sample after it; zeros in between are kept inside the
    window, so a throttle that momentarily reads 0 while reversing does not
    split one manoeuvre into two (and does not make the signal flap off and on).
    end_ts is None when the session ends while still reversing.
    """
    windows = []
    i = 0
    while i < len(samples):
        if samples[i][1] >= 0:
            i += 1
            continue
        start_ts = samples[i][0]
        j = i + 1
        while j < len(samples) and samples[j][1] <= 0:
            j += 1
        end_ts = samples[j][0] if j < len(samples) else None
        windows.append((start_ts, end_ts))
        i = j
    return windows


def build_events(existing, windows, signal, lead_ns):
    """Merge the reverse windows into the existing indicator events.

    Existing events that fall inside a reverse window are dropped - they would
    cancel the reverse signal partway through - and reported back to the caller
    so a real turn signal never disappears silently.
    """
    inserted = []
    for start_ts, end_ts in windows:
        inserted.append((start_ts - lead_ns, signal))
        if end_ts is not None:
            inserted.append((end_ts, 0))

    dropped = []
    kept = []
    for ts, value in existing:
        in_window = any(
            (start_ts - lead_ns) < ts and (end_ts is None or ts < end_ts)
            for start_ts, end_ts in windows
        )
        if in_window:
            dropped.append((ts, value))
        else:
            kept.append((ts, value))

    events = sorted(kept + inserted)

    # Keep timestamps strictly increasing (a long lead can reach back past the
    # preceding event) and drop events that do not change the signal.
    result = []
    for ts, value in events:
        if result and ts <= result[-1][0]:
            ts = result[-1][0] + 1
        if result and value == result[-1][1]:
            continue
        result.append((ts, value))
    return result, dropped


def write_events(path, events):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for ts, value in events:
            writer.writerow([ts, value])


def restore(sensor_dir):
    backup = os.path.join(sensor_dir, INDICATOR_LOG + BACKUP_SUFFIX)
    if not os.path.isfile(backup):
        return False
    shutil.copyfile(backup, os.path.join(sensor_dir, INDICATOR_LOG))
    os.remove(backup)
    return True


def clean_derived(sensor_dir, dry_run):
    """Remove the matched files that embed the cmd column so they get rebuilt."""
    removed = []
    for name in DERIVED_FILES:
        path = os.path.join(sensor_dir, name)
        if os.path.isfile(path):
            if not dry_run:
                os.remove(path)
            removed.append(name)
    return removed


def process(sensor_dir, signal, lead_ns, dry_run, keep_derived):
    log_path = os.path.join(sensor_dir, INDICATOR_LOG)
    backup_path = log_path + BACKUP_SUFFIX
    source_path = backup_path if os.path.isfile(backup_path) else log_path

    samples = read_ctrl(os.path.join(sensor_dir, "ctrlLog.txt"))
    windows = find_reverse_windows(samples)
    if not windows:
        return 0

    existing = read_events(source_path)
    events, dropped = build_events(existing, windows, signal, lead_ns)

    session = os.path.dirname(sensor_dir)
    print(f"{session}: {len(windows)} reverse window(s)")
    for start_ts, end_ts in windows:
        span = "to end of session" if end_ts is None else f"{(end_ts - start_ts) / 1e9:.2f}s"
        print(f"  signal {signal} at -{lead_ns / 1e6:.0f}ms before {start_ts}, {span}")
    for ts, value in dropped:
        print(f"  dropped existing event ({ts},{value}) inside a reverse window")

    if dry_run:
        return len(windows)

    if not os.path.isfile(backup_path):
        shutil.copyfile(log_path, backup_path)
    write_events(log_path, events)
    if not keep_derived:
        clean_derived(sensor_dir, dry_run)
    return len(windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset", help="dataset root")
    parser.add_argument(
        "--splits", default="train_data,test_data", help="comma-separated splits"
    )
    parser.add_argument(
        "--signal",
        type=int,
        default=1,
        choices=[-1, 1],
        help="indicator value that means reverse (1 = right, -1 = left)",
    )
    parser.add_argument(
        "--lead-ms",
        type=float,
        default=500.0,
        help="how long before the first negative throttle the signal is raised",
    )
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    parser.add_argument(
        "--restore", action="store_true", help="undo: put the .orig logs back"
    )
    parser.add_argument(
        "--keep-derived",
        action="store_true",
        help="keep matched_frame_ctrl_cmd*.txt instead of removing them",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    sessions = list(list_sessions(args.data_dir, splits))

    if args.restore:
        restored = sum(1 for s in sessions if restore(s))
        print(f"Restored {restored} indicator log(s) from {BACKUP_SUFFIX}")
        return

    lead_ns = int(args.lead_ms * 1e6)
    changed = sum(
        1
        for s in sessions
        if process(s, args.signal, lead_ns, args.dry_run, args.keep_derived)
    )
    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb} {changed} of {len(sessions)} session(s); signal {args.signal} = reverse")
    if not args.dry_run and not args.keep_derived:
        print("Removed matched_frame_ctrl_cmd*.txt - rerun matching to rebuild them.")


if __name__ == "__main__":
    main()
