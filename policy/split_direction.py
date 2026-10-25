"""Split recordings that mix forward and backward driving into a forward and a backward set.

The Autopilot Session screen drives with two models (donkey_memory forward, cil_mobile_fast
backward, switched with triangle/Y), so each recording is split by throttle sign:

  fwdbwd_forward   one session folder per contiguous forward run (throttle > 0), named
                   <session>_fNN, so donkey_memory's memory windows never reach across a
                   reverse. A run also breaks on a gap > MAX_GAP_S. cmd stays as recorded (0).
  fwdbwd_backward  one session folder per recording with every throttle < 0 frame, cmd
                   relabelled to 2 (reverse) - what the phone feeds the backward model.

Throttle-0 frames (the ESC's F/R pause) go to neither.

Each output folder gets its own raw logs (rgbFrames subset, full ctrlLog, an indicatorLog),
so re-matching from scratch - which tfrecord.convert_dataset does - reproduces the labels;
every folder is re-matched and checked. Images are hard links, so the split costs no space.
Train/test is per recording: every 5th (offset 2) goes to test_data.

The raw recordings live in dataset/_other_sessions/fwdbwd_raw/<raw_set>/ and are matched
first if they aren't yet. Output goes wherever each set currently is - dataset/ or
dataset/_other_sessions/, since only one of the two is in train_data/test_data at a time -
or under _other_sessions/ if it doesn't exist yet. Output folders that already exist make
it stop, so a raw set can't be split twice.

Afterwards rebuild the records, each with its set in place:
  forward:  python -m openbot.tfrecord_mem --mem_len 3
  backward: tfrecord.convert_dataset(...) or train with --create_tf_record

Usage (from policy/, conda env):
    python split_direction.py fwdbwd_session_2 --dry_run
    python split_direction.py fwdbwd_session_2
"""

import argparse
import os

from openbot.associate_frames import match_frame_ctrl_input, match_frame_session

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "_other_sessions", "fwdbwd_raw")
FWD_SET = "fwdbwd_forward"
BWD_SET = "fwdbwd_backward"
SPLITS = ("train_data", "test_data")
MAX_GAP_S = 0.5
REVERSE = 2


def read_lines(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines[0], [l for l in lines[1:] if l.strip()]


def processed_rows(session_dir):
    _, lines = read_lines(
        os.path.join(session_dir, "sensor_data", "matched_frame_ctrl_cmd_processed.txt")
    )
    rows = []
    for l in lines:
        ts, path, steer, thr, cmd = l.split(",")
        rows.append((int(ts), os.path.basename(path), int(steer), int(thr), int(cmd)))
    return sorted(rows)


def forward_runs(rows):
    runs, cur, last_ts = [], [], None
    for r in rows:
        ts, thr = r[0], r[3]
        if thr < 0:
            if cur:
                runs.append(cur)
            cur = []
        elif thr > 0:
            if cur and (ts - last_ts) / 1e9 > MAX_GAP_S:
                runs.append(cur)
                cur = []
            cur.append(r)
            last_ts = ts
    if cur:
        runs.append(cur)
    return runs


def set_root(name):
    """dataset/ if the set is in train_data/test_data now, else dataset/_other_sessions/."""
    if os.path.isdir(os.path.join(DATASET_DIR, "train_data", name)):
        return DATASET_DIR
    return os.path.join(DATASET_DIR, "_other_sessions")


def write_folder(src, dst, rows, indicator_lines):
    os.makedirs(os.path.join(dst, "images"))
    os.makedirs(os.path.join(dst, "sensor_data"))
    for r in rows:
        os.link(os.path.join(src, "images", r[1]), os.path.join(dst, "images", r[1]))
    header, frames = read_lines(os.path.join(src, "sensor_data", "rgbFrames.txt"))
    keep = {r[0] for r in rows}
    with open(os.path.join(dst, "sensor_data", "rgbFrames.txt"), "w") as f:
        f.write(header + "\n")
        f.writelines(l + "\n" for l in frames if int(l.split(",")[0]) in keep)
    with open(os.path.join(src, "sensor_data", "ctrlLog.txt")) as fi, open(
        os.path.join(dst, "sensor_data", "ctrlLog.txt"), "w"
    ) as fo:
        fo.write(fi.read())
    with open(os.path.join(dst, "sensor_data", "indicatorLog.txt"), "w") as f:
        f.write("timestamp[ns],signal\n")
        f.writelines(l + "\n" for l in indicator_lines)

    # Re-match from the new raw logs and check it reproduces exactly these labels.
    match_frame_session(dst, 1e3, redo_matching=True)
    got = processed_rows(dst)
    assert got == rows, f"{dst}: re-matched labels differ ({len(got)} vs {len(rows)})"


def plan(raw_dir):
    """(src, dst, rows, indicator_lines, split, kind) for every output folder."""
    roots = {FWD_SET: set_root(FWD_SET), BWD_SET: set_root(BWD_SET)}
    jobs = []
    for i, s in enumerate(sorted(os.listdir(raw_dir))):
        split = "test_data" if i % 5 == 2 else "train_data"
        src = os.path.join(raw_dir, s)
        rows = processed_rows(src)
        _, ind = read_lines(os.path.join(src, "sensor_data", "indicatorLog.txt"))
        for k, run in enumerate(forward_runs(rows)):
            dst = os.path.join(roots[FWD_SET], split, FWD_SET, f"{s}_f{k:02d}")
            jobs.append((src, dst, run, ind, split, "fwd"))
        back = [(r[0], r[1], r[2], r[3], REVERSE) for r in rows if r[3] < 0]
        if back:
            dst = os.path.join(roots[BWD_SET], split, BWD_SET, s)
            jobs.append((src, dst, back, [f"{back[0][0] - 1},{REVERSE}"], split, "bwd"))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "raw_set", help=f"folder under {RAW_DIR}, e.g. fwdbwd_session_2"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="match the raw set and print the split, but write no output folders",
    )
    args = parser.parse_args()

    raw_dir = os.path.join(RAW_DIR, args.raw_set)
    # Skips recordings that are already matched; writes only the matched_* files.
    match_frame_ctrl_input(RAW_DIR, [args.raw_set], 1e3)
    jobs = plan(raw_dir)
    existing = [j[1] for j in jobs if os.path.exists(j[1])]
    if existing:
        raise SystemExit(
            f"{len(existing)} output folders already exist, e.g. {existing[0]}"
        )

    stats = {}
    for src, dst, rows, ind, split, kind in jobs:
        if not args.dry_run:
            write_folder(src, dst, rows, ind)
        st = stats.setdefault((split, kind), [0, 0])
        st[0] += 1
        st[1] += len(rows)
    for (split, kind), (folders, frames) in sorted(stats.items()):
        print(f"{split:10} {kind}: {folders:4} folders, {frames:6} frames")
    sessions = sorted(os.listdir(raw_dir))
    print("test recordings:", [s for i, s in enumerate(sessions) if i % 5 == 2])
    print(
        f"forward set under {set_root(FWD_SET)}, backward set under {set_root(BWD_SET)}"
    )


if __name__ == "__main__":
    main()
