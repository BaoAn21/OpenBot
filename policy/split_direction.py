"""Split new recordings into forward and backward driving, and into train and test.

Everything for the Autopilot Session models lives in dataset/fwdbwd/:

  inbox/                         copy new recordings from the phone here
  raw/session_<N>/<recording>/   the recordings, matched, untouched otherwise
                                 + test_recordings.txt, note.txt
  train/fwd/session_<N>/...      forward runs   (donkey_memory)
  train/bwd/session_<N>/...      backward parts (cil_mobile_fast)
  test/fwd/session_<N>/...
  test/bwd/session_<N>/...
  parked/empty/, parked/autopilot/   recordings that can't be used

A session is one batch of recordings, so you can train on any choice of sessions
(train_fwdbwd.py --sessions 1 3 4) and keep trial ones out of the final model.

Each recording is split by throttle sign:
  fwd  one folder per contiguous forward run (throttle > 0), named <recording>_fNN, so
       donkey_memory's memory windows never reach across a reverse. A run also breaks on
       a gap > MAX_GAP_S. cmd stays as recorded (0).
  bwd  one folder per recording with every throttle < 0 frame, cmd relabelled to 2
       (reverse) - what the phone feeds the backward model.
Throttle-0 frames (the ESC's F/R pause) go to neither.

Train/test is per recording, so no test frame has a neighbour in training: a random
--test_fraction of the new recordings (seeded, so a dry run and the real run agree) goes
to test, both its forward and its backward part.

Recordings without frames or controls go to parked/empty/; recordings the autopilot drove
(inferenceTime.txt has rows, so the labels are the model's own output) to
parked/autopilot/.

Each output folder gets its own raw logs (rgbFrames subset, full ctrlLog, an
indicatorLog), so re-matching from scratch - which the record builder does - reproduces
the labels; every folder is re-matched and checked. Images are hard links, so the split
costs no space. Output folders that already exist make it stop before anything is moved.

Usage (from policy/, conda env):
    python split_direction.py --dry_run
    python split_direction.py --note "daylight, both directions"     # a new session
    python split_direction.py --session 3                            # add to session 3
"""

import argparse
import os
import random
import shutil

from openbot.associate_frames import match_frame_session

POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(POLICY_DIR, "dataset", "fwdbwd")
SPLITS = ("train", "test")
DIRECTIONS = ("fwd", "bwd")
MAX_GAP_S = 0.5
REVERSE = 2


def read_lines(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines[0], [l for l in lines[1:] if l.strip()]


def data_rows(path):
    """Number of data rows in a log, 0 if it's missing or empty."""
    if not os.path.isfile(path) or not os.path.getsize(path):
        return 0
    return len(read_lines(path)[1])


def unusable_reason(recording):
    """Why a recording can't be split, or None."""
    sensor = os.path.join(recording, "sensor_data")
    images = os.path.join(recording, "images")
    if not os.path.isdir(images) or not os.listdir(images):
        return "empty"
    if data_rows(os.path.join(sensor, "rgbFrames.txt")) == 0:
        return "empty"
    if data_rows(os.path.join(sensor, "ctrlLog.txt")) == 0:
        return "empty"
    if data_rows(os.path.join(sensor, "inferenceTime.txt")) > 0:
        return "autopilot"
    return None


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


def session_numbers(root):
    raw = os.path.join(root, "raw")
    if not os.path.isdir(raw):
        return []
    return sorted(
        int(d.split("_")[1])
        for d in os.listdir(raw)
        if d.startswith("session_") and d.split("_")[1].isdigit()
    )


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
    shutil.copyfile(
        os.path.join(src, "sensor_data", "ctrlLog.txt"),
        os.path.join(dst, "sensor_data", "ctrlLog.txt"),
    )
    with open(os.path.join(dst, "sensor_data", "indicatorLog.txt"), "w") as f:
        f.write("timestamp[ns],signal\n")
        f.writelines(l + "\n" for l in indicator_lines)

    # Re-match from the new raw logs and check it reproduces exactly these labels.
    match_frame_session(dst, 1e3, redo_matching=True)
    got = processed_rows(dst)
    assert got == rows, f"{dst}: re-matched labels differ ({len(got)} vs {len(rows)})"


def plan(root, session, recordings, test):
    """(src, dst, rows, indicator_lines, split, direction) for every output folder."""
    jobs = []
    for src in sorted(recordings):
        rec = os.path.basename(src)
        split = "test" if rec in test else "train"
        out = lambda d: os.path.join(root, split, d, session)
        rows = processed_rows(src)
        _, ind = read_lines(os.path.join(src, "sensor_data", "indicatorLog.txt"))
        for k, run in enumerate(forward_runs(rows)):
            jobs.append(
                (
                    src,
                    os.path.join(out("fwd"), f"{rec}_f{k:02d}"),
                    run,
                    ind,
                    split,
                    "fwd",
                )
            )
        back = [(r[0], r[1], r[2], r[3], REVERSE) for r in rows if r[3] < 0]
        if back:
            ind_back = [f"{back[0][0] - 1},{REVERSE}"]
            jobs.append(
                (src, os.path.join(out("bwd"), rec), back, ind_back, split, "bwd")
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--session",
        type=int,
        help="add the recordings to this existing session (default: a new session)",
    )
    parser.add_argument("--note", help="what this session is, kept in its note.txt")
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="share of the recordings that go to test (default 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the train/test pick (default 0)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="match and print the split, but move nothing and write no folders",
    )
    parser.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    inbox = os.path.join(root, "inbox")
    new = sorted(
        os.path.join(inbox, d)
        for d in (os.listdir(inbox) if os.path.isdir(inbox) else [])
        if os.path.isdir(os.path.join(inbox, d))
    )
    if not new:
        raise SystemExit(f"no recordings in {inbox}")

    existing = session_numbers(root)
    if args.session is not None:
        if args.session not in existing:
            raise SystemExit(f"session {args.session} doesn't exist (have {existing})")
        number = args.session
    else:
        number = max(existing, default=0) + 1
    session = f"session_{number}"
    raw_dir = os.path.join(root, "raw", session)
    print(f"{len(new)} recordings in the inbox -> {session}")

    parked, recordings = [], []
    for rec in new:
        reason = unusable_reason(rec)
        if reason is None:
            recordings.append(rec)
        else:
            parked.append((rec, reason))
            print(f"  {os.path.basename(rec)}: {reason}, goes to parked/{reason}/")
    clash = [
        r
        for r in recordings
        if os.path.exists(os.path.join(raw_dir, os.path.basename(r)))
    ]
    if clash:
        raise SystemExit(f"{session} already has {os.path.basename(clash[0])}")

    # Matched in the inbox; only writes the matched_* files next to the logs.
    for rec in recordings:
        match_frame_session(rec, 1e3)
    names = sorted(os.path.basename(r) for r in recordings)
    test = set(
        random.Random(args.seed).sample(names, round(len(names) * args.test_fraction))
    )
    jobs = plan(root, session, recordings, test)
    taken = [j[1] for j in jobs if os.path.exists(j[1])]
    if taken:
        raise SystemExit(f"{len(taken)} output folders already exist, e.g. {taken[0]}")

    if not args.dry_run:
        for rec, reason in parked:
            os.makedirs(os.path.join(root, "parked", reason), exist_ok=True)
            shutil.move(rec, os.path.join(root, "parked", reason))
        os.makedirs(raw_dir, exist_ok=True)
        for rec in recordings:
            shutil.move(rec, raw_dir)
        moved = {r: os.path.join(raw_dir, os.path.basename(r)) for r in recordings}
        for src, dst, rows, ind, split, direction in jobs:
            write_folder(moved[src], dst, rows, ind)
        with open(os.path.join(raw_dir, "test_recordings.txt"), "a") as f:
            f.writelines(n + "\n" for n in sorted(test))
        if args.note:
            with open(os.path.join(raw_dir, "note.txt"), "a") as f:
                f.write(args.note + "\n")

    stats = {}
    for _, _, rows, _, split, direction in jobs:
        st = stats.setdefault((split, direction), [0, 0])
        st[0] += 1
        st[1] += len(rows)
    print(f"{session}, dry run, nothing written:" if args.dry_run else f"{session}:")
    for split in SPLITS:
        for direction in DIRECTIONS:
            folders, frames = stats.get((split, direction), (0, 0))
            print(f"  {split}/{direction}: {folders:4} folders, {frames:6} frames")
    print(f"test recordings ({len(test)} of {len(names)}):", sorted(test))
    if not args.dry_run:
        print("next: python train_fwdbwd.py --list")


if __name__ == "__main__":
    main()
