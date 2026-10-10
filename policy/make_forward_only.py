"""Strip reverse driving from the sessions in dataset/train_data and dataset/test_data.

For every session, rewrites the two label files the tfrecord builders read:
  matched_frame_ctrl_cmd.txt            (used with --trim_stationary)
  matched_frame_ctrl_cmd_processed.txt  (used without it)
dropping every frame with throttle < 0 and setting cmd to 0 on the rest.

The originals are copied to <name>.with_reverse on the first run and always read
from there, so running it twice gives the same result. --restore puts them back.

Usage (from policy/):
    python make_forward_only.py --dry_run
    python make_forward_only.py
    python make_forward_only.py --restore
"""

import argparse
import glob
import os
import shutil

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
BACKUP_SUFFIX = ".with_reverse"

# file name -> (throttle column, cmd column); the "left,right" columns hold
# steering,throttle
LABEL_FILES = {
    "matched_frame_ctrl_cmd.txt": (5, 6),
    "matched_frame_ctrl_cmd_processed.txt": (3, 4),
}


def forward_only(lines, throttle_col, cmd_col):
    header, rows = lines[0], lines[1:]
    kept = [header]
    dropped = 0
    for line in rows:
        fields = line.rstrip("\n").split(",")
        if len(fields) <= cmd_col:
            continue
        if int(fields[throttle_col]) < 0:
            dropped += 1
            continue
        fields[cmd_col] = "0"
        kept.append(",".join(fields) + "\n")
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry_run", action="store_true", help="only print counts")
    parser.add_argument("--restore", action="store_true", help="undo, from backups")
    args = parser.parse_args()

    totals = {name: [0, 0] for name in LABEL_FILES}  # kept, dropped
    sessions = sorted(
        glob.glob(os.path.join(DATASET_DIR, "t*_data", "*", "*", "sensor_data"))
    )
    for sensor_dir in sessions:
        for name, (throttle_col, cmd_col) in LABEL_FILES.items():
            path = os.path.join(sensor_dir, name)
            backup = path + BACKUP_SUFFIX

            if args.restore:
                if os.path.isfile(backup):
                    shutil.move(backup, path)
                continue
            if not os.path.isfile(path) and not os.path.isfile(backup):
                print("missing", path)
                continue

            with open(backup if os.path.isfile(backup) else path) as f:
                lines = f.readlines()
            kept, dropped = forward_only(lines, throttle_col, cmd_col)
            totals[name][0] += len(kept) - 1
            totals[name][1] += dropped

            if args.dry_run:
                continue
            if not os.path.isfile(backup):
                shutil.copy2(path, backup)
            with open(path, "w") as f:
                f.writelines(kept)

    if args.restore:
        print(f"Restored {len(sessions)} sessions.")
        return
    for name, (kept, dropped) in totals.items():
        print(f"{name}: kept {kept}, dropped {dropped} reverse frames")
    print(f"{len(sessions)} sessions" + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
