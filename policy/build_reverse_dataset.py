"""
One-off helper: build a reverse-only copy of the collected dataset for a quick
"can the car go backward" training test. Reads matched_frame_ctrl_cmd_processed.txt
(and the two upstream matched files) from the existing dataset/{train_data,test_data}
sessions, keeps only frames where throttle (the "right" column) is negative, and
writes a fully separate dataset tree at dataset_reverse/. Nothing under dataset/
is read for writing and nothing there is modified.
"""

import csv
import os
import shutil

SRC_ROOT = "dataset"
DST_ROOT = "dataset_reverse"
SPLITS = ["train_data", "test_data"]

MATCHED_FILES = [
    # (filename, frame_field_index) - matched_frame_ctrl.txt has header
    # "timestamp (frame),time_offset (ctrl-frame),frame,left,right" (frame at index 2);
    # matched_frame_ctrl_cmd.txt has header
    # "timestamp (frame),time_offset (cmd-frame),time_offset (ctrl-frame),frame,left,right,cmd"
    # (frame at index 3).
    ("matched_frame_ctrl.txt", 2),
    ("matched_frame_ctrl_cmd.txt", 3),
]
PROCESSED_FILE = "matched_frame_ctrl_cmd_processed.txt"


def filter_processed(src_path, dst_path, dst_images_dir):
    """Filter matched_frame_ctrl_cmd_processed.txt to throttle<0 rows, rewriting
    the image path to the new dataset location. Returns the set of frame ids kept
    (parsed from the image filename, e.g. "813" from ".../813_crop.jpeg")."""
    kept_frame_ids = set()
    with open(src_path, newline="") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        rows = list(reader)

    kept_rows = []
    for row in rows:
        if len(row) < 5:
            continue
        timestamp, frame_path, left, right, cmd = row[:5]
        if int(right) < 0:
            kept_rows.append(row)
            fname = os.path.basename(frame_path)
            kept_frame_ids.add(fname.split("_crop")[0])

    if not kept_rows:
        return kept_frame_ids

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    os.makedirs(dst_images_dir, exist_ok=True)
    with open(dst_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        for row in kept_rows:
            timestamp, frame_path, left, right, cmd = row[:5]
            fname = os.path.basename(frame_path)
            new_frame_path = os.path.join(dst_images_dir, fname)
            writer.writerow([timestamp, new_frame_path, left, right, cmd])

    return kept_frame_ids


def filter_matched(src_path, dst_path, kept_frame_ids, frame_field_index):
    """Filter matched_frame_ctrl.txt / matched_frame_ctrl_cmd.txt to rows whose
    frame id is in kept_frame_ids."""
    with open(src_path, newline="") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        rows = list(reader)

    kept_rows = [
        r
        for r in rows
        if len(r) > frame_field_index and r[frame_field_index] in kept_frame_ids
    ]
    if not kept_rows:
        return

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(kept_rows)


def main():
    if os.path.exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)

    total_sessions = 0
    total_kept_sessions = 0
    total_frames = 0

    for split in SPLITS:
        src_split_dir = os.path.join(SRC_ROOT, split)
        if not os.path.isdir(src_split_dir):
            continue
        for dataset_name in sorted(os.listdir(src_split_dir)):
            src_dataset_dir = os.path.join(src_split_dir, dataset_name)
            if not os.path.isdir(src_dataset_dir):
                continue
            for session in sorted(os.listdir(src_dataset_dir)):
                src_session_dir = os.path.join(src_dataset_dir, session)
                src_sensor_dir = os.path.join(src_session_dir, "sensor_data")
                src_processed = os.path.join(src_sensor_dir, PROCESSED_FILE)
                if not os.path.isfile(src_processed):
                    continue
                total_sessions += 1

                dst_session_dir = os.path.join(DST_ROOT, split, dataset_name, session)
                dst_sensor_dir = os.path.join(dst_session_dir, "sensor_data")
                dst_images_dir = os.path.join(dst_session_dir, "images")
                dst_processed = os.path.join(dst_sensor_dir, PROCESSED_FILE)

                kept_frame_ids = filter_processed(
                    src_processed, dst_processed, dst_images_dir
                )
                if not kept_frame_ids:
                    continue

                for fname, frame_field_index in MATCHED_FILES:
                    src_f = os.path.join(src_sensor_dir, fname)
                    if os.path.isfile(src_f):
                        filter_matched(
                            src_f,
                            os.path.join(dst_sensor_dir, fname),
                            kept_frame_ids,
                            frame_field_index,
                        )

                src_images_dir = os.path.join(src_session_dir, "images")
                for frame_id in kept_frame_ids:
                    fname = f"{frame_id}_crop.jpeg"
                    src_img = os.path.join(src_images_dir, fname)
                    if os.path.isfile(src_img):
                        shutil.copy2(src_img, os.path.join(dst_images_dir, fname))

                total_kept_sessions += 1
                total_frames += len(kept_frame_ids)
                print(f"{src_session_dir}: kept {len(kept_frame_ids)} reverse frames")

    print(
        f"\nDone. {total_kept_sessions}/{total_sessions} sessions had reverse frames, "
        f"{total_frames} reverse frames total copied to {DST_ROOT}/"
    )


if __name__ == "__main__":
    main()
