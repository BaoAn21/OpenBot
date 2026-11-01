"""
Tfrecords for the donkeycar-style memory policy (models.donkey_memory).

Ported from donkeycar's KerasMemory (donkeycar/parts/keras.py) and the Collator that
feeds it (donkeycar/pipeline/types.py). Each example is the current image plus the
recorded (steering, throttle) of the mem_len frames before it, and the label is the
current frame's control:

    mem = [s(t-N), t(t-N), ..., s(t-1), t(t-1)]    oldest -> newest, normalized / 255

Like donkeycar, which only records while the throttle is non-zero, the rows come
from the processed labels with the stationary frames already removed, so a window
runs straight across a stop. Also like donkeycar, a frame without a full history
behind it (the first mem_len rows of each session) is dropped rather than padded.
With max_gap_s a session is first cut wherever two rows are further apart than that,
and each piece is windowed on its own, so no memory reaches across a pause.

Usage:
    python -m openbot.tfrecord_mem --mem_len 3
"""

import argparse
import os

import tensorflow as tf

from . import dataset_dir, utils
from .tfrecord_seq import (
    IMAGE_SUFFIX,
    PROCESSED_FILE_NAME,
    _bytes_feature,
    _float_feature,
    _raw_bytes_feature,
    read_processed_labels,
    session_image_ids,
)

DEFAULT_MEM_LEN = 3


def tfrecords_dir_name(mem_len):
    return f"tfrecords_mem{mem_len}"


def build_windows(rows, mem_len):
    """(label_row, history_rows) for every row with mem_len rows before it."""
    return [(rows[k], rows[k - mem_len : k]) for k in range(mem_len, len(rows))]


def read_timestamps(session_dir):
    """Frame id -> timestamp (ns), from the same processed labels as the rows."""
    stamps = {}
    with open(os.path.join(session_dir, "sensor_data", PROCESSED_FILE_NAME)) as f:
        f.readline()  # discard header
        for line in f:
            fields = line.strip().split(",")
            if len(fields) < 2:
                continue
            basename = os.path.basename(fields[1].replace("\\", "/"))
            stamps[int(basename.split("_")[0])] = int(fields[0])
    return stamps


def split_at_gaps(rows, timestamps, max_gap_s):
    """rows cut into runs wherever consecutive rows are more than max_gap_s apart."""
    runs, current = [], []
    for row in rows:
        if current and (timestamps[row[0]] - timestamps[current[-1][0]]) / 1e9 > max_gap_s:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def make_example(session_dir, label_row, history_rows):
    frame_id, steering, throttle, cmd = label_row
    image_path = os.path.join(session_dir, "images", f"{frame_id}{IMAGE_SUFFIX}")
    with open(image_path, "rb") as f:
        image = f.read()

    mem = []
    for _, past_steering, past_throttle, _ in history_rows:
        mem += [past_steering / 255.0, past_throttle / 255.0]

    feature = {
        "image": _raw_bytes_feature(image),
        "path": _bytes_feature(image_path),
        # Named left/right like every other record here; under the Ackermann
        # convention they hold steering and throttle.
        "left": _float_feature(steering / 255.0),
        "right": _float_feature(throttle / 255.0),
        "cmd": _float_feature(float(cmd)),
        "mem": tf.train.Feature(float_list=tf.train.FloatList(value=mem)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_dataset(
    data_dir, tfrecords_dir, tfrecords_name, mem_len, verbose=True, max_gap_s=None
):
    """Write one tfrecord covering every session under data_dir. max_gap_s (seconds)
    cuts the windows at pauses, see the module docstring; None windows across them."""
    os.makedirs(tfrecords_dir, exist_ok=True)
    out_path = os.path.join(tfrecords_dir, tfrecords_name)

    total = session_count = 0
    with tf.io.TFRecordWriter(out_path) as writer:
        for dataset in sorted(utils.list_dirs(data_dir)):
            for session in sorted(utils.list_dirs(os.path.join(data_dir, dataset))):
                session_dir = os.path.join(data_dir, dataset, session)
                available = session_image_ids(session_dir)
                rows = [
                    r for r in read_processed_labels(session_dir) if r[0] in available
                ]
                if max_gap_s is None:
                    windows = build_windows(rows, mem_len)
                else:
                    timestamps = read_timestamps(session_dir)
                    windows = [
                        window
                        for run in split_at_gaps(rows, timestamps, max_gap_s)
                        for window in build_windows(run, mem_len)
                    ]
                if not windows:
                    print(f"  skipping {dataset}/{session}: no usable labels")
                    continue

                for label_row, history_rows in windows:
                    writer.write(
                        make_example(
                            session_dir, label_row, history_rows
                        ).SerializeToString()
                    )

                session_count += 1
                total += len(windows)
                if verbose:
                    print(f"  {dataset}/{session}: {len(windows):4d} examples")

    print(f"{out_path}: {session_count} sessions, {total} examples")
    return total


def make_parse_fn(mem_len):
    """Parser emitting features["image"] (decoded, float in [0, 1]) and
    features["mem"] of shape (2 * mem_len,)."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "left": tf.io.FixedLenFeature([], tf.float32),
        "right": tf.io.FixedLenFeature([], tf.float32),
        "cmd": tf.io.FixedLenFeature([], tf.float32),
        "mem": tf.io.FixedLenFeature([2 * mem_len], tf.float32),
    }

    def parse(example):
        parsed = tf.io.parse_single_example(example, feature_description)
        parsed["image"] = tf.image.convert_image_dtype(
            tf.io.decode_jpeg(parsed["image"], channels=3), tf.float32
        )
        return parsed

    return parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build tfrecords for the donkeycar-style memory policy"
    )
    parser.add_argument(
        "--mem_len",
        type=int,
        default=DEFAULT_MEM_LEN,
        help=f"previous controls fed back to the model (default: {DEFAULT_MEM_LEN})",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=dataset_dir,
        help="dataset root holding train_data/ and test_data/ (read only)",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.dataset_dir, tfrecords_dir_name(args.mem_len))
    print(f"Building memory records (mem_len {args.mem_len}) into {out_dir}")
    for split, name in (("train_data", "train.tfrec"), ("test_data", "test.tfrec")):
        print(f"{split}:")
        convert_dataset(
            os.path.join(args.dataset_dir, split), out_dir, name, args.mem_len
        )
