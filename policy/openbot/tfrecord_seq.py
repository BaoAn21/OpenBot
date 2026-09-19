"""
Multi-frame (sequence) tfrecord support for the autopilot policy.

A single-frame model cannot tell how fast - or in which direction - the robot is
currently travelling, because a still image carries no motion. Stacking several past
frames makes apparent motion visible to the network, which is what lets one policy
drive both forward and backward.

Each example holds one jpeg per offset, ordered oldest -> newest, plus the control
label of the *newest* frame. At parse time the frames are decoded and concatenated
along the channel axis, so an image is (height, width, 3 * len(offsets)) and every
downstream consumer that just reads features["image"] keeps working unchanged.

Offsets
-------
Offsets are measured in timeline steps back from the current frame, starting at 0.
Without trimming a step is one recorded frame (~33ms at the ~30fps this dataset was
captured at), so `--seq_len 5 --seq_stride 2` reaches 8 frames (~264ms) back.

Trimming
--------
Sessions in this dataset are: stationary, one forward run, stationary at the
turnaround, one reverse run, stationary. The stationary runs are long - typically ~35
frames at the start and 10-21 at the turnaround - which is a problem for a short
stack, because a window sitting at the turnaround then contains nothing but a still
scene, indistinguishable from the still scene at the start of a session where the
correct action is the opposite one.

`--trim_stationary` rewrites the timeline to fix that: leading and trailing stationary
runs are dropped entirely, and interior ones are cut down to `--keep_stationary`
frames. Those few kept frames matter and are not a rounding error - they are what
teaches the policy to stop before it reverses, and they give it the "was moving
forward, now stopped" input that triggers the reversal. Dropping them completely
would ask the model to jump from full forward to full reverse in a single step.

Nothing under dataset/ is modified; trimming happens while the tfrecord is built.

Usage:
    python -m openbot.tfrecord_seq --seq_len 5 --seq_stride 2
    python -m openbot.tfrecord_seq --seq_len 5 --seq_stride 2 --trim_stationary
    python -m openbot.tfrecord_seq --seq_offsets 0,2,5,12,24
"""

import argparse
import os
import tensorflow as tf

from . import dataset_dir, utils


# Dilated by default, so an untrimmed stack still reaches over the turnaround stop.
# With --trim_stationary that stop is gone and uniform spacing is the better choice.
DEFAULT_SEQ_OFFSETS = (0, 2, 5, 12, 24)
DEFAULT_KEEP_STATIONARY = 2

# Labels after preprocessing: stationary frames have already been removed from this
# one, which is why trimming reads the file below instead.
PROCESSED_FILE_NAME = "matched_frame_ctrl_cmd_processed.txt"
# Labels before stationary frames were removed: every matched frame, contiguous.
MATCHED_FILE_NAME = "matched_frame_ctrl_cmd.txt"

IMAGE_SUFFIX = "_crop.jpeg"


def parse_offsets(spec):
    """Turn "0,2,5,12,24" into a validated tuple, newest (0) first."""
    if isinstance(spec, str):
        values = [int(v) for v in spec.replace(" ", "").split(",") if v != ""]
    else:
        values = [int(v) for v in spec]

    if not values:
        raise ValueError("no offsets given")
    if values[0] != 0:
        raise ValueError(f"offsets must start at 0 (the current frame), got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"offsets must be distinct, got {values}")
    if sorted(values) != values:
        raise ValueError(f"offsets must increase, got {values}")
    return tuple(values)


def uniform_offsets(seq_len, seq_stride):
    return tuple(k * seq_stride for k in range(seq_len))


def offsets_tag(offsets):
    return "-".join(str(o) for o in offsets)


def tfrecords_dir_name(offsets, trim_stationary=False, keep_stationary=None):
    """Directory name encoding how the windows were built, so records made with
    different settings can never be silently reused for one another."""
    name = f"tfrecords_seq{offsets_tag(offsets)}"
    if trim_stationary:
        name += f"_trim{keep_stationary}"
    return name


def _raw_bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature(value: str):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _float_feature(value: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _read_label_file(path, frame_field, left_field, right_field, cmd_field):
    """Read one label file into rows of (frame_id, left, right, cmd), sorted by id.

    The frame column is not trusted for its directory part: datasets recorded through
    the docker workflow have it baked in as an absolute /workspace/... path that does
    not exist outside that container. Only the frame id is taken from it.
    """
    if not os.path.isfile(path):
        return []

    needed = max(frame_field, left_field, right_field, cmd_field) + 1
    rows = []
    with open(path) as f:
        f.readline()  # discard header
        for line in f:
            fields = [v.strip() for v in line.strip().split(",") if v.strip() != ""]
            if len(fields) < needed:
                continue
            basename = os.path.basename(fields[frame_field].replace("\\", "/"))
            frame_id = int(basename.split("_")[0])
            rows.append(
                (
                    frame_id,
                    int(fields[left_field]),
                    int(fields[right_field]),
                    int(fields[cmd_field]),
                )
            )

    rows.sort()
    return rows


def read_processed_labels(session_dir):
    """Labels with stationary frames already removed (timestamp,frame,left,right,cmd)."""
    return _read_label_file(
        os.path.join(session_dir, "sensor_data", PROCESSED_FILE_NAME), 1, 2, 3, 4
    )


def read_matched_labels(session_dir):
    """Every matched frame, stationary ones included.

    Columns are timestamp,time_offset(cmd),time_offset(ctrl),frame,left,right,cmd.
    """
    return _read_label_file(
        os.path.join(session_dir, "sensor_data", MATCHED_FILE_NAME), 3, 4, 5, 6
    )


def is_stationary(row):
    """True when the robot was not moving on this frame.

    Throttle alone decides it: with Ackermann-style control a steering value with no
    throttle does not move the robot anywhere.
    """
    return row[2] == 0


def runs_of(rows):
    """Run-length encode the motion state, as [(state, count), ...] for reporting."""
    runs = []
    for row in rows:
        state = "0" if is_stationary(row) else ("+" if row[2] > 0 else "-")
        if runs and runs[-1][0] == state:
            runs[-1][1] += 1
        else:
            runs.append([state, 1])
    return runs


def trim_stationary_rows(rows, keep_stationary=DEFAULT_KEEP_STATIONARY):
    """Drop leading and trailing stationary runs; cut interior ones down to size.

    The first `keep_stationary` frames of an interior run are the ones kept, not the
    last: they are the frames immediately following the deceleration, so they carry
    straight on from the motion before them.
    """
    if not rows:
        return []

    first_moving = next((i for i, r in enumerate(rows) if not is_stationary(r)), None)
    if first_moving is None:
        return []  # the robot never moved in this session
    last_moving = max(i for i, r in enumerate(rows) if not is_stationary(r))

    trimmed = []
    run_length = 0
    for row in rows[first_moving : last_moving + 1]:
        if is_stationary(row):
            run_length += 1
            if run_length > keep_stationary:
                continue
        else:
            run_length = 0
        trimmed.append(row)

    return trimmed


def session_image_ids(session_dir):
    """Every frame id with an image on disk for this session.

    Deliberately not the same set as the labelled rows: preprocessing drops the
    stationary frames from the labels but their images are still on disk, and those
    frames make perfectly good history - they only need to be looked at, not predicted
    from. Reading history off the image directory keeps the windows that span the
    turnaround, and gives real history to frames at the start of a session instead of
    padding them.
    """
    images_dir = os.path.join(session_dir, "images")
    if not os.path.isdir(images_dir):
        return set()
    ids = set()
    for name in os.listdir(images_dir):
        if name.endswith(IMAGE_SUFFIX):
            try:
                ids.add(int(name[: -len(IMAGE_SUFFIX)]))
            except ValueError:
                continue
    return ids


def build_windows(timeline_ids, rows, offsets):
    """Build the frame windows for one session.

    `timeline_ids` is the ordered list of frame ids a window may step back through -
    every recorded image when not trimming, or the trimmed sequence when trimming, so
    that removed frames are stepped over rather than stepped on. Offsets count
    positions along that timeline, not raw frame ids.

    Returns (frame_ids, label_row) pairs with frame_ids ordered oldest -> newest.

    Reaching back past the start of the timeline repeats the oldest frame available.
    That case is worth keeping rather than dropping: it is the state a live robot is in
    on its first frames, a buffer holding no motion, and those samples are labelled
    "drive forward", which is what we want on a cold start.
    """
    position = {frame_id: i for i, frame_id in enumerate(timeline_ids)}

    windows = []
    for row in rows:
        current = position.get(row[0])
        if current is None:
            # Labelled frame missing from the timeline - no image on disk for it.
            continue
        ids = [timeline_ids[max(0, current - offset)] for offset in offsets]
        ids.reverse()  # oldest -> newest
        windows.append((ids, row))

    return windows


def make_example(session_dir, frame_ids, label_row):
    images_dir = os.path.join(session_dir, "images")
    feature = {}
    for slot, frame_id in enumerate(frame_ids):
        image_path = os.path.join(images_dir, f"{frame_id}{IMAGE_SUFFIX}")
        with open(image_path, "rb") as f:
            # Stored as the original jpeg bytes rather than decoded and re-encoded,
            # which keeps the build fast and lossless.
            feature[f"image_{slot}"] = _raw_bytes_feature(f.read())

    frame_id, left, right, cmd = label_row
    feature["path"] = _bytes_feature(
        os.path.join(images_dir, f"{frame_id}{IMAGE_SUFFIX}")
    )
    feature["left"] = _float_feature(float(left) / 255.0)
    feature["right"] = _float_feature(float(right) / 255.0)
    feature["cmd"] = _float_feature(float(cmd))

    return tf.train.Example(features=tf.train.Features(feature=feature))


def session_rows_and_timeline(
    session_dir, trim_stationary=False, keep_stationary=DEFAULT_KEEP_STATIONARY
):
    """The labelled rows to emit and the timeline their windows step back through."""
    if trim_stationary:
        rows = trim_stationary_rows(read_matched_labels(session_dir), keep_stationary)
        # The trimmed sequence is the timeline: stepping back one offset steps over
        # whatever was removed, which is the whole point of trimming.
        available = session_image_ids(session_dir)
        rows = [r for r in rows if r[0] in available]
        return rows, [r[0] for r in rows]

    rows = read_processed_labels(session_dir)
    return rows, sorted(session_image_ids(session_dir))


def convert_dataset(
    data_dir,
    tfrecords_dir,
    tfrecords_name,
    offsets,
    trim_stationary=False,
    keep_stationary=DEFAULT_KEEP_STATIONARY,
    verbose=True,
):
    """Write one tfrecord covering every session under data_dir."""
    os.makedirs(tfrecords_dir, exist_ok=True)
    out_path = os.path.join(tfrecords_dir, tfrecords_name)

    total_windows = total_padded = session_count = 0

    with tf.io.TFRecordWriter(out_path) as writer:
        for dataset in sorted(utils.list_dirs(data_dir)):
            for session in sorted(utils.list_dirs(os.path.join(data_dir, dataset))):
                session_dir = os.path.join(data_dir, dataset, session)
                rows, timeline = session_rows_and_timeline(
                    session_dir, trim_stationary, keep_stationary
                )
                if not rows or not timeline:
                    print(f"  skipping {dataset}/{session}: no usable labels")
                    continue

                windows = build_windows(timeline, rows, offsets)
                padded = sum(1 for ids, _ in windows if ids[0] == ids[1])
                for frame_ids, label_row in windows:
                    writer.write(
                        make_example(
                            session_dir, frame_ids, label_row
                        ).SerializeToString()
                    )

                session_count += 1
                total_windows += len(windows)
                total_padded += padded
                if verbose:
                    shape = " ".join(f"{s}x{n}" for s, n in runs_of(rows))
                    print(
                        f"  {dataset}/{session}: {len(windows):4d} windows "
                        f"({padded} start-padded)  [{shape}]"
                    )

    print(
        f"{out_path}: {session_count} sessions, {total_windows} windows "
        f"({total_padded} start-padded)"
    )
    return total_windows


def make_parse_fn(seq_len):
    """Build a parser for records with seq_len stacked frames.

    Emits features["image"] with shape (height, width, 3 * seq_len), frames ordered
    oldest -> newest along the channel axis.
    """
    feature_description = {
        f"image_{i}": tf.io.FixedLenFeature([], tf.string) for i in range(seq_len)
    }
    feature_description.update(
        {
            "path": tf.io.FixedLenFeature([], tf.string),
            "left": tf.io.FixedLenFeature([], tf.float32),
            "right": tf.io.FixedLenFeature([], tf.float32),
            "cmd": tf.io.FixedLenFeature([], tf.float32),
        }
    )

    def parse(example):
        parsed = tf.io.parse_single_example(example, feature_description)
        frames = [
            tf.image.convert_image_dtype(
                tf.io.decode_jpeg(parsed[f"image_{i}"], channels=3), tf.float32
            )
            for i in range(seq_len)
        ]
        parsed["image"] = tf.concat(frames, axis=-1)
        return parsed

    return parse


def resolve_offsets(seq_offsets=None, seq_len=None, seq_stride=None):
    if seq_offsets:
        return parse_offsets(seq_offsets)
    if seq_len:
        return uniform_offsets(seq_len, seq_stride or 1)
    return DEFAULT_SEQ_OFFSETS


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build multi-frame tfrecords for the autopilot policy"
    )
    parser.add_argument(
        "--seq_offsets",
        type=str,
        default=None,
        help="comma separated offsets back from the current frame, starting at 0 "
        f"(default: {','.join(str(o) for o in DEFAULT_SEQ_OFFSETS)})",
    )
    parser.add_argument(
        "--seq_len", type=int, default=None, help="uniform windows: number of frames"
    )
    parser.add_argument(
        "--seq_stride", type=int, default=None, help="uniform windows: frame gap"
    )
    parser.add_argument(
        "--trim_stationary",
        action="store_true",
        help="drop leading and trailing stationary runs and cut interior ones down "
        "to --keep_stationary frames (dataset/ is not modified)",
    )
    parser.add_argument(
        "--keep_stationary",
        type=int,
        default=DEFAULT_KEEP_STATIONARY,
        help="frames to keep from an interior stationary run when trimming "
        f"(default: {DEFAULT_KEEP_STATIONARY})",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=dataset_dir,
        help="dataset root holding train_data/ and test_data/ (read only)",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    offsets = resolve_offsets(args.seq_offsets, args.seq_len, args.seq_stride)

    out_dir = os.path.join(
        args.dataset_dir,
        tfrecords_dir_name(offsets, args.trim_stationary, args.keep_stationary),
    )
    print(
        f"Building {len(offsets)}-frame windows at offsets {list(offsets)} into {out_dir}"
    )
    if args.trim_stationary:
        print(
            f"Trimming stationary frames: leading and trailing runs dropped, "
            f"interior runs cut to {args.keep_stationary}"
        )

    for split, name in (("train_data", "train.tfrec"), ("test_data", "test.tfrec")):
        print(f"{split}:")
        convert_dataset(
            os.path.join(args.dataset_dir, split),
            out_dir,
            name,
            offsets,
            args.trim_stationary,
            args.keep_stationary,
        )
