"""Reference inference for a sequence policy, matching what the Android app does.

The point of this script is not to drive anything - it is to give the on-device code something
to be checked against. Two mistakes in the app are invisible at runtime because the model keeps
emitting plausible numbers either way: frames stacked newest-first instead of oldest-first, and
channels written frame-contiguously instead of interleaved per pixel. Running the same window
through here and through the phone, and comparing the two control outputs, catches both.

The stack is assembled the way AutopilotSeqFragment assembles it: offsets are read from the model
file name, turned into ages in milliseconds on a 30fps timeline, and each slot is filled with the
recorded frame whose timestamp is nearest that age - not with the frame that many rows up in the
file. On a recorded session those two agree; on the phone they do not, which is the whole reason
the app samples by time.

Usage:
    python -m openbot.infer_seq --model <model.tflite> --session <session dir> --frame <id>
    python -m openbot.infer_seq --model <model.tflite> --session <session dir> --frame <id> \
        --check-order
"""

import argparse
import os
import re

import numpy as np
import tensorflow as tf

from .tfrecord_seq import IMAGE_SUFFIX, read_matched_labels, session_image_ids

# The rate the training sessions were recorded at, measured at 33.17ms per frame. The app uses the
# same figure to turn frame offsets into ages.
DATASET_FRAME_PERIOD_MS = 1000.0 / 30.0

SEQ_TAG = re.compile(r"_seq(\d+(?:-\d+)*)")


def offsets_from_name(model_path, seq_len):
    """The training offsets carried in the model file name, or a uniform stride 2 fallback.

    This mirrors AutopilotSeqFragment.seqOffsets - if the two ever disagree about a name, the app
    is driving with different spacing than this script reports.
    """
    match = SEQ_TAG.search(os.path.basename(model_path))
    if match:
        offsets = [int(v) for v in match.group(1).split("-")]
        if len(offsets) == seq_len:
            return offsets
        print(f"name declares {len(offsets)} offsets but model takes {seq_len} frames - ignoring")
    return [i * 2 for i in range(seq_len)]


def read_timeline(session_dir):
    """Frame ids with an image on disk, and the capture timestamp of each, in milliseconds."""
    stamps = {}
    path = os.path.join(session_dir, "sensor_data", "rgbFrames.txt")
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                stamps[int(parts[1])] = int(parts[0]) / 1e6

    available = session_image_ids(session_dir)
    ids = sorted(i for i in available if i in stamps)
    return ids, stamps


def nearest(ids, stamps, target_ms):
    """The frame whose timestamp is closest to target_ms - the app's lookup, on recorded data."""
    return min(ids, key=lambda i: abs(stamps[i] - target_ms))


def build_stack(session_dir, frame_id, offsets):
    """The frames the app would have in its buffer at frame_id, oldest first."""
    ids, stamps = read_timeline(session_dir)
    if frame_id not in stamps:
        raise SystemExit(f"frame {frame_id} has no timestamp in this session")

    now = stamps[frame_id]
    # Only frames already captured are candidates; the robot cannot see the future.
    past = [i for i in ids if stamps[i] <= now]
    chosen = [nearest(past, stamps, now - o * DATASET_FRAME_PERIOD_MS) for o in offsets]
    chosen.reverse()  # oldest -> newest, the channel order tf.concat produced at training time

    frames = []
    for i in chosen:
        path = os.path.join(session_dir, "images", f"{i}{IMAGE_SUFFIX}")
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        frames.append(tf.image.convert_image_dtype(image, tf.float32).numpy())

    span = stamps[chosen[-1]] - stamps[chosen[0]]
    return np.concatenate(frames, axis=-1), chosen, span


def run(interpreter, stack, cmd):
    inputs = {d["name"]: d for d in interpreter.get_input_details()}
    img = next(d for n, d in inputs.items() if "img" in n)
    cmd_input = next(d for n, d in inputs.items() if "cmd" in n)

    interpreter.set_tensor(img["index"], stack[np.newaxis, ...].astype(np.float32))
    interpreter.set_tensor(cmd_input["index"], np.array([[cmd]], dtype=np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]


def label_for(session_dir, frame_id):
    """The recorded control for this frame, if there is one, scaled the way training scaled it."""
    for row_id, left, right, _ in read_matched_labels(session_dir):
        if row_id == frame_id:
            return left / 255.0, right / 255.0
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to the .tflite file")
    parser.add_argument("--session", required=True, help="a recorded session directory")
    parser.add_argument("--frame", type=int, required=True, help="frame id to predict for")
    parser.add_argument("--cmd", type=float, default=0.0, help="driving command input")
    parser.add_argument(
        "--check-order",
        action="store_true",
        help="also run the stack reversed, to show how much the frame order matters",
    )
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    img_shape = next(
        d["shape"] for d in interpreter.get_input_details() if len(d["shape"]) == 4
    )
    seq_len = int(img_shape[3]) // 3
    print(f"model input {list(img_shape)} -> {seq_len} frames of 3 channels")

    offsets = offsets_from_name(args.model, seq_len)
    stack, chosen, span = build_stack(args.session, args.frame, offsets)
    print(f"offsets {offsets} -> frames {chosen} (oldest first), spanning {span:.0f}ms")

    steering, throttle = run(interpreter, stack, args.cmd)
    print(f"predicted  steering {steering:+.4f}  throttle {throttle:+.4f}")
    print(f"raw units  steering {steering * 255:+7.1f}  throttle {throttle * 255:+7.1f}")

    label = label_for(args.session, args.frame)
    if label:
        print(f"recorded   steering {label[0]:+.4f}  throttle {label[1]:+.4f}")

    if args.check_order:
        reversed_stack = np.concatenate(
            np.split(stack, seq_len, axis=-1)[::-1], axis=-1
        )
        r_steering, r_throttle = run(interpreter, reversed_stack, args.cmd)
        print(
            f"reversed   steering {r_steering:+.4f}  throttle {r_throttle:+.4f}"
            "   (if this is close to the above, the stack carries no motion"
            " - pick a frame where the robot is actually moving)"
        )


if __name__ == "__main__":
    main()
