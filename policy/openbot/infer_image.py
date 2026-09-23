"""Reference inference for a single-frame policy (e.g. pilot_net), matching what the
Android app does. Lets you sanity-check a freshly trained .tflite model against a
recorded frame - or an arbitrary image - on the PC, before flashing it to the phone.

Usage:
    python -m openbot.infer_image --model <model.tflite> --session <session dir> --frame <id>
    python -m openbot.infer_image --model <model.tflite> --image <path/to/frame.jpeg> --cmd 0

    # Batch mode: every moving frame in the session, predicted with its own recorded
    # cmd, reported as aggregate error against the recorded left/right.
    python -m openbot.infer_image --model <model.tflite> --session <session dir>
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from .tfrecord_seq import IMAGE_SUFFIX, is_stationary, read_matched_labels


def load_image(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.convert_image_dtype(img, tf.float32).numpy()


def label_for(session_dir, frame_id):
    """The recorded control for this frame, if there is one.

    The label columns are still named left/right in the log files, but under the
    Ackermann convention column 0 is steering and column 1 is throttle.
    """
    for row_id, steering, throttle, _ in read_matched_labels(session_dir):
        if row_id == frame_id:
            return steering / 255.0, throttle / 255.0
    return None


def run(interpreter, img, cmd):
    inputs = {d["name"]: d for d in interpreter.get_input_details()}
    img_input = next(d for n, d in inputs.items() if "img" in n)
    cmd_input = next(d for n, d in inputs.items() if "cmd" in n)

    interpreter.set_tensor(img_input["index"], img[np.newaxis, ...].astype(np.float32))
    interpreter.set_tensor(cmd_input["index"], np.array([[cmd]], dtype=np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]


def run_batch(interpreter, session_dir):
    """Predicts every moving frame in the session against its own recorded cmd,
    and reports the error against the recorded left/right."""
    rows = [row for row in read_matched_labels(session_dir) if not is_stationary(row)]
    if not rows:
        raise SystemExit("no moving frames found in this session")

    errors = []
    worst = []
    missing = 0
    wrong_sign = 0
    for frame_id, steering_raw, throttle_raw, cmd in rows:
        img_path = os.path.join(session_dir, "images", f"{frame_id}{IMAGE_SUFFIX}")
        if not os.path.isfile(img_path):
            missing += 1
            continue
        img = load_image(img_path)
        pred_steering, pred_throttle = run(interpreter, img, float(cmd))
        steering, throttle = steering_raw / 255.0, throttle_raw / 255.0
        err_steering = abs(pred_steering - steering)
        err_throttle = abs(pred_throttle - throttle)
        errors.append((err_steering, err_throttle))
        # Driving the wrong way matters more than driving the right way too slowly.
        if np.sign(pred_throttle) != np.sign(throttle):
            wrong_sign += 1
        worst.append(
            (err_steering + err_throttle, frame_id, pred_steering, pred_throttle, steering, throttle)
        )

    errors = np.array(errors)
    print(f"{len(errors)} moving frames evaluated ({missing} missing on disk, skipped)")
    print(f"MAE        steering {errors[:, 0].mean():.4f}  throttle {errors[:, 1].mean():.4f}")
    print(f"max error  steering {errors[:, 0].max():.4f}  throttle {errors[:, 1].max():.4f}")
    print(f"throttle sign wrong on {wrong_sign}/{len(errors)} frames")

    worst.sort(reverse=True)
    print("worst 5 frames (combined error):")
    for _, frame_id, pred_steering, pred_throttle, steering, throttle in worst[:5]:
        print(
            f"  frame {frame_id:>6}  predicted ({pred_steering:+.4f}, {pred_throttle:+.4f})"
            f"  recorded ({steering:+.4f}, {throttle:+.4f})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to the .tflite file")
    parser.add_argument("--session", help="a recorded session directory")
    parser.add_argument(
        "--frame",
        type=int,
        help="frame id to predict for (with --session). Omit to batch-evaluate "
        "every moving frame in the session instead.",
    )
    parser.add_argument(
        "--image", help="a single image file, instead of --session/--frame"
    )
    parser.add_argument("--cmd", type=float, default=0.0, help="driving command input")
    args = parser.parse_args()

    if args.image and args.session:
        raise SystemExit("pass either --image, or --session (with an optional --frame)")
    if not args.image and not args.session:
        raise SystemExit("pass either --image, or --session (with an optional --frame)")
    if args.frame is not None and args.image:
        raise SystemExit("--frame is used with --session, not --image")

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    img_shape = next(
        d["shape"] for d in interpreter.get_input_details() if len(d["shape"]) == 4
    )
    print(f"model input {list(img_shape)}")

    if args.session and args.frame is None:
        run_batch(interpreter, args.session)
        return

    if args.image:
        img = load_image(args.image)
        label = None
    else:
        img_path = os.path.join(args.session, "images", f"{args.frame}{IMAGE_SUFFIX}")
        img = load_image(img_path)
        label = label_for(args.session, args.frame)

    if tuple(img.shape[:2]) != tuple(img_shape[1:3]):
        print(
            f"warning: image is {tuple(img.shape[:2])}, model expects "
            f"{tuple(img_shape[1:3])} - results may be meaningless"
        )

    steering, throttle = run(interpreter, img, args.cmd)
    print(f"predicted  steering {steering:+.4f}  throttle {throttle:+.4f}")
    print(f"raw units  steering {steering * 255:+7.1f}  throttle {throttle * 255:+7.1f}")

    if label:
        print(f"recorded   steering {label[0]:+.4f}  throttle {label[1]:+.4f}")


if __name__ == "__main__":
    main()
