"""Post-training preview for an autopilot model: predicted vs. true control on the
test set.

Writes two figures:
  eval_samples.png - a grid of random test frames, each with the recorded (true)
                     and predicted control drawn as arrows; titles turn red when the
                     predicted steering goes the wrong way
  eval_scatter.png - predicted vs. true steering and throttle over the whole test
                     set, with MAE and the same accuracy thresholds as metrics.py

Runs automatically at the end of `python -m openbot.train` (autopilot only), or on
an already trained model:

    python -m openbot.eval_preview --model_dir models/<model_name>
    python -m openbot.eval_preview --model_dir models/<model_name> --checkpoint last
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import dataset_dir, metrics, tfrecord_seq, tfrecord_utils

# Same thresholds as metrics.py (normalized units, 0.1 ~ 25 raw).
TOLERANCE = 0.1

TRUE_COLOR = "limegreen"
PRED_COLOR = "orange"


def load_test_dataset(tfrec_path, seq_len):
    """Unbatched ((image, cmd), label) pairs, matching train.process_test_sample."""
    if seq_len > 1:
        parse_fn = tfrecord_seq.make_parse_fn(seq_len)
    else:
        parse_fn = tfrecord_utils.parse_tfrecord_fn_autopilot

    def to_sample(features):
        label = tf.stack([features["left"], features["right"]])
        return (features["image"], features["cmd"]), label

    return tf.data.TFRecordDataset(tfrec_path).map(parse_fn).map(to_sample)


def predict_all(model, dataset, batch_size=64):
    """Run the model over the whole dataset. Returns (cmds, labels, preds)."""
    cmds, labels, preds = [], [], []
    for (image, cmd), label in dataset.batch(batch_size):
        pred = model((image, tf.reshape(cmd, (-1, 1))), training=False)
        cmds.append(cmd.numpy())
        labels.append(label.numpy())
        preds.append(np.asarray(pred))
    return np.concatenate(cmds), np.concatenate(labels), np.concatenate(preds)


def steering_wrong_way(true_steer, pred_steer):
    # The inverse of metrics.direction_metric.
    return np.sign(pred_steer) != np.sign(true_steer) and abs(pred_steer) >= TOLERANCE


def draw_control(ax, width, height, steering, throttle, color):
    """An arrow from the image centre: tilt is steering (+ = right, since
    steering = (left - right) / 2), length is throttle (pointing down = reverse)."""
    length = 0.45 * height * float(np.clip(throttle, -1, 1))
    tilt = np.deg2rad(60) * float(np.clip(steering, -1, 1))
    x0, y0 = width / 2, height / 2
    ax.arrow(
        x0,
        y0,
        length * np.sin(tilt),
        -length * np.cos(tilt),
        color=color,
        width=max(width, height) * 0.012,
        length_includes_head=True,
    )


def plot_samples(dataset, indices, cmds, labels, preds, out_path):
    wanted = tf.constant(indices, dtype=tf.int64)
    samples = dataset.enumerate().filter(
        lambda i, sample: tf.reduce_any(tf.equal(i, wanted))
    )

    cols = 5
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, (i, ((image, _), _)) in zip(axes.flat, samples):
        i = int(i)
        # Stacked-frame inputs: show the newest frame, the one the label belongs to.
        image = image.numpy()[..., -3:]
        height, width = image.shape[:2]
        ax.imshow(image)

        (true_s, true_t), (pred_s, pred_t) = labels[i], preds[i]
        draw_control(ax, width, height, true_s, true_t, TRUE_COLOR)
        draw_control(ax, width, height, pred_s, pred_t, PRED_COLOR)
        ax.set_title(
            "#%d  cmd %+d\ntrue  s %+.2f  t %+.2f\npred  s %+.2f  t %+.2f"
            % (i, cmds[i], true_s, true_t, pred_s, pred_t),
            fontsize=9,
            family="monospace",
            color="red" if steering_wrong_way(true_s, pred_s) else "black",
        )

    fig.legend(
        handles=[
            plt.Line2D([], [], color=TRUE_COLOR, lw=4, label="true"),
            plt.Line2D([], [], color=PRED_COLOR, lw=4, label="predicted"),
        ],
        loc="upper right",
    )
    fig.suptitle(
        "Random test frames - arrow tilt = steering, length = throttle; "
        "red title = steering predicted the wrong way"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summary(labels, preds):
    y_true, y_pred = tf.constant(labels), tf.constant(preds)

    def pct(metric):
        return 100.0 * float(tf.reduce_mean(tf.cast(metric(y_true, y_pred), tf.float32)))

    err = np.abs(labels - preds)
    return {
        "steering": {
            "MAE": err[:, 0].mean(),
            "within 0.1": pct(metrics.angle_metric),
            "direction": pct(metrics.direction_metric),
        },
        "throttle": {
            "MAE": err[:, 1].mean(),
            "within 0.1": pct(metrics.throttle_metric),
            "direction": pct(metrics.throttle_direction_metric),
        },
    }


def plot_scatter(labels, preds, stats, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for col, (ax, name) in enumerate(zip(axes, ("steering", "throttle"))):
        lo = min(labels[:, col].min(), preds[:, col].min(), -0.1)
        hi = max(labels[:, col].max(), preds[:, col].max(), 0.1)
        ax.plot([lo, hi], [lo, hi], color="gray", lw=1, label="perfect")
        ax.fill_between(
            [lo, hi],
            [lo - TOLERANCE, hi - TOLERANCE],
            [lo + TOLERANCE, hi + TOLERANCE],
            color="gray",
            alpha=0.15,
            label="within %.1f" % TOLERANCE,
        )
        ax.scatter(labels[:, col], preds[:, col], s=6, alpha=0.4)
        ax.set_xlabel("true " + name)
        ax.set_ylabel("predicted " + name)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        s = stats[name]
        ax.set_title(
            "%s  MAE %.3f | within 0.1 %.1f%% | direction %.1f%%"
            % (name, s["MAE"], s["within 0.1"], s["direction"])
        )
    fig.suptitle("Test set: %d frames" % len(labels))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(model, dataset, out_dir, num_samples=20, seed=0):
    """Write eval_samples.png and eval_scatter.png for an unbatched
    ((image, cmd), label) dataset into out_dir. Returns the summary stats."""
    cmds, labels, preds = predict_all(model, dataset)
    # Fixed seed: every model is previewed on the same frames, so runs compare.
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(len(labels), size=min(num_samples, len(labels)), replace=False)
    )

    os.makedirs(out_dir, exist_ok=True)
    plot_samples(
        dataset, indices, cmds, labels, preds, os.path.join(out_dir, "eval_samples.png")
    )
    stats = summary(labels, preds)
    plot_scatter(labels, preds, stats, os.path.join(out_dir, "eval_scatter.png"))

    for name, s in stats.items():
        print(
            "%-8s MAE %.3f | within 0.1: %.1f%% | direction: %.1f%%"
            % (name, s["MAE"], s["within 0.1"], s["direction"])
        )
    print("Saved eval_samples.png and eval_scatter.png to", out_dir)
    return stats


def default_tfrec_path(model_dir):
    """The test record a model was trained on, read back from its directory name
    (train.Hyperparameters.__str__ appends _seq<offsets>[_trim<n>])."""
    m = re.search(r"_seq(\d+(?:-\d+)*)(?:_trim(\d+))?", os.path.basename(model_dir))
    if m is None:
        records_dir = "tfrecords"
    else:
        offsets = tuple(int(o) for o in m[1].split("-"))
        trim = m[2] is not None
        records_dir = tfrecord_seq.tfrecords_dir_name(
            offsets, trim, int(m[2]) if trim else None
        )
    return os.path.join(dataset_dir, records_dir, "test.tfrec")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model_dir", required=True, help="models/<model_name>")
    parser.add_argument(
        "--checkpoint",
        default="best-val",
        choices=["best-val", "best-train", "last"],
        help="which checkpoint to evaluate (default: best-val, the one exported "
        "as best.tflite)",
    )
    parser.add_argument(
        "--tfrec", default=None, help="test record (default: from the model name)"
    )
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint = os.path.join(args.model_dir, "checkpoints", f"cp-{args.checkpoint}.ckpt")
    model = tf.keras.models.load_model(checkpoint, compile=False)
    seq_len = model.inputs[0].shape[-1] // 3
    tfrec = args.tfrec or default_tfrec_path(args.model_dir)
    print(f"Model: {checkpoint} (seq_len {seq_len})\nTest data: {tfrec}")

    run(
        model,
        load_test_dataset(tfrec, seq_len),
        os.path.join(args.model_dir, "logs"),
        args.num_samples,
        args.seed,
    )


if __name__ == "__main__":
    main()
