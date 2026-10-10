"""Post-training preview for an autopilot model: predicted vs. true control on the
test set.

Writes two figures:
  eval_samples.png - a grid of random test frames, each with the recorded (true)
                     and predicted control drawn as arrows; titles turn red when the
                     predicted steering goes the wrong way
  eval_scatter.png - predicted vs. true steering and throttle over the whole test
                     set, with MAE and the same accuracy thresholds as metrics.py

and, for a memory model (donkey_memory), a third:
  eval_rollout.png - each test session replayed in order with the model fed its own
                     previous predictions as memory, the way it drives, next to the
                     teacher-forced prediction that gets the recorded memory

Runs automatically at the end of `python -m openbot.train` (autopilot only), or on
an already trained model:

    python -m openbot.eval_preview --model_dir models/<model_name>
    python -m openbot.eval_preview --model_dir models/<model_name> --checkpoint last
"""

import argparse
import collections
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import dataset_dir, metrics, tfrecord_mem, tfrecord_seq, tfrecord_utils

# Same thresholds as metrics.py (normalized units, 0.1 ~ 25 raw).
TOLERANCE = 0.1

TRUE_COLOR = "limegreen"
PRED_COLOR = "orange"
TEACHER_COLOR = "gray"

# Sessions drawn in eval_rollout.png; the MAE printed covers all of them.
MAX_ROLLOUT_SESSIONS = 4


def input_named(model, name):
    return next((t for t in model.inputs if name in t.name), None)


def mem_len_of(model):
    """Previous controls a memory model takes (0 for every other model)."""
    mem = input_named(model, "mem_input")
    return 0 if mem is None else mem.shape[-1] // 2


def load_test_dataset(tfrec_path, seq_len, mem_len=0):
    """Unbatched ((image, cmd), label) pairs, matching train.process_test_sample.
    For a memory model the memory takes the place of cmd."""
    if mem_len:
        parse_fn = tfrecord_mem.make_parse_fn(mem_len)
    elif seq_len > 1:
        parse_fn = tfrecord_seq.make_parse_fn(seq_len)
    else:
        parse_fn = tfrecord_utils.parse_tfrecord_fn_autopilot

    def to_sample(features):
        label = tf.stack([features["left"], features["right"]])
        second = features["mem"] if mem_len else features["cmd"]
        return (features["image"], second), label

    return tf.data.TFRecordDataset(tfrec_path).map(parse_fn).map(to_sample)


def predict_all(model, dataset, batch_size=64):
    """Run the model over the whole dataset. Returns (cmds, labels, preds); cmds holds
    the memory vectors instead for a memory model."""
    cmds, labels, preds = [], [], []
    for (image, cmd), label in dataset.batch(batch_size):
        second = cmd if cmd.shape.rank == 2 else tf.reshape(cmd, (-1, 1))
        pred = model((image, second), training=False)
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
    # Row height follows the image shape (OpenBot crops are a wide 256x96), plus room
    # for the three-line title.
    (first_image, _), _ = next(iter(dataset.take(1)))
    height, width = first_image.shape[:2]
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, (4 * height / width + 0.9) * rows), squeeze=False
    )
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
        # A memory model has no cmd; its memory is the recorded history anyway.
        header = "#%d" % i if np.ndim(cmds[i]) else "#%d  cmd %+d" % (i, cmds[i])
        ax.set_title(
            "%s\ntrue  s %+.2f  t %+.2f\npred  s %+.2f  t %+.2f"
            % (header, true_s, true_t, pred_s, pred_t),
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


def rollout(model, tfrec_path, mem_len, out_path):
    """Replay each test session in order, feeding the model its own previous
    predictions as memory, the way donkeycar's KerasMemory.run drives: the memory
    starts at [0, 0] per pair (mem_start_speed 0) and each prediction is pushed in.

    The teacher-forced prediction (recorded memory, as in training) is computed on
    the same frames, so the gap between the two curves is what feeding back its own
    output costs. Returns the rollout and teacher-forced MAE over every session.
    """
    parse_fn = tfrecord_mem.make_parse_fn(mem_len)
    step = tf.function(lambda image, mem: model((image, mem), training=False))

    sessions = collections.OrderedDict()
    for features in tf.data.TFRecordDataset(tfrec_path).map(parse_fn):
        # path is <session>/images/<frame>_crop.jpeg
        session = os.path.dirname(os.path.dirname(features["path"].numpy().decode()))
        if session not in sessions:
            sessions[session] = {"true": [], "teacher": [], "rollout": []}
            memory = collections.deque([[0.0, 0.0]] * mem_len, maxlen=mem_len)
        own_mem = np.array(memory, dtype=np.float32).reshape(-1)
        # One call for both: the same image with the recorded and the fed-back memory.
        pred = step(
            tf.stack([features["image"]] * 2), tf.stack([features["mem"], own_mem])
        ).numpy()
        trace = sessions[session]
        trace["true"].append([features["left"].numpy(), features["right"].numpy()])
        trace["teacher"].append(pred[0])
        trace["rollout"].append(pred[1])
        memory.append(pred[1].tolist())

    traces = [
        (session, {k: np.array(v) for k, v in trace.items()})
        for session, trace in sessions.items()
    ]
    everything = {
        k: np.concatenate([trace[k] for _, trace in traces])
        for k in ("true", "teacher", "rollout")
    }
    mae = {
        k: np.abs(everything[k] - everything["true"]).mean(axis=0)
        for k in ("teacher", "rollout")
    }

    shown = traces[:MAX_ROLLOUT_SESSIONS]
    fig, axes = plt.subplots(
        len(shown), 2, figsize=(16, 3.2 * len(shown)), squeeze=False
    )
    for row, (session, trace) in zip(axes, shown):
        for col, (ax, name) in enumerate(zip(row, ("steering", "throttle"))):
            ax.plot(trace["true"][:, col], color=TRUE_COLOR, lw=1.5, label="true")
            ax.plot(
                trace["rollout"][:, col],
                color=PRED_COLOR,
                lw=1.5,
                label="rollout (own predictions as memory)",
            )
            # Dashed and on top: it often runs right along the rollout.
            ax.plot(
                trace["teacher"][:, col],
                color=TEACHER_COLOR,
                lw=1,
                ls="--",
                label="teacher-forced (recorded memory)",
            )
            ax.set_ylabel(name)
            ax.set_title(
                "%s - %s  MAE rollout %.3f | teacher-forced %.3f"
                % (
                    os.path.basename(session),
                    name,
                    np.abs(trace["rollout"][:, col] - trace["true"][:, col]).mean(),
                    np.abs(trace["teacher"][:, col] - trace["true"][:, col]).mean(),
                ),
                fontsize=10,
            )
    axes[0, 0].legend(loc="upper left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("frame (stationary frames removed)")
    fig.suptitle(
        "Closed-loop replay of %d of %d test sessions" % (len(shown), len(traces))
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return mae


def run(model, dataset, out_dir, num_samples=20, seed=0, tfrec_path=None):
    """Write eval_samples.png and eval_scatter.png for an unbatched
    ((image, cmd), label) dataset into out_dir, plus eval_rollout.png for a memory
    model when its test tfrecord is given. Returns the summary stats."""
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

    mem_len = mem_len_of(model)
    if mem_len and tfrec_path:
        mae = rollout(model, tfrec_path, mem_len, os.path.join(out_dir, "eval_rollout.png"))
        for k in ("teacher", "rollout"):
            print("%-8s MAE steering %.3f | throttle %.3f" % (k, mae[k][0], mae[k][1]))
        print("Saved eval_rollout.png to", out_dir)
    return stats


def default_tfrec_path(model_dir):
    """The test record a model was trained on, read back from its directory name
    (train.Hyperparameters.__str__ appends _seq<offsets>[_trim<n>] or _mem<n>)."""
    mem = re.search(r"_mem(\d+)", os.path.basename(model_dir))
    if mem is not None:
        records_dir = tfrecord_mem.tfrecords_dir_name(int(mem[1]))
        return os.path.join(dataset_dir, records_dir, "test.tfrec")

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
    seq_len = input_named(model, "img_input").shape[-1] // 3
    mem_len = mem_len_of(model)
    tfrec = args.tfrec or default_tfrec_path(args.model_dir)
    print(
        f"Model: {checkpoint} (seq_len {seq_len}, mem_len {mem_len})\n"
        f"Test data: {tfrec}"
    )

    run(
        model,
        load_test_dataset(tfrec, seq_len, mem_len),
        os.path.join(args.model_dir, "logs"),
        args.num_samples,
        args.seed,
        tfrec_path=tfrec,
    )


if __name__ == "__main__":
    main()
