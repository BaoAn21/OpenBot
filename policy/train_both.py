"""Train one Autopilot Session model for both directions on a choice of sessions.

cil_mobile_fast (default) or cil_mobile_fast_cat (--batch_norm --flip_aug), trained on the
fwd and the bwd folders of the chosen sessions together, from
dataset/fwdbwd/{train,test}/{fwd,bwd}/session_<N>/ (see split_direction.py). The backward
frames carry cmd 2, which is what the phone feeds a model while reversing, and the model's
output is linear, so one model drives both ways: pick it in both spinners of the Autopilot
Session screen (a plain CMDNAV model).

The records go to dataset/tfrecords_both/ and are rebuilt only when the choice of sessions
or their content changed (manifest.txt lists every folder in them). They are written folder
by folder and openbot.train shuffles only 10 batches deep, so the train file is shuffled
once here; otherwise every epoch would go through all forward frames, then all backward
ones. The test file stays in order.

The model lands in models/both_s<sessions>_<settings>/ with the manifest as
data_manifest.txt. Anything this script doesn't know goes on to openbot.train.

train_fwdbwd.py is untouched; this script only borrows its helpers.

Usage (from policy/, conda env):
    python train_both.py --sessions 4 5 6 7 8 9 10 11 --build_only
    python train_both.py --sessions 4 5 6 7 8 9 10 11
    python train_both.py --sessions 4 5 6 7 8 9 10 11 --model cil_mobile_fast_cat
"""

import argparse
import contextlib
import os
import random
import subprocess
import sys
import tempfile

from train_fwdbwd import (
    COMMON_ARGS,
    DATASET_DIR,
    POLICY_DIR,
    SPLITS,
    TRAIN_ENV,
    build_plain,
    folders,
    hide_gpu,
    sessions_in,
)

DIRECTIONS = ("fwd", "bwd")
MODELS = {
    "cil_mobile_fast": ["--batch_norm", "--flip_aug"],
    "cil_mobile_fast_cat": ["--batch_norm", "--flip_aug"],
}
RECORDS = "tfrecords_both"


def manifest(root, numbers):
    lines = ["# both (fwd + bwd)"]
    for split, _, _ in SPLITS:
        for n in numbers:
            for direction in DIRECTIONS:
                lines += [
                    f"{split}/{direction}/session_{n}/{f}"
                    for f in folders(root, split, direction, n)
                ]
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def staged(root, numbers):
    """A temporary dataset root: train_data/ and test_data/ hold one link per session and
    direction (fwd_session_4, bwd_session_4, ...)."""
    stage = tempfile.mkdtemp(prefix=".build_both_", dir=root)
    links = []
    try:
        for split, data_dir, _ in SPLITS:
            os.mkdir(os.path.join(stage, data_dir))
            for n in numbers:
                for direction in DIRECTIONS:
                    target = os.path.join(root, split, direction, f"session_{n}")
                    if os.path.isdir(target):
                        link = os.path.join(stage, data_dir, f"{direction}_session_{n}")
                        os.symlink(target, link)
                        links.append(link)
        yield stage
    finally:
        # Unlink by hand rather than rmtree, so nothing can ever follow a link into the data.
        for link in links:
            os.unlink(link)
        for _, data_dir, _ in SPLITS:
            with contextlib.suppress(FileNotFoundError):
                os.rmdir(os.path.join(stage, data_dir))
        os.rmdir(stage)


def check_both(path):
    """cmd 2 must go with throttle < 0 (the phone feeds the model cmd 2 while reversing),
    every other cmd with throttle > 0."""
    import tensorflow as tf

    count = backward = bad = 0
    for raw in tf.data.TFRecordDataset(path):
        f = tf.train.Example.FromString(raw.numpy()).features.feature
        is_bwd = f["cmd"].float_list.value[0] == 2
        throttle = f["right"].float_list.value[0]
        count += 1
        backward += is_bwd
        bad += throttle >= 0 if is_bwd else throttle <= 0
    print(f"{path}: {count} examples, {count - backward} forward, {backward} backward")
    if bad:
        raise SystemExit(f"{path}: {bad} examples whose throttle doesn't match their cmd")


def shuffle_records(path, seed=0):
    """Shuffle a records file in place (it fits in memory: a few hundred MB)."""
    import tensorflow as tf

    records = [raw.numpy() for raw in tf.data.TFRecordDataset(path)]
    random.Random(seed).shuffle(records)
    tmp = path + ".shuffling"
    with tf.io.TFRecordWriter(tmp) as writer:
        for raw in records:
            writer.write(raw)
    os.replace(tmp, path)


def main():
    hide_gpu()
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Other options go on to openbot.train.",
    )
    parser.add_argument(
        "--sessions", type=int, nargs="+", help="session numbers (default: all)"
    )
    parser.add_argument(
        "--model",
        default="cil_mobile_fast",
        choices=tuple(MODELS),
        help="cil_mobile_fast (default) or cil_mobile_fast_cat",
    )
    parser.add_argument(
        "--build_only", action="store_true", help="build the records, don't train"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="build the records even if unchanged"
    )
    parser.add_argument("--dataset_dir", default=DATASET_DIR, help=argparse.SUPPRESS)
    args, train_extra = parser.parse_known_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    root = os.path.join(dataset_dir, "fwdbwd")
    available = sorted(set(sessions_in(root, "fwd")) | set(sessions_in(root, "bwd")))
    numbers = sorted(set(args.sessions)) if args.sessions else available
    unknown = [n for n in numbers if n not in available]
    if unknown or not numbers:
        raise SystemExit(f"no data for sessions {unknown or numbers}; have {available}")
    for direction in DIRECTIONS:
        missing = [n for n in numbers if n not in sessions_in(root, direction)]
        if len(missing) == len(numbers):
            raise SystemExit(f"the chosen sessions have no {direction} data")
        if missing:
            print(f"note: sessions {missing} have no {direction} data")
    empty_test = [
        n
        for n in numbers
        if not any(folders(root, "test", d, n) for d in DIRECTIONS)
    ]
    if len(empty_test) == len(numbers):
        raise SystemExit("the chosen sessions have no test data")
    if empty_test:
        print(f"note: sessions {empty_test} have no test data")

    out_dir = os.path.join(dataset_dir, RECORDS)
    wanted = manifest(root, numbers)
    manifest_path = os.path.join(out_dir, "manifest.txt")
    have = open(manifest_path).read() if os.path.isfile(manifest_path) else None
    if args.rebuild or have != wanted:
        print(f"building fwd + bwd records for sessions {numbers} -> {out_dir}")
        with contextlib.suppress(FileNotFoundError):
            os.remove(manifest_path)  # half-built records must never look current
        with staged(root, numbers) as stage:
            build_plain(stage, out_dir, check=check_both)
        shuffle_records(os.path.join(out_dir, "train.tfrec"))
        with open(manifest_path, "w") as f:
            f.write(wanted)
    else:
        print(f"records for sessions {numbers} are up to date in {out_dir}")
    if args.build_only:
        return

    run_name = "both_s" + "-".join(str(n) for n in numbers)
    cmd = [sys.executable, "-m", "openbot.train", "--run_name", run_name]
    cmd += ["--model", args.model] + MODELS[args.model] + COMMON_ARGS
    cmd += ["--records_dir", out_dir]
    cmd += train_extra  # later options win in argparse
    print(" ".join(cmd[1:]))
    # openbot.train copies manifest.txt into the model folder as data_manifest.txt.
    sys.exit(subprocess.run(cmd, cwd=POLICY_DIR, env=TRAIN_ENV).returncode)


if __name__ == "__main__":
    main()
