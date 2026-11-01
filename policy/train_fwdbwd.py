"""Train the forward or the backward Autopilot Session model on a choice of sessions.

  fwd  donkey_memory (--color_aug), the default, or
       cil_mobile_fast_memory (--batch_norm --flip_aug); records in dataset/tfrecords_mem<N>/
  bwd  cil_mobile_fast (--batch_norm --flip_aug), the default, records in dataset/tfrecords/,
       or cil_mobile_fast_memory (--batch_norm --flip_aug) or donkey_memory_reverse
       (--color_aug), records in dataset/tfrecords_bwd_mem<N>/

Pick the model with --model; cil_mobile_fast_memory is cil_mobile_fast plus donkey_memory's
memory of its last --mem_len (default 3) outputs, donkey_memory_reverse is donkey_memory with
a tanh throttle so it can reverse.

The sessions come from dataset/fwdbwd/{train,test}/<fwd|bwd>/session_<N>/ (see
split_direction.py). The records are rebuilt only when the choice of sessions or their
content changed: dataset/<records>/manifest.txt lists every folder in them.

The model folder is named after the direction and the sessions, e.g.
models/fwd_s1-2-3_donkey_memory_lr0.0003_bz128_mem3_color, and openbot.train copies the
manifest into it as data_manifest.txt, so every model says what it was trained on. It
refuses a model folder that already exists unless you pass --resume.

Anything this script doesn't know goes on to openbot.train, e.g. --num_epochs 60.

Usage (from policy/, conda env):
    python train_fwdbwd.py --list
    python train_fwdbwd.py fwd                      # all sessions
    python train_fwdbwd.py bwd --sessions 1 3 4
    python train_fwdbwd.py fwd --model cil_mobile_fast_memory
    python train_fwdbwd.py bwd --model cil_mobile_fast_memory
    python train_fwdbwd.py bwd --model donkey_memory_reverse
    python train_fwdbwd.py fwd --sessions 2 --build_only
    python train_fwdbwd.py fwd --num_epochs 60
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile

POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(POLICY_DIR, "dataset")
ROOT = os.path.join(DATASET_DIR, "fwdbwd")
SPLITS = (("train", "train_data", "train.tfrec"), ("test", "test_data", "test.tfrec"))

# The models each direction can train, with their options; the first is the default.
MODELS = {
    "fwd": {
        "donkey_memory": ["--color_aug"],
        "cil_mobile_fast_memory": ["--batch_norm", "--flip_aug"],
    },
    "bwd": {
        "cil_mobile_fast": ["--batch_norm", "--flip_aug"],
        "cil_mobile_fast_memory": ["--batch_norm", "--flip_aug"],
        "donkey_memory_reverse": ["--color_aug"],
    },
}
COMMON_ARGS = [
    "--batch_size",
    "128",
    "--learning_rate",
    "0.0003",
    "--num_epochs",
    "100",
    "--early_stop",
    "20",
]


def sessions_in(root, direction):
    found = set()
    for split, _, _ in SPLITS:
        d = os.path.join(root, split, direction)
        if os.path.isdir(d):
            found.update(
                int(s.split("_")[1]) for s in os.listdir(d) if s.startswith("session_")
            )
    return sorted(found)


def folders(root, split, direction, number):
    d = os.path.join(root, split, direction, f"session_{number}")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def manifest(root, direction, numbers, mem_len):
    """mem_len is 0 for the plain backward records."""
    lines = [f"# {direction}" + (f", mem_len {mem_len}" if mem_len else "")]
    for split, _, _ in SPLITS:
        for n in numbers:
            lines += [
                f"{split}/{direction}/session_{n}/{f}"
                for f in folders(root, split, direction, n)
            ]
    return "\n".join(lines) + "\n"


def show_list(root):
    raw = os.path.join(root, "raw")
    numbers = sorted(set(sessions_in(root, "fwd")) | set(sessions_in(root, "bwd")))
    if not numbers:
        print(f"no sessions under {root}")
    for n in numbers:
        note_path = os.path.join(raw, f"session_{n}", "note.txt")
        note = (
            open(note_path).read().strip().replace("\n", "; ")
            if os.path.isfile(note_path)
            else ""
        )
        recs = (
            [
                d
                for d in os.listdir(os.path.join(raw, f"session_{n}"))
                if os.path.isdir(os.path.join(raw, f"session_{n}", d))
            ]
            if os.path.isdir(os.path.join(raw, f"session_{n}"))
            else []
        )
        counts = []
        for direction in ("fwd", "bwd"):
            for split, _, _ in SPLITS:
                fs = folders(root, split, direction, n)
                frames = sum(
                    len(
                        os.listdir(
                            os.path.join(
                                root, split, direction, f"session_{n}", f, "images"
                            )
                        )
                    )
                    for f in fs
                )
                counts.append(f"{direction} {split} {frames:5}")
        print(
            f"session {n}: {len(recs):3} recordings | "
            + " | ".join(counts)
            + (f" | {note}" if note else "")
        )
    inbox = os.path.join(root, "inbox")
    waiting = len(os.listdir(inbox)) if os.path.isdir(inbox) else 0
    if waiting:
        print(f"inbox: {waiting} recordings waiting for split_direction.py")


@contextlib.contextmanager
def staged(root, direction, numbers):
    """A temporary dataset root: train_data/ and test_data/ hold one link per session."""
    stage = tempfile.mkdtemp(prefix=".build_", dir=root)
    links = []
    try:
        for split, data_dir, _ in SPLITS:
            os.mkdir(os.path.join(stage, data_dir))
            for n in numbers:
                target = os.path.join(root, split, direction, f"session_{n}")
                if os.path.isdir(target):
                    link = os.path.join(stage, data_dir, f"session_{n}")
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


def build_fwd(stage, out_dir, mem_len, max_gap_s=None):
    from openbot import tfrecord_mem

    for _, data_dir, name in SPLITS:
        tfrecord_mem.convert_dataset(
            os.path.join(stage, data_dir),
            out_dir,
            name,
            mem_len,
            verbose=False,
            max_gap_s=max_gap_s,
        )


def check_bwd(path):
    """The backward model is fed cmd 2 and must never learn forward throttle, nor
    remember any."""
    import tensorflow as tf

    count = bad = 0
    for raw in tf.data.TFRecordDataset(path):
        f = tf.train.Example.FromString(raw.numpy()).features.feature
        count += 1
        # Memory records only: (steering, throttle) pairs, so every odd entry.
        past_throttles = list(f["mem"].float_list.value)[1::2] if "mem" in f else []
        bad += (
            f["cmd"].float_list.value[0] != 2
            or f["right"].float_list.value[0] >= 0
            or any(t >= 0 for t in past_throttles)
        )
    print(f"{path}: {count} examples")
    if bad:
        raise SystemExit(f"{path}: {bad} examples without cmd 2 and throttle < 0")


def build_bwd(stage, out_dir):
    from openbot import tfrecord

    for _, data_dir, name in SPLITS:
        # convert_dataset prints every label line; keep only the missing images.
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            tfrecord.convert_dataset(
                os.path.join(stage, data_dir), out_dir, name, policy="autopilot"
            )
        missing = log.getvalue().count("cannot be found")
        if missing:
            raise SystemExit(f"{data_dir}: {missing} images not found")
        check_bwd(os.path.join(out_dir, name))


def build_bwd_mem(stage, out_dir, mem_len):
    from split_direction import MAX_GAP_S

    # A recording's backward frames share one folder, reverse after reverse, so cut the
    # memory where split_direction cuts the forward runs: no memory from the previous
    # reverse, just as the phone clears it on every switch.
    build_fwd(stage, out_dir, mem_len, max_gap_s=MAX_GAP_S)
    for _, _, name in SPLITS:
        check_bwd(os.path.join(out_dir, name))


def records_dir(dataset_dir, direction, mem_len):
    """mem_len is 0 for the plain backward records."""
    if direction == "fwd":
        return os.path.join(dataset_dir, f"tfrecords_mem{mem_len}")
    if mem_len:
        # Apart from the forward memory records, so switching direction never
        # rebuilds (or trains on) the other one's.
        return os.path.join(dataset_dir, f"tfrecords_bwd_mem{mem_len}")
    return os.path.join(dataset_dir, "tfrecords")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Other options go on to openbot.train.",
    )
    parser.add_argument("direction", nargs="?", choices=("fwd", "bwd"))
    parser.add_argument(
        "--list", action="store_true", help="show the sessions and exit"
    )
    parser.add_argument(
        "--sessions", type=int, nargs="+", help="session numbers (default: all)"
    )
    parser.add_argument(
        "--model",
        help="fwd: donkey_memory (default) or cil_mobile_fast_memory; "
        "bwd: cil_mobile_fast (default), cil_mobile_fast_memory or donkey_memory_reverse",
    )
    parser.add_argument(
        "--mem_len",
        type=int,
        default=3,
        help="memory length of a memory model (default 3)",
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
    if args.list:
        show_list(root)
        return
    if args.direction is None:
        parser.error("give fwd or bwd (or --list)")
    direction = args.direction
    model = args.model or next(iter(MODELS[direction]))
    if model not in MODELS[direction]:
        parser.error(f"{direction} trains {' or '.join(MODELS[direction])}, not {model}")
    mem_len = args.mem_len if "memory" in model else 0

    available = sessions_in(root, direction)
    numbers = sorted(set(args.sessions)) if args.sessions else available
    unknown = [n for n in numbers if n not in available]
    if unknown or not numbers:
        raise SystemExit(
            f"no {direction} data for sessions {unknown or numbers}; have {available}"
        )
    empty_test = [n for n in numbers if not folders(root, "test", direction, n)]
    if len(empty_test) == len(numbers):
        raise SystemExit(f"the chosen sessions have no {direction} test data")
    if empty_test:
        print(f"note: sessions {empty_test} have no {direction} test data")

    out_dir = records_dir(dataset_dir, direction, mem_len)
    wanted = manifest(root, direction, numbers, mem_len)
    manifest_path = os.path.join(out_dir, "manifest.txt")
    have = open(manifest_path).read() if os.path.isfile(manifest_path) else None
    if args.rebuild or have != wanted:
        print(f"building {direction} records for sessions {numbers} -> {out_dir}")
        with contextlib.suppress(FileNotFoundError):
            os.remove(manifest_path)  # half-built records must never look current
        with staged(root, direction, numbers) as stage:
            if direction == "fwd":
                build_fwd(stage, out_dir, mem_len)
            elif mem_len:
                build_bwd_mem(stage, out_dir, mem_len)
            else:
                build_bwd(stage, out_dir)
        with open(manifest_path, "w") as f:
            f.write(wanted)
    else:
        print(f"{direction} records for sessions {numbers} are up to date in {out_dir}")
    if args.build_only:
        return

    run_name = f"{direction}_s" + "-".join(str(n) for n in numbers)
    cmd = [sys.executable, "-m", "openbot.train", "--run_name", run_name]
    cmd += ["--model", model] + MODELS[direction][model] + COMMON_ARGS
    if mem_len:
        cmd += ["--mem_len", str(mem_len)]
    if direction == "bwd" and mem_len:
        cmd += ["--records_dir", out_dir]
    cmd += train_extra  # later options win in argparse
    print(" ".join(cmd[1:]))
    # openbot.train copies manifest.txt into the model folder as data_manifest.txt.
    sys.exit(subprocess.run(cmd, cwd=POLICY_DIR).returncode)


if __name__ == "__main__":
    main()
