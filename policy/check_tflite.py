"""Check that a model folder's .tflite files compute what its checkpoints compute.

Feeds real test frames to each .tflite one after another, the way the phone does, and
compares every answer with the Keras checkpoint it was exported from (see
openbot/check_tflite.py). PASS means the file on the phone is the model you trained.

    best.tflite / best-val.tflite  <-  checkpoints/cp-best-val.ckpt
    best-train.tflite              <-  checkpoints/cp-best-train.ckpt
    last.tflite                    <-  checkpoints/cp-last.ckpt
    any other .tflite (a renamed copy) is checked against cp-best-val.ckpt

Usage (from policy/, conda env):
    python check_tflite.py models/fwd_s4-5-6-7_donkey_rnn_lr0.0003_bz128_seq0-3-6
    python check_tflite.py models/fwd_s7_*              # several folders
    python check_tflite.py models/<folder> --frames 4000 --records dataset/tfrecords_fwd/test.tfrec

--export first re-exports the folder's .tflite files from its checkpoints with the
current utils.generate_tflite (renamed copies of best.tflite included), then checks them.
"""

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from openbot import check_tflite, utils  # noqa: E402

SOURCES = {
    "best.tflite": "cp-best-val.ckpt",
    "best-val.tflite": "cp-best-val.ckpt",
    "best-train.tflite": "cp-best-train.ckpt",
    "last.tflite": "cp-last.ckpt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dirs", nargs="+", help="model folders under models/")
    parser.add_argument("--frames", type=int, default=1000, help="test frames (default 1000)")
    parser.add_argument("--records", help="test.tfrec to use instead of the model's own")
    parser.add_argument(
        "--export", action="store_true", help="re-export the .tflite files first"
    )
    args = parser.parse_args()

    failed = []
    for model_dir in args.model_dirs:
        checkpoints = os.path.join(model_dir, "checkpoints")
        if not os.path.isdir(checkpoints):
            print(f"{model_dir}: no checkpoints/, skipped")
            continue
        files = sorted(f for f in os.listdir(checkpoints) if f.endswith(".tflite"))
        records = args.records or check_tflite.records_for(model_dir)

        if args.export:
            for f in files:
                source = SOURCES.get(f, "cp-best-val.ckpt")
                utils.save_tflite(
                    utils.generate_tflite(checkpoints, source), checkpoints, f[: -len(".tflite")]
                )
                print(f"exported {f} from {source}")

        for f in files:
            source = SOURCES.get(f, "cp-best-val.ckpt")
            ok = check_tflite.check(
                os.path.join(checkpoints, source),
                os.path.join(checkpoints, f),
                records,
                args.frames,
            )
            if not ok:
                failed.append(os.path.join(checkpoints, f))

    print()
    if failed:
        print("FAIL:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    print("all PASS")


if __name__ == "__main__":
    main()
