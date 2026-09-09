"""
One-off training script for the reverse-only sanity test. Reuses openbot.train's
existing Training/load_tfrecord/do_training/do_evaluation machinery unmodified,
just pointed at the reverse-only tfrecords built by build_reverse_dataset.py, so
it writes checkpoints to their own model_name and never touches the regular
dataset/ or the existing trained model under models/.

Usage (run from policy/, with the openbot conda env's lib dir on LD_LIBRARY_PATH
so TF can find the GPU libraries):
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python run_reverse_test_training.py [--num_epochs N]
"""

import argparse
import threading

from openbot.train import (
    Hyperparameters,
    Training,
    MyCallback,
    load_tfrecord,
    do_training,
    do_evaluation,
)

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=0.0003)
args = parser.parse_args()

params = Hyperparameters()
params.MODEL = "pilot_net"
params.POLICY = "autopilot"
params.TRAIN_BATCH_SIZE = args.batch_size
params.TEST_BATCH_SIZE = args.batch_size
params.LEARNING_RATE = args.learning_rate
params.NUM_EPOCHS = args.num_epochs
params.BATCH_NORM = True
params.FLIP_AUG = False  # would flip steering sign, not what we want to validate here
params.CMD_AUG = False

callback = MyCallback(broadcast=lambda *a, **k: None, cancelled=threading.Event())

tr = Training(params)
tr.dataset_name = "reverse_test_fixedloss"  # keeps checkpoints in models/reverse_test_*, separate from the real model
tr.train_data_dir = "dataset_reverse/tfrecords/train.tfrec"
tr.test_data_dir = "dataset_reverse/tfrecords/test.tfrec"

load_tfrecord(tr, verbose=1)
# Skipping visualize_train_data(tr): it writes to the shared models/train_preview.png
# (root-owned from a prior docker training run) rather than a per-model path, and is
# just a preview grid, not needed for this test.
do_training(tr, callback, verbose=1)
do_evaluation(tr, callback, verbose=1)

print(f"\nDone. Checkpoints/tflite written under models/{tr.model_name}/checkpoints/")
