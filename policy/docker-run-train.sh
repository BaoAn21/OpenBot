#!/usr/bin/env bash
# Runs the openbot-train image against ./dataset, writing checkpoints and
# .tflite files to ./models. All arguments are forwarded to `openbot.train`
# (see README.md's Shell section, or `python -m openbot.train --help`).
#
# Example:
#   ./docker-run-train.sh --create_tf_record --model pilot_net --batch_size 128 --num_epochs 100 --batch_norm --flip_aug

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p dataset models

docker run --rm --gpus all \
    -v "$(pwd)/dataset:/workspace/dataset" \
    -v "$(pwd)/models:/workspace/models" \
    openbot-train "$@"
