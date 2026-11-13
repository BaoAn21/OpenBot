"""Does a .tflite file compute what its Keras checkpoint computes, the way the phone runs it?

The phone calls the interpreter once per frame, frame after frame, with no reset in
between. So the check does the same: it feeds real test frames to the .tflite one after
another and compares every answer with the Keras model's on the same inputs. A
conversion that is wrong anywhere - a mis-ordered input, a broken op, or state carried
from one call to the next (TFLite's LSTM op does that) - shows up as a difference.

Two tests:
  match      every frame's (steering, throttle) against Keras. The .tflite stores its
             weights quantized, which costs a few hundredths at most; MATCH_P99 and
             MATCH_MAX are the limits.
  stateless  the first frame again after the others must give exactly its first answer.
             A model that keeps anything between calls fails this.

Inputs are found by name (img_input, cmd_input, mem_input), as on the phone. The test
frames come from the model's records (or --records); which sessions they hold doesn't
matter here, only that the model and the file see the same ones.
"""

import os
import re

import numpy as np
import tensorflow as tf

from . import dataset_dir, tfrecord_mem, tfrecord_seq, tfrecord_utils

# Worst differences a correct, quantized export shows on this project's models
# (cil_mobile_fast, pilot_net, donkey_memory, donkey_rnn): ~0.03. A broken one was 0.3.
MATCH_P99 = 0.05
MATCH_MAX = 0.1


def input_name(name):
    """serving_default_img_input:0 -> img_input."""
    return name.replace("serving_default_", "").split(":")[0]


def records_for(model_dir):
    """The test records a model folder was trained from, by its name."""
    name = os.path.basename(os.path.normpath(model_dir))
    # The settings tags train.py puts in the name, e.g. _seq0-3-6 and _mem3 (with the
    # digits, so the "_memory" of donkey_memory doesn't count).
    seq = re.search(r"_seq(\d+(?:-\d+)*)", name)
    mem = re.search(r"_mem(\d+)", name)
    if seq:
        return os.path.join(dataset_dir, f"tfrecords_seq{seq[1]}", "test.tfrec")
    if mem:
        folder = f"tfrecords_bwd_mem{mem[1]}" if name.startswith("bwd_") else f"tfrecords_mem{mem[1]}"
        return os.path.join(dataset_dir, folder, "test.tfrec")
    if name.startswith("fwd_"):
        return os.path.join(dataset_dir, "tfrecords_fwd", "test.tfrec")
    return os.path.join(dataset_dir, "tfrecords", "test.tfrec")


def load_inputs(model, records, max_frames):
    """Test frames as the model's inputs, by name, in record order."""
    shapes = {input_name(t.name): t.shape for t in model.inputs}
    img_depth = shapes["img_input"][-1]
    if "mem_input" in shapes:
        parse = tfrecord_mem.make_parse_fn(shapes["mem_input"][-1] // 2)
    elif img_depth > 3:
        parse = tfrecord_seq.make_parse_fn(img_depth // 3)
    else:
        parse = tfrecord_utils.parse_tfrecord_fn_autopilot

    inputs = {name: [] for name in shapes}
    for f in tf.data.TFRecordDataset(records).map(parse).take(max_frames):
        inputs["img_input"].append(f["image"].numpy())
        if "cmd_input" in inputs:
            inputs["cmd_input"].append([float(f["cmd"])])
        if "mem_input" in inputs:
            inputs["mem_input"].append(f["mem"].numpy())
    return {name: np.array(v, np.float32) for name, v in inputs.items()}


def check(checkpoint, tflite_path, records, max_frames=1000, verbose=True):
    """Returns True when the .tflite passes both tests."""
    model = tf.keras.models.load_model(checkpoint, compile=False)
    inputs = load_inputs(model, records, max_frames)
    n = len(inputs["img_input"])
    if n == 0:
        raise SystemExit(f"no test frames in {records}")

    names = [input_name(t.name) for t in model.inputs]
    expected = model.predict([inputs[name] for name in names], batch_size=64, verbose=0)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    index = {input_name(d["name"]): d["index"] for d in interpreter.get_input_details()}
    if set(index) != set(names):
        raise SystemExit(f"{tflite_path} has inputs {sorted(index)}, the model {sorted(names)}")
    output = interpreter.get_output_details()[0]["index"]

    def run(i):
        for name in names:
            interpreter.set_tensor(index[name], inputs[name][i : i + 1])
        interpreter.invoke()
        return interpreter.get_tensor(output)[0].copy()

    # Frame after frame, no reset, as on the phone.
    got = np.array([run(i) for i in range(n)])
    diff = np.abs(got - expected)
    p99, worst = np.percentile(diff, 99), diff.max()
    match_ok = p99 <= MATCH_P99 and worst <= MATCH_MAX

    again = run(0)
    stateless_ok = np.array_equal(again, got[0])

    if verbose:
        print(f"{tflite_path}")
        print(f"  {n} test frames from {records}")
        print(
            f"  match:     {'PASS' if match_ok else 'FAIL'}  biggest difference {worst:.4f}, "
            f"99% of them under {p99:.4f}, mean {diff.mean():.4f} "
            f"(limits {MATCH_MAX} / {MATCH_P99})"
        )
        print(
            f"  stateless: {'PASS' if stateless_ok else 'FAIL'}  first frame again: "
            f"{np.round(again, 4)} vs {np.round(got[0], 4)} the first time"
        )
    return match_ok and stateless_ok
