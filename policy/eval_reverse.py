import csv
import glob
import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "/workspace/models/openbot_pilot_net_lr0.0003_bz128_bn_flip/checkpoints/best-val.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_idx = next(d["index"] for d in input_details if "img" in d["name"])
cmd_idx = next(d["index"] for d in input_details if "cmd" in d["name"])
out_idx = output_details[0]["index"]


def load_img(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img.numpy()[None, ...]


rows = []
for f in glob.glob("/workspace/dataset/test_data/*/*/sensor_data/matched_frame_ctrl_cmd_processed.txt"):
    with open(f, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

print(f"total test rows: {len(rows)}")

# raw ground truth throttle is row['right'] in RAW units (not yet /255)
results = []
for row in rows:
    steer_raw = float(row["left"])
    throttle_raw = float(row["right"])
    cmd = float(row["cmd"])
    img_path = row["frame"]
    if not os.path.exists(img_path):
        continue
    img = load_img(img_path)
    interpreter.set_tensor(img_idx, img)
    interpreter.set_tensor(cmd_idx, np.array([[cmd]], dtype=np.float32))
    interpreter.invoke()
    pred = interpreter.get_tensor(out_idx)[0]
    pred_steer_raw = pred[0] * 255.0
    pred_throttle_raw = pred[1] * 255.0
    results.append((throttle_raw, pred_throttle_raw, steer_raw, pred_steer_raw))

results.sort(key=lambda x: x[0])  # sort by ground-truth throttle ascending (most negative first)

print("\n=== Most negative (reverse) ground-truth throttle samples ===")
print(f"{'gt_throttle':>12} {'pred_throttle':>14} {'gt_steer':>10} {'pred_steer':>12}")
for gt_t, pred_t, gt_s, pred_s in results[:20]:
    print(f"{gt_t:12.1f} {pred_t:14.1f} {gt_s:10.1f} {pred_s:12.1f}")

# aggregate stats for strongly-negative gt throttle
strong_rev = [r for r in results if r[0] <= -150]
if strong_rev:
    gt_mean = sum(r[0] for r in strong_rev) / len(strong_rev)
    pred_mean = sum(r[1] for r in strong_rev) / len(strong_rev)
    print(f"\nFor {len(strong_rev)} samples with gt_throttle <= -150:")
    print(f"  mean gt_throttle   = {gt_mean:.1f}")
    print(f"  mean pred_throttle = {pred_mean:.1f}")
