# Created by Matthias Mueller - Intel Intelligent Systems Lab - 2020

import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from IPython.display import Javascript
from nbconvert import HTMLExporter
import codecs
import nbformat

# Select non-interactive backend for compatability
plt.switch_backend("agg")


def save_notebook():
    Javascript("IPython.notebook.save_notebook()")


def output_HTML(read_file, output_file):
    exporter = HTMLExporter()
    # read_file is '.ipynb', output_file is '.html'
    output_notebook = nbformat.read(read_file, as_version=4)
    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_file, "w", encoding="utf-8").write(output)


def prepare_for_training(
    ds, batch_sz, shuffle_buffer_sz=10000, prefetch_buffer_sz=1000, cache=True
):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_sz)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_sz)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=prefetch_buffer_sz)

    return ds


def show_batch(dataset, policy="autopilot", model=None, fig_num=1):
    # extra_batches holds the cmd of a memory model trained with --mem_cmd, which
    # comes after its memory; it is empty for every other model.
    (image_batch, cmd_batch, *extra_batches), label_batch = next(iter(dataset))
    NUM_SAMPLES = min(image_batch.numpy().shape[0], 15)

    if policy == "autopilot":
        # A memory model (donkey_memory) takes its previous controls in the cmd slot.
        command_input_name = "Cmd" if cmd_batch.shape.rank == 1 else "Mem"
        size = (15, 10)
        if model is not None:
            pred_batch = model.predict(
                (
                    tf.slice(image_batch, [0, 0, 0, 0], [NUM_SAMPLES, -1, -1, -1]),
                    cmd_batch[:NUM_SAMPLES],
                    *(batch[:NUM_SAMPLES] for batch in extra_batches),
                )
            )
    elif policy == "point_goal_nav":
        command_input_name = "Goal"
        size = (15, 15)
        if model is not None:
            pred_batch = model.predict(
                (
                    tf.slice(image_batch, [0, 0, 0, 0], [NUM_SAMPLES, -1, -1, -1]),
                    tf.slice(cmd_batch, [0, 0], [NUM_SAMPLES, -1]),
                )
            )
    else:
        raise Exception("Unknown policy")

    plt.figure(num=fig_num, figsize=size)

    def command_text(n):
        text = "%s: %s" % (command_input_name, np.round(cmd_batch.numpy()[n], 2))
        for batch in extra_batches:
            text += ", Cmd: %s" % np.round(batch.numpy()[n], 2)
        return text

    for n in range(NUM_SAMPLES):
        ax = plt.subplot(5, 3, n + 1)
        # For stacked-frame inputs only the newest frame (the last three channels) is
        # shown; it is the one the label belongs to.
        plt.imshow(image_batch[n][..., -3:])
        if model is None:
            plt.title(
                "%s, Label: [%.2f %.2f]"
                % (
                    command_text(n),
                    float(label_batch[n][0]),
                    float(label_batch[n][1]),
                )
            )
        else:
            plt.title(
                "%s, Label: [%.2f %.2f], Pred: [%.2f %.2f]"
                % (
                    command_text(n),
                    float(label_batch[n][0]),
                    float(label_batch[n][1]),
                    float(pred_batch[n][0]),
                    float(pred_batch[n][1]),
                )
            )
        plt.axis("off")


def savefig(path):
    plt.savefig(path, bbox_inches="tight")
    plt.clf()


def generate_tflite(path, filename):
    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path, filename))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model


def save_tflite(tflite_model, path, filename):
    open(os.path.join(path, filename + ".tflite"), "wb").write(tflite_model)


def load_tflite(*parts):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(*parts))
    return interpreter


def load_model(model_path, loss_fn, metric_list, custom_objects):
    model: tf.keras.Model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects, compile=False
    )
    model.compile(loss=loss_fn, metrics=metric_list)
    return model


def compare_tf_tflite(
    model, tflite_model, img=None, cmd=None, policy="autopilot", debug=False
):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if debug:
        print("input_details:", input_details)
        print("output_details:", output_details)

    if policy == "autopilot":
        # A memory model (donkey_memory) has mem_input where the others have cmd_input.
        if any("mem_input" in d["name"] for d in input_details):
            command_input_name = "mem_input"
        else:
            command_input_name = "cmd_input"
    elif policy == "point_goal_nav":
        command_input_name = "goal_input"
    else:
        raise Exception("Unknown policy")

    # Test the TensorFlow Lite model on input data. If no data provided, generate random data.
    # Every input is filled, keyed by its Keras name (the converter turns img_input into
    # serving_default_img_input:0), so a memory model's extra cmd_input gets data too.
    input_data = {}
    for input_detail in input_details:
        name = input_detail["name"].replace("serving_default_", "").split(":")[0]
        if name == "img_input" and img is not None:
            print(img)
            value = img
        elif name == command_input_name and cmd is not None:
            print(cmd)
            value = cmd[0]
        else:
            value = np.array(
                np.random.random_sample(input_detail["shape"]), dtype=np.float32
            )
        input_data[name] = value
        interpreter.set_tensor(input_detail["index"], value)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tflite_results = interpreter.get_tensor(output_details[0]["index"])
    tf_results = model.predict(
        tuple(tf.constant(input_data[name]) for name in model.input_names)
    )
    print("tflite:", tflite_results)
    print("tf:", tf_results)

    # Compare the result.
    for tf_result, tflite_result in zip(tf_results, tflite_results):
        print(
            "Almost equal (10% tolerance):",
            np.allclose(tf_result, tflite_result, rtol=0.1),
        )
        # np.testing.assert_almost_equal(tf_result, tflite_result, decimal=2)


def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def load_img(file_path, is_crop=False):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
    if is_crop:
        img = tf.image.crop_to_bounding_box(
            img, tf.shape(img)[0] - 90, tf.shape(img)[1] - 160, 90, 160
        )
    return img


def read_csv_dict(csv_path):
    logs = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            logs.append(row)
    return logs
