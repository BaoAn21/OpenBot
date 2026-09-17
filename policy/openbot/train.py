from dataclasses import dataclass
import inspect
import os
import re
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import absl.logging

from . import (
    associate_frames,
    callbacks,
    dataloader,
    dataset_dir,
    data_augmentation,
    losses,
    metrics,
    models,
    models_dir,
    tfrecord,
    tfrecord_seq,
    tfrecord_utils,
    utils,
)

absl.logging.set_verbosity(absl.logging.ERROR)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# On Mac you may encounter an error related to OMP, this is a workaround, but slows down the code
# https://github.com/dmlc/xgboost/issues/1715
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

AUTOTUNE = tf.data.experimental.AUTOTUNE

if tf.test.gpu_device_name():
    print("Default GPU Device:{}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF if you have one.")


@dataclass
class Hyperparameters:
    MODEL: str = "pilot_net"
    POLICY: str = "autopilot"

    TRAIN_BATCH_SIZE: int = 128
    TEST_BATCH_SIZE: int = 16
    LEARNING_RATE: float = 0.0003
    NUM_EPOCHS: int = 100

    BATCH_NORM: bool = True
    IS_CROP: bool = False
    FLIP_AUG: bool = False
    CMD_AUG: bool = False

    # Frame offsets back from the current frame, comma separated and starting at 0
    # (each step is one timeline position). Empty means the single-frame policy.
    SEQ_OFFSETS: str = ""
    # Drop leading/trailing stationary runs and cut interior ones to KEEP_STATIONARY.
    TRIM_STATIONARY: bool = False
    KEEP_STATIONARY: int = tfrecord_seq.DEFAULT_KEEP_STATIONARY

    USE_LAST: bool = False

    WANDB: bool = False

    @property
    def seq_offsets(self):
        if not self.SEQ_OFFSETS:
            return (0,)
        return tfrecord_seq.parse_offsets(self.SEQ_OFFSETS)

    @property
    def SEQ_LEN(self):
        return len(self.seq_offsets)

    @classmethod
    def parse(cls, name):
        m = re.match(
            r".*_((cil|pilot)_.+)_lr(\d+.\d+)_bz(\d+)(_bn)?(_flip)?(_cmd)?", name
        )
        params = Hyperparameters(m[1], int(m[4]), int(m[4]), float(m[3]))
        params.BATCH_NORM = m[5] is not None
        params.FLIP_AUG = m[6] is not None
        params.CMD_AUG = m[7] is not None
        return params

    def __str__(self):
        model_name = "{0}_lr{1}_bz{2}".format(
            self.MODEL, self.LEARNING_RATE, self.TRAIN_BATCH_SIZE
        )

        if self.BATCH_NORM:
            model_name += "_bn"
        if self.FLIP_AUG:
            model_name += "_flip"
        if self.CMD_AUG:
            model_name += "_cmd"
        if self.SEQ_LEN > 1:
            # Part of the name so runs with different window shapes cannot land in
            # the same checkpoint directory.
            model_name += "_seq" + tfrecord_seq.offsets_tag(self.seq_offsets)
            if self.TRIM_STATIONARY:
                model_name += f"_trim{self.KEEP_STATIONARY}"

        return model_name


@dataclass
class Training:
    def __init__(self, params: Hyperparameters):
        self.hyperparameters = params
        self.NETWORK_IMG_WIDTH = 0
        self.NETWORK_IMG_HEIGHT = 0
        self.INITIAL_EPOCH = 0
        self.train_data_dir = ""
        self.test_data_dir = ""
        self.train_datasets = []
        self.test_datasets = []
        self.redo_matching = True
        self.remove_zeros = True
        self.image_count_train = 0
        self.image_count_test = 0
        self.train_ds = None
        self.test_ds = None
        self.history = None
        self.model_name = ""
        self.dataset_name = "openbot"
        self.checkpoint_path = ""
        self.log_path = ""
        self.loss_fn = None
        self.metric_list = None
        self.custom_objects = None


class CancelledException(BaseException):
    pass


class MyCallback(tf.keras.callbacks.Callback):
    cancelled: threading.Event
    model: tf.keras.Model

    def __init__(self, broadcast, cancelled, show_progress=False):
        super().__init__()
        self.broadcast = broadcast
        self.cancelled = cancelled
        self.show_progress = show_progress

        self.epoch = 0
        self.step = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        msg = logs.copy()
        msg["epoch"] = epoch
        print("on_epoch_end", msg)
        self.broadcast("logs", msg)

    def on_batch_end(self, batch, logs=None):
        if self.cancelled.is_set():
            raise CancelledException
        self.step = batch + 1
        epochs = self.params["epochs"]
        steps = self.params["steps"]
        if self.show_progress:
            self.broadcast(
                "progress",
                dict(
                    epoch=round(100 * self.step / steps, 1),
                    train=round(
                        100 * (self.epoch * steps + self.step) / (epochs * steps), 1
                    ),
                ),
            )


def process_data(tr: Training):
    tr.train_datasets = utils.list_dirs(tr.train_data_dir)
    tr.test_datasets = utils.list_dirs(tr.test_data_dir)

    print("Train Datasets: ", len(tr.train_datasets))
    print("Test Datasets: ", len(tr.test_datasets))

    # 1ms
    max_offset = 1e3
    train_frames = associate_frames.match_frame_ctrl_input(
        tr.train_data_dir,
        tr.train_datasets,
        max_offset,
        redo_matching=tr.redo_matching,
        remove_zeros=tr.remove_zeros,
        policy=tr.hyperparameters.POLICY,
    )

    test_frames = associate_frames.match_frame_ctrl_input(
        tr.test_data_dir,
        tr.test_datasets,
        max_offset,
        redo_matching=tr.redo_matching,
        remove_zeros=tr.remove_zeros,
        policy=tr.hyperparameters.POLICY,
    )

    tr.image_count_train = len(train_frames)
    tr.image_count_test = len(test_frames)
    print(
        "There are %d train images and %d test images"
        % (tr.image_count_train, tr.image_count_test)
    )


def load_tfrecord(tr: Training, verbose=0):
    seq_len = tr.hyperparameters.SEQ_LEN
    if seq_len > 1:
        autopilot_parse_fn = tfrecord_seq.make_parse_fn(seq_len)
    else:
        autopilot_parse_fn = tfrecord_utils.parse_tfrecord_fn_autopilot

    def process_train_sample(features):
        # image = tf.image.resize(features["image"], size=(224, 224))
        image = features["image"]

        if tr.hyperparameters.POLICY == "autopilot":
            cmd_input = features["cmd"]
            label = [features["left"], features["right"]]
            image = data_augmentation.augment_img_seq(image, seq_len)
            if tr.hyperparameters.FLIP_AUG:
                image, cmd_input, label = data_augmentation.flip_sample(
                    image, cmd_input, label
                )
            if tr.hyperparameters.CMD_AUG:
                cmd_input = data_augmentation.augment_cmd(cmd_input)
        elif tr.hyperparameters.POLICY == "point_goal_nav":
            image = tf.image.crop_to_bounding_box(
                image, tf.shape(image)[0] - 90, tf.shape(image)[1] - 160, 90, 160
            )
            cmd_input = [features["dist"], features["sinYaw"], features["cosYaw"]]
            label = [features["left"], features["right"]]
            image = data_augmentation.augment_img(image)
            if tr.hyperparameters.FLIP_AUG:
                print(
                    "Image flip augmentation is not implemented for Point Goal Navigation."
                )
            if tr.hyperparameters.CMD_AUG:
                print(
                    "Command augmentation is not implemented for Point Goal Navigation."
                )

        return (image, cmd_input), label

    def process_test_sample(features):
        image = features["image"]

        if tr.hyperparameters.POLICY == "autopilot":
            cmd_input = features["cmd"]

        elif tr.hyperparameters.POLICY == "point_goal_nav":
            image = tf.image.crop_to_bounding_box(
                image, tf.shape(image)[0] - 90, tf.shape(image)[1] - 160, 90, 160
            )
            cmd_input = [features["dist"], features["sinYaw"], features["cosYaw"]]

        label = [features["left"], features["right"]]
        return (image, cmd_input), label

    if tr.hyperparameters.POLICY == "autopilot":
        train_dataset = (
            tf.data.TFRecordDataset(tr.train_data_dir, num_parallel_reads=AUTOTUNE)
            .map(autopilot_parse_fn, num_parallel_calls=AUTOTUNE)
            .map(process_train_sample, num_parallel_calls=AUTOTUNE)
        )
    elif tr.hyperparameters.POLICY == "point_goal_nav":
        train_dataset = (
            tf.data.TFRecordDataset(tr.train_data_dir, num_parallel_reads=AUTOTUNE)
            .map(
                tfrecord_utils.parse_tfrecord_fn_point_goal_nav,
                num_parallel_calls=AUTOTUNE,
            )
            .map(process_train_sample, num_parallel_calls=AUTOTUNE)
        )

    # Obtains the images shapes of records from .tfrecords.
    for (image, cmd_input), label in train_dataset.take(1):
        shape = image.numpy().shape
        tr.NETWORK_IMG_HEIGHT = shape[0]
        tr.NETWORK_IMG_WIDTH = shape[1]
        print("Image shape: ", shape)
        print("Command: ", cmd_input.numpy())
        print("Label: ", label.numpy())

    if tr.hyperparameters.POLICY == "autopilot":
        test_dataset = (
            tf.data.TFRecordDataset(tr.test_data_dir, num_parallel_reads=AUTOTUNE)
            .map(autopilot_parse_fn, num_parallel_calls=AUTOTUNE)
            .map(process_test_sample, num_parallel_calls=AUTOTUNE)
        )
    elif tr.hyperparameters.POLICY == "point_goal_nav":
        test_dataset = (
            tf.data.TFRecordDataset(tr.test_data_dir, num_parallel_reads=AUTOTUNE)
            .map(
                tfrecord_utils.parse_tfrecord_fn_point_goal_nav,
                num_parallel_calls=AUTOTUNE,
            )
            .map(process_test_sample, num_parallel_calls=AUTOTUNE)
        )

    # Obtains the total number of records from .tfrecords file
    # https://stackoverflow.com/questions/40472139/obtaining-total-number-of-records-from-tfrecords-file-in-tensorflow
    tr.image_count_train = sum(1 for _ in train_dataset)
    print("Number of training instances: ", tr.image_count_train)

    tr.image_count_test = sum(1 for _ in test_dataset)
    print("Number of test instances: ", tr.image_count_test)

    # Prepare train and test datasets for training
    tr.train_ds = (
        train_dataset.shuffle(tr.hyperparameters.TRAIN_BATCH_SIZE * 10)
        .repeat()
        .batch(tr.hyperparameters.TRAIN_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    tr.test_ds = test_dataset.batch(tr.hyperparameters.TEST_BATCH_SIZE).prefetch(
        AUTOTUNE
    )


def load_data(tr: Training, verbose=0):
    # list_train_ds = tf.data.Dataset.list_files(train_frames)
    # list_test_ds = tf.data.Dataset.list_files(test_frames)
    list_train_ds = tf.data.Dataset.list_files(
        [str(tr.train_data_dir + "/" + ds + "/*/images/*") for ds in tr.train_datasets]
    )
    list_test_ds = tf.data.Dataset.list_files(
        [str(tr.test_data_dir + "/" + ds + "/*/images/*") for ds in tr.test_datasets]
    )
    train_data = dataloader.dataloader(
        tr.train_data_dir, tr.train_datasets, tr.hyperparameters.POLICY
    )
    test_data = dataloader.dataloader(
        tr.test_data_dir, tr.test_datasets, tr.hyperparameters.POLICY
    )

    if verbose:
        for f in list_train_ds.take(5):
            print(f.numpy())
        print()
        for f in list_test_ds.take(5):
            print(f.numpy())
        print()
    print("Number of train samples: %d" % len(train_data.labels))
    print("Number of test samples: %d" % len(test_data.labels))

    def process_train_path(file_path):
        cmd_input, label = train_data.get_label(
            tf.strings.regex_replace(file_path, "[/\\\\]", "/")
        )
        img = utils.load_img(file_path, tr.hyperparameters.IS_CROP)
        img = data_augmentation.augment_img(img)
        if tr.hyperparameters.FLIP_AUG:
            img, cmd_input, label = data_augmentation.flip_sample(img, cmd_input, label)
        if tr.hyperparameters.CMD_AUG:
            cmd_input = data_augmentation.augment_cmd(cmd_input)
        return (img, cmd_input), label

    def process_test_path(file_path):
        cmd_input, label = test_data.get_label(
            tf.strings.regex_replace(file_path, "[/\\\\]", "/")
        )
        img = utils.load_img(file_path, tr.hyperparameters.IS_CROP)
        return (img, cmd_input), label

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_train_ds.map(process_train_path, num_parallel_calls=4)
    for (image, cmd_input), label in labeled_ds.take(1):
        shape = image.numpy().shape
        tr.NETWORK_IMG_HEIGHT = shape[0]
        tr.NETWORK_IMG_WIDTH = shape[1]
        print("Image shape: ", shape)
        print("Command: ", cmd_input.numpy())
        print("Label: ", label.numpy())
    tr.train_ds = utils.prepare_for_training(
        ds=labeled_ds,
        batch_sz=tr.hyperparameters.TRAIN_BATCH_SIZE,
        shuffle_buffer_sz=100 * tr.hyperparameters.TRAIN_BATCH_SIZE,
        prefetch_buffer_sz=10 * tr.hyperparameters.TRAIN_BATCH_SIZE,
    )
    test_ds = list_test_ds.map(process_test_path, num_parallel_calls=4)
    test_ds = test_ds.batch(tr.hyperparameters.TEST_BATCH_SIZE)
    tr.test_ds = test_ds.prefetch(buffer_size=10 * tr.hyperparameters.TRAIN_BATCH_SIZE)


def visualize_train_data(tr: Training):
    utils.show_batch(dataset=tr.train_ds, policy=tr.hyperparameters.POLICY, model=None)
    try:
        utils.savefig(os.path.join(models_dir, "train_preview.png"))
    except OSError as err:
        # This preview lands in a path shared by every run, so it can be left behind
        # owned by another user (a docker run as root, for instance). It is a
        # debugging aid - not worth losing a training run over.
        print(f"Could not write train preview, continuing anyway: {err}")


def do_training(tr: Training, callback: tf.keras.callbacks.Callback, verbose=0):
    tr.model_name = tr.dataset_name + "_" + str(tr.hyperparameters)
    tr.checkpoint_path = os.path.join(models_dir, tr.model_name, "checkpoints")
    tr.log_path = os.path.join(models_dir, tr.model_name, "logs")

    tr.custom_objects = {
        "direction_metric": metrics.direction_metric,
        "angle_metric": metrics.angle_metric,
        "throttle_metric": metrics.throttle_metric,
        "throttle_direction_metric": metrics.throttle_direction_metric,
    }

    if tr.hyperparameters.WANDB:
        import wandb
        from wandb.keras import WandbCallback

        wandb.init(project="openbot")

        config = wandb.config
        config.epochs = tr.hyperparameters.NUM_EPOCHS
        config.learning_rate = tr.hyperparameters.LEARNING_RATE
        config.batch_size = tr.hyperparameters.TRAIN_BATCH_SIZE
        config["model_name"] = tr.model_name

    resume_training = False
    model: tf.keras.Model

    if tr.hyperparameters.USE_LAST:
        try:
            dirs = utils.list_dirs(tr.checkpoint_path)
            last_checkpoint = os.path.join(tr.checkpoint_path, "cp-last.ckpt")
            os.path.join(tr.checkpoint_path, last_checkpoint)
            model = tf.keras.models.load_model(
                last_checkpoint,
                custom_objects=tr.custom_objects,
                compile=False,
            )
            log_file = open(os.path.join(tr.log_path, "log.csv"), "r")
            tr.INITIAL_EPOCH = int(log_file.readlines()[-1].split(",")[0]) + 1
            log_file.close()
            resume_training = True
            print(f"Resuming from checkpoint: {last_checkpoint}")
            # print(f"Resuming from saved model.")
        except FileNotFoundError as err:
            print("No checkpoint or log file found, training new model!")
            print(err)
        except Exception as err:
            print(err)
            raise

    if not resume_training:
        build_model = getattr(models, tr.hyperparameters.MODEL)
        model_kwargs = {}
        # Only the sequence architectures take seq_len; the single-frame ones keep
        # their original four-argument signature.
        if "seq_len" in inspect.signature(build_model).parameters:
            model_kwargs["seq_len"] = tr.hyperparameters.SEQ_LEN
        model = build_model(
            tr.NETWORK_IMG_WIDTH,
            tr.NETWORK_IMG_HEIGHT,
            tr.hyperparameters.BATCH_NORM,
            tr.hyperparameters.POLICY,
            **model_kwargs,
        )
        # Define the directory where the model image should be saved (if not already created)
        model_dir = os.path.join(models_dir, tr.model_name)
        os.makedirs(model_dir, exist_ok=True)
        dot_img_file = os.path.join(model_dir, "model.png")
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    callback.broadcast("model", tr.model_name)

    if tr.hyperparameters.POLICY == "autopilot":
        tr.loss_fn = losses.sq_weighted_mse_angle
    elif tr.hyperparameters.POLICY == "point_goal_nav":
        tr.loss_fn = losses.mae_raw_weighted_mse_angle

    tr.metric_list = [
        "mean_absolute_error",
        tr.custom_objects["direction_metric"],
        tr.custom_objects["angle_metric"],
        tr.custom_objects["throttle_metric"],
        tr.custom_objects["throttle_direction_metric"],
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=tr.hyperparameters.LEARNING_RATE)

    model.compile(optimizer=optimizer, loss=tr.loss_fn, metrics=tr.metric_list)
    if verbose:
        print(model.summary())

    if verbose:
        print(tr.model_name)

    STEPS_PER_EPOCH = np.ceil(
        tr.image_count_train / tr.hyperparameters.TRAIN_BATCH_SIZE
    )
    callback.broadcast("message", "Fit model...")
    callback_list = [
        callbacks.checkpoint_last_cb(tr.checkpoint_path),
        callbacks.checkpoint_best_train_cb(tr.checkpoint_path),
        callbacks.checkpoint_best_val_cb(tr.checkpoint_path),
        callbacks.tensorboard_cb(tr.log_path),
        callbacks.logger_cb(tr.log_path, resume_training),
        callback,
    ]

    if tr.hyperparameters.WANDB:
        callback_list += [WandbCallback()]
    tr.history = model.fit(
        tr.train_ds,
        epochs=tr.hyperparameters.NUM_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        initial_epoch=tr.INITIAL_EPOCH,
        validation_data=tr.test_ds,
        verbose=verbose,
        callbacks=callback_list,
    )

    if tr.hyperparameters.WANDB:
        wandb.save(tr.log_path)
        wandb.finish()

    callback.broadcast("message", "...Done")


def do_evaluation(tr: Training, callback: tf.keras.callbacks.Callback, verbose=0):
    callback.broadcast("message", "Generate plots...")

    x = np.arange(tr.INITIAL_EPOCH + 1, tr.history.params["epochs"] + 1, 1)

    plt.figure().gca().xaxis.get_major_locator().set_params(integer=True)
    plt.plot(x, tr.history.history["loss"], label="loss")
    plt.plot(x, tr.history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    utils.savefig(os.path.join(tr.log_path, "loss.png"))

    plt.figure().gca().xaxis.get_major_locator().set_params(integer=True)
    plt.plot(x, tr.history.history["mean_absolute_error"], label="mean_absolute_error")
    plt.plot(
        x,
        tr.history.history["val_mean_absolute_error"],
        label="val_mean_absolute_error",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend(loc="upper right")
    utils.savefig(os.path.join(tr.log_path, "error.png"))

    plt.figure().gca().xaxis.get_major_locator().set_params(integer=True)
    plt.plot(x, tr.history.history["direction_metric"], label="direction_metric")
    plt.plot(
        x, tr.history.history["val_direction_metric"], label="val_direction_metric"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Direction Metric")
    plt.legend(loc="lower right")
    utils.savefig(os.path.join(tr.log_path, "direction.png"))

    plt.figure().gca().xaxis.get_major_locator().set_params(integer=True)
    plt.plot(x, tr.history.history["angle_metric"], label="angle_metric")
    plt.plot(x, tr.history.history["val_angle_metric"], label="val_angle_metric")
    plt.xlabel("Epoch")
    plt.ylabel("Angle Metric")
    plt.legend(loc="lower right")
    utils.savefig(os.path.join(tr.log_path, "angle.png"))

    plt.figure().gca().xaxis.get_major_locator().set_params(integer=True)
    plt.plot(x, tr.history.history["throttle_metric"], label="throttle_metric")
    plt.plot(x, tr.history.history["val_throttle_metric"], label="val_throttle_metric")
    plt.plot(
        x,
        tr.history.history["throttle_direction_metric"],
        label="throttle_direction_metric",
    )
    plt.plot(
        x,
        tr.history.history["val_throttle_direction_metric"],
        label="val_throttle_direction_metric",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Throttle Metric")
    plt.legend(loc="lower right")
    utils.savefig(os.path.join(tr.log_path, "throttle.png"))

    def summarize(title, index):
        h = tr.history.history
        print(
            "%s - angle: %.4f, val_angle: %.4f, direction: %.4f, val_direction: %.4f, "
            "throttle: %.4f, val_throttle: %.4f, throttle_dir: %.4f, val_throttle_dir: %.4f"
            % (
                title,
                h["angle_metric"][index],
                h["val_angle_metric"][index],
                h["direction_metric"][index],
                h["val_direction_metric"][index],
                h["throttle_metric"][index],
                h["val_throttle_metric"][index],
                h["throttle_direction_metric"][index],
                h["val_throttle_direction_metric"][index],
            )
        )

    callback.broadcast("message", "Generate tflite models...")
    checkpoint_path = tr.checkpoint_path
    print("checkpoint_path", checkpoint_path)

    best_train_checkpoint = "cp-best-train.ckpt"
    best_train_tflite = utils.generate_tflite(tr.checkpoint_path, best_train_checkpoint)
    utils.save_tflite(best_train_tflite, tr.checkpoint_path, "best-train")
    best_train_index = np.argmin(np.array(tr.history.history["loss"]))
    summarize("Best Train Checkpoint (epoch %s)" % best_train_index, best_train_index)

    best_val_checkpoint = "cp-best-val.ckpt"
    best_val_tflite = utils.generate_tflite(tr.checkpoint_path, best_val_checkpoint)
    utils.save_tflite(best_val_tflite, tr.checkpoint_path, "best")
    utils.save_tflite(best_val_tflite, tr.checkpoint_path, "best-val")
    best_val_index = np.argmin(np.array(tr.history.history["val_loss"]))
    summarize("Best Val Checkpoint (epoch %s)" % best_val_index, best_val_index)

    last_checkpoint = "cp-last.ckpt"
    last_tflite = utils.generate_tflite(tr.checkpoint_path, last_checkpoint)
    utils.save_tflite(last_tflite, tr.checkpoint_path, "last")
    summarize("Last Checkpoint", -1)

    callback.broadcast("message", "Evaluate model...")
    last_model = utils.load_model(
        os.path.join(tr.checkpoint_path, last_checkpoint),
        tr.loss_fn,
        tr.metric_list,
        tr.custom_objects,
    )
    # test_loss, test_acc, test_dir, test_ang
    res = last_model.evaluate(
        tr.test_ds,
        steps=tr.image_count_test / tr.hyperparameters.TEST_BATCH_SIZE,
        verbose=2,
    )
    print(res)

    utils.show_batch(
        dataset=tr.test_ds, policy=tr.hyperparameters.POLICY, model=last_model
    )
    utils.savefig(os.path.join(tr.log_path, "test_preview.png"))
    utils.compare_tf_tflite(last_model, last_tflite, policy=tr.hyperparameters.POLICY)


def tfrecords_path(params: Hyperparameters):
    """Where the tfrecords for this configuration live.

    Multi-frame records go in their own directory named after the window shape, so
    changing seq_len or seq_stride can never silently reuse records built for a
    different one.
    """
    if params.SEQ_LEN > 1:
        return os.path.join(
            dataset_dir,
            tfrecord_seq.tfrecords_dir_name(
                params.seq_offsets, params.TRIM_STATIONARY, params.KEEP_STATIONARY
            ),
        )
    return os.path.join(dataset_dir, "tfrecords")


def start_train(
    params: Hyperparameters, callback: MyCallback, verbose=0, no_tf_record=False
):
    tr = Training(params)
    if no_tf_record:
        callback.broadcast("message", "Processing data...")
        tr.train_data_dir = os.path.join(dataset_dir, "train_data")
        tr.test_data_dir = os.path.join(dataset_dir, "test_data")
        process_data(tr)
        callback.broadcast("message", "Loading data...")
        load_data(tr, verbose)
    else:
        callback.broadcast("message", "Loading data from tfrecord...")
        records_dir = tfrecords_path(params)
        tr.train_data_dir = os.path.join(records_dir, "train.tfrec")
        tr.test_data_dir = os.path.join(records_dir, "test.tfrec")
        if not os.path.isfile(tr.train_data_dir):
            if params.SEQ_LEN > 1:
                hint = f"python -m openbot.tfrecord_seq --seq_offsets {params.SEQ_OFFSETS}"
                if params.TRIM_STATIONARY:
                    hint += (
                        f" --trim_stationary --keep_stationary "
                        f"{params.KEEP_STATIONARY}"
                    )
            else:
                hint = "rerun with --create_tf_record"
            raise FileNotFoundError(
                f"No tfrecords at {tr.train_data_dir}. Build them first: {hint}"
            )
        load_tfrecord(tr, verbose)

    visualize_train_data(tr)
    callback.broadcast("preview")
    do_training(tr, callback, verbose)
    do_evaluation(tr, callback, verbose)

    return tr


def create_tfrecord(callback: MyCallback, policy="autopilot", params=None):
    callback.broadcast(
        "message", "Converting data to tfrecord (this may take some time)..."
    )

    if params is not None and params.SEQ_LEN > 1:
        out_dir = tfrecords_path(params)
        for split, name in (("train_data", "train.tfrec"), ("test_data", "test.tfrec")):
            tfrecord_seq.convert_dataset(
                os.path.join(dataset_dir, split),
                out_dir,
                name,
                params.seq_offsets,
                params.TRIM_STATIONARY,
                params.KEEP_STATIONARY,
            )
        return

    tfrecord.convert_dataset(
        os.path.join(dataset_dir, "train_data"),
        os.path.join(dataset_dir, "tfrecords"),
        "train.tfrec",
        policy=policy,
    )
    tfrecord.convert_dataset(
        os.path.join(dataset_dir, "test_data"),
        os.path.join(dataset_dir, "tfrecords"),
        "test.tfrec",
        policy=policy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input Arguments")
    parser.add_argument(
        "--no_tf_record",
        action="store_true",
        help="do not load a tfrecord but a directory of files",
    )
    parser.add_argument(
        "--create_tf_record", action="store_true", help="create a new tfrecord"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pilot_net",
        choices=[
            "cil_mobile",
            "cil_mobile_fast",
            "cil",
            "pilot_net",
            "pilot_net_seq",
        ],
        help="network architecture (default: pilot_net)",
    )
    parser.add_argument(
        "--seq_offsets",
        type=str,
        default=None,
        help="stacked-frame policy: comma separated frame offsets back from the "
        "current frame, starting at 0 (e.g. 0,2,5,12,24). Each step is one recorded "
        "frame, ~33ms. Omit for the single-frame policy",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="stacked-frame policy with uniform spacing: number of frames "
        "(use with --seq_stride instead of --seq_offsets)",
    )
    parser.add_argument(
        "--seq_stride",
        type=int,
        default=None,
        help="stacked-frame policy with uniform spacing: gap between frames",
    )
    parser.add_argument(
        "--trim_stationary",
        action="store_true",
        help="stacked-frame policy: drop leading and trailing stationary runs and "
        "cut interior ones to --keep_stationary frames",
    )
    parser.add_argument(
        "--keep_stationary",
        type=int,
        default=tfrecord_seq.DEFAULT_KEEP_STATIONARY,
        help="frames to keep from an interior stationary run when trimming "
        f"(default: {tfrecord_seq.DEFAULT_KEEP_STATIONARY})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="number of training epochs (default: 128)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="number of epochs (default: 100)"
    )
    parser.add_argument("--batch_norm", action="store_true", help="use batch norm")
    parser.add_argument(
        "--flip_aug",
        action="store_true",
        help="randomly flip images and controls for augmentation",
    )
    parser.add_argument(
        "--cmd_aug",
        action="store_true",
        help="add noise to command input for augmentation",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume previous training"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="training logs with weights & biases"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="autopilot",
        choices=["autopilot", "point_goal_nav"],
        help="the type of policy to be trained (default: autopilot)",
    )

    args = parser.parse_args()

    params = Hyperparameters()
    params.MODEL = args.model
    params.POLICY = args.policy
    params.TRAIN_BATCH_SIZE = args.batch_size
    params.TEST_BATCH_SIZE = args.batch_size
    params.LEARNING_RATE = args.learning_rate
    params.NUM_EPOCHS = args.num_epochs
    params.BATCH_NORM = args.batch_norm
    params.FLIP_AUG = args.flip_aug
    params.CMD_AUG = args.cmd_aug
    if args.seq_offsets or args.seq_len:
        try:
            offsets = tfrecord_seq.resolve_offsets(
                args.seq_offsets, args.seq_len, args.seq_stride
            )
        except ValueError as err:
            parser.error(str(err))
        params.SEQ_OFFSETS = tfrecord_seq.offsets_tag(offsets).replace("-", ",")

    params.TRIM_STATIONARY = args.trim_stationary
    params.KEEP_STATIONARY = args.keep_stationary
    params.USE_LAST = args.resume
    params.WANDB = args.wandb
    params.IS_CROP = args.policy == "point_goal_nav"

    is_seq_model = (
        "seq_len" in inspect.signature(getattr(models, params.MODEL)).parameters
    )
    if params.SEQ_LEN > 1 and not is_seq_model:
        parser.error(
            f"stacked frames need --model pilot_net_seq, not {params.MODEL}"
        )
    if params.SEQ_LEN == 1 and is_seq_model:
        parser.error(f"--model {params.MODEL} needs --seq_offsets (or --seq_len)")

    def broadcast(event, payload=None):
        print()
        print(event, payload)

    event = threading.Event()
    my_callback = MyCallback(broadcast, event)

    if args.create_tf_record:
        create_tfrecord(my_callback, args.policy, params)

    start_train(params, my_callback, verbose=1, no_tf_record=args.no_tf_record)
