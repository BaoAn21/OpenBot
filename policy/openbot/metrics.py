# Created by Matthias Mueller - Intel Intelligent Systems Lab - 2020

import tensorflow as tf


def angle_metric(y_true, y_pred):
    # Ackermann-style labels: column 0 is the steering value itself, not a
    # left/right wheel-speed difference.
    angle_true = y_true[:, 0]
    angle_pred = y_pred[:, 0]
    return tf.abs(angle_true - angle_pred) < 0.1


def direction_metric(y_true, y_pred):
    angle_true = y_true[:, 0]
    angle_pred = y_pred[:, 0]
    return tf.math.logical_or(
        tf.math.sign(angle_pred) == tf.math.sign(angle_true), tf.abs(angle_pred) < 0.1
    )


def mean_absolute_error(y_true, y_pred):
    # Keras's, over (steering, throttle) only: cil_mobile_fast_cat appends its steering
    # bin probabilities to its output. Same name, so the history key stays the same.
    return tf.reduce_mean(tf.abs(y_true - y_pred[:, :2]), axis=-1)
