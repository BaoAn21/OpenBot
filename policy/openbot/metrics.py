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


def throttle_metric(y_true, y_pred):
    # Column 1 is throttle. Until this existed, every logged metric looked only at
    # steering, so a model that had stopped predicting usable throttle still showed
    # healthy numbers all the way through training.
    return tf.abs(y_true[:, 1] - y_pred[:, 1]) < 0.1


def throttle_direction_metric(y_true, y_pred):
    # Deliberately without the small-magnitude escape hatch that direction_metric has:
    # predicting ~0 throttle is exactly the failure this metric exists to catch, so it
    # must not be scored as correct.
    return tf.math.sign(y_pred[:, 1]) == tf.math.sign(y_true[:, 1])
