# Created by Matthias Mueller - Intel Intelligent Systems Lab - 2020

import tensorflow as tf


def angle(y):
    # Ackermann-style labels: column 0 is the steering value itself, not a
    # left/right wheel-speed difference.
    return y[:, 0]


def throttle(y):
    return y[:, 1]


def angle_weight(y_gt, eps=0.05):
    return tf.math.square(angle(y_gt)) + eps


def command_weight(y_gt, eps=0.05):
    # Up-weights any "non-trivial" command sample - a sharp turn OR strong
    # throttle (e.g. driving straight backward at full speed) - instead of
    # only sharp turns. Weighting by steering alone starves the loss of
    # signal on straight-line driving, since most reverse driving has near
    # -zero steering: it would otherwise be weighted down to ~eps^2, ~175x
    # less than a sharp-turn sample, and the model learns to under-predict
    # throttle whenever steering is small.
    return tf.math.maximum(tf.math.abs(angle(y_gt)), tf.math.abs(throttle(y_gt))) + eps


def mse_raw(y_gt, y_pred):
    return tf.keras.losses.mean_squared_error(y_gt, y_pred)


def mae_raw(y_gt, y_pred):
    return tf.keras.losses.mean_absolute_error(y_gt, y_pred)


def huber_raw(y_gt, y_pred):
    huber = tf.keras.losses.Huber()
    return tf.keras.losses.huber(y_gt, y_pred)


def mse_angle(y_gt, y_pred):
    return tf.keras.losses.mean_squared_error(angle(y_gt), angle(y_pred))


def weighted_mse_raw(y_true, y_pred):
    weight = command_weight(y_true)
    return weight * tf.keras.losses.mean_squared_error(y_true, y_pred)


def weighted_mse_angle(y_gt, y_pred):
    return angle_weight(y_gt) * mse_angle(y_gt, y_pred)


def sq_weighted_mse_angle(y_true, y_pred):
    weight = command_weight(y_true)
    return tf.math.square(weight) * (
        tf.keras.losses.mean_squared_error(y_true, y_pred) + mse_angle(y_true, y_pred)
    )


def weighted_mse_raw_angle(y_gt, y_pred):
    return angle_weight(y_gt) * (mse_raw(y_gt, y_pred) + mse_angle(y_gt, y_pred))


def mae_raw_weighted_mse_angle(y_gt, y_pred):
    return mae_raw(y_gt, y_pred) + weighted_mse_angle(y_gt, y_pred)
