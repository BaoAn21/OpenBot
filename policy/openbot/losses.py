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


# The constant in sq_weighted_mse_angle's weight, upstream's 0.05.
TURN_FLOOR = 0.05


def to_left_right(y):
    # (steering, throttle) back to upstream's (left, right): Control.fromLeftRight
    # inverted, left = throttle + steering, right = throttle - steering.
    return tf.stack([throttle(y) + angle(y), throttle(y) - angle(y)], axis=1)


def sq_weighted_mse_angle(y_true, y_pred):
    # The official OpenBot loss, unchanged: the labels are turned back into (left,
    # right) and upstream's code runs on them line for line, so the only difference
    # from OpenBot is the Ackermann label. That includes upstream's own behaviour:
    # the weight |angle + 0.05| is slightly asymmetric, and mean_squared_error on the
    # 1-D angle tensor is the batch mean, so every sample gets the batch's average
    # angle error and only the (left, right) MSE term is weighted per sample.
    y_true, y_pred = to_left_right(y_true), to_left_right(y_pred)
    angle_true = y_true[:, 1] - y_true[:, 0]
    angle_pred = y_pred[:, 1] - y_pred[:, 0]
    weight = tf.math.abs(angle_true + TURN_FLOOR)
    return tf.math.square(weight) * (
        tf.keras.losses.mean_squared_error(angle_true, angle_pred)
        + tf.keras.losses.mean_squared_error(y_true, y_pred)
    )


def turn_weighted(floor=TURN_FLOOR):
    # --turn_floor: an experiment off the OpenBot baseline. sq_weighted_mse_angle with
    # the weight |angle| + floor instead of |angle + 0.05| (inside the abs, a large
    # floor would zero the weight at angle = -floor), so straights count more. The
    # default is sq_weighted_mse_angle itself.
    if floor == TURN_FLOOR:
        return sq_weighted_mse_angle

    def loss(y_true, y_pred):
        y_true, y_pred = to_left_right(y_true), to_left_right(y_pred)
        angle_true = y_true[:, 1] - y_true[:, 0]
        angle_pred = y_pred[:, 1] - y_pred[:, 0]
        weight = tf.math.abs(angle_true) + floor
        return tf.math.square(weight) * (
            tf.keras.losses.mean_squared_error(angle_true, angle_pred)
            + tf.keras.losses.mean_squared_error(y_true, y_pred)
        )

    loss.__name__ = f"sq_weighted_mse_angle_floor{floor:g}"
    return loss


def steer_only(loss_fn):
    # --steer_only: loss_fn without its throttle term. The throttle label is swapped
    # for the model's own throttle with the gradient stopped, so the throttle error and
    # its gradient are exactly 0, and everything else - the steering term, and the turn
    # weight, which reads only the steering label - is loss_fn's own, unchanged. For
    # mse_raw that leaves half the steering squared error (the mean over two columns),
    # the same steering gradient as before. Only for losses whose y_pred is
    # (steering, throttle): not ce_angle.
    def loss(y_true, y_pred):
        y_true = tf.concat([y_true[:, :1], tf.stop_gradient(y_pred[:, 1:2])], axis=1)
        return loss_fn(y_true, y_pred)

    loss.__name__ = loss_fn.__name__ + "_steer_only"
    return loss


# cil_mobile_fast_cat: spread of the soft steering target, in bins, and the weight of the
# throttle error against the steering cross-entropy (which is ~1-3 where a squared
# throttle error is ~0.01, so without it throttle would barely train).
STEER_SIGMA_BINS = 1.0
CAT_THROTTLE_WEIGHT = 10.0


def ce_angle(y_true, y_pred):
    # Loss of cil_mobile_fast_cat, whose y_pred is (steering, throttle, steering bin
    # probabilities). The steering label becomes a Gaussian over the bin centres
    # (STEER_SIGMA_BINS wide), so being one bin off costs less than being far off, and
    # the decoded steering can fall between centres; the probabilities are trained on it
    # with cross-entropy. Throttle stays a squared error.
    # Unweighted, as donkeycar's KerasCategorical: with sq_weighted_mse_angle's turn
    # weight w^2 (0.0025 straight, up to 4.2 on a hard turn) the "straight" bins got no
    # signal and the first run never predicted them, though 39% of its test frames are
    # straight driving (EXPERIMENTS.md).
    probs = y_pred[:, 2:]
    n = probs.shape[-1]
    centers = tf.linspace(-1.0, 1.0, n)
    sigma = STEER_SIGMA_BINS * 2.0 / (n - 1)
    steer = tf.clip_by_value(angle(y_true), -1.0, 1.0)
    target = tf.math.exp(-0.5 * tf.math.square((steer[:, None] - centers) / sigma))
    target = target / tf.reduce_sum(target, axis=-1, keepdims=True)
    steering_ce = tf.keras.losses.categorical_crossentropy(target, probs)
    throttle_err = tf.math.square(throttle(y_true) - throttle(y_pred))
    return steering_ce + CAT_THROTTLE_WEIGHT * throttle_err


def weighted_mse_raw_angle(y_gt, y_pred):
    return angle_weight(y_gt) * (mse_raw(y_gt, y_pred) + mse_angle(y_gt, y_pred))


def mae_raw_weighted_mse_angle(y_gt, y_pred):
    return mae_raw(y_gt, y_pred) + weighted_mse_angle(y_gt, y_pred)
