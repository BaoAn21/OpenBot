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


# The constant in sq_weighted_mse_angle's weight: a straight frame weighs 0.05^2 against
# 2.05^2 at full lock, ~1,700x less, so the model is barely trained to hold 0 on a
# straight (fwd_s1-2 cil predicted ~90/255 there).
TURN_FLOOR = 0.05


def sq_weighted_mse_angle(y_true, y_pred):
    # The official OpenBot loss, ported to (steering, throttle). Upstream works on
    # (left, right) with angle = right - left; since left = t + s and right = t - s
    # (Control.fromLeftRight), that angle is -2 * steering and its
    # w^2 * (MSE + angle error^2) reduces to w^2 * (dt^2 + 5 * ds^2): steering
    # errors cost 5x throttle errors, and samples are weighted by turn sharpness
    # only, so the rare turns are not drowned out by straight driving.
    # Two upstream quirks are dropped: its weight |angle + 0.05| is made symmetric
    # (|angle| + 0.05), and the angle error is per sample instead of the batch mean
    # that mean_squared_error returns for a 1-D tensor.
    return _turn_weighted(y_true, y_pred, TURN_FLOOR)


def _turn_weighted(y_true, y_pred, floor):
    weight = 2.0 * tf.math.abs(angle(y_true)) + floor
    steering_err = tf.math.square(angle(y_true) - angle(y_pred))
    throttle_err = tf.math.square(throttle(y_true) - throttle(y_pred))
    return tf.math.square(weight) * (throttle_err + 5.0 * steering_err)


def turn_weighted(floor=TURN_FLOOR):
    # --turn_floor: sq_weighted_mse_angle with a larger constant, so straights count
    # more (0.5: a full-lock frame weighs 26x a straight one; 1.0: 9x). The default is
    # sq_weighted_mse_angle itself.
    if floor == TURN_FLOOR:
        return sq_weighted_mse_angle

    def loss(y_true, y_pred):
        return _turn_weighted(y_true, y_pred, floor)

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
