"""
Created by Marcel Santos - Intel Intelligent Systems Lab - 2021
This script implements several routines for data augmentation.
"""

import tensorflow as tf
import numpy as np


def augment_img(img):
    """Color augmentation

    Args:
      img: input image

    Returns:
      img: augmented image
    """
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)
    return img


def augment_img_seq(img, seq_len):
    """Colour augmentation for a stack of seq_len frames.

    Args:
      img: (height, width, 3 * seq_len) stack, frames concatenated oldest-first
      seq_len: number of frames in the stack

    Returns:
      img: augmented stack, same shape

    The frames are unstacked to (seq_len, height, width, 3) and augmented together, so
    a single random hue/saturation/brightness/contrast draw is shared across the whole
    window. Jittering each frame independently would add colour flicker that looks
    like motion and would teach the model to distrust exactly the signal it is here
    to read.
    """
    if seq_len <= 1:
        return augment_img(img)

    shape = tf.shape(img)
    height, width = shape[0], shape[1]
    frames = tf.reshape(img, (height, width, seq_len, 3))
    frames = tf.transpose(frames, [2, 0, 1, 3])
    frames = augment_img(frames)
    frames = tf.transpose(frames, [1, 2, 0, 3])
    return tf.reshape(frames, (height, width, seq_len * 3))


def augment_cmd(cmd):
    """
    Command augmentation

    Args:
      cmd: input command

    Returns:
      cmd: augmented command

    Drawn with a tf op rather than numpy for the same reason as flip_sample: a
    numpy draw inside a tf.data.map is frozen into the graph at trace time.
    """
    coin = tf.random.uniform([])
    is_zero = tf.equal(cmd, 0.0)
    cmd = tf.where(tf.logical_and(is_zero, coin < 0.25), -1.0, cmd)
    cmd = tf.where(
        tf.logical_and(is_zero, tf.logical_and(coin >= 0.25, coin < 0.5)), 1.0, cmd
    )
    return cmd


def flip_sample(img, cmd, label):
    """
    Mirrors an (image, steering, throttle) sample horizontally, per sample.

    label is (steering, throttle) in Ackermann convention, so a horizontal
    flip negates steering and leaves throttle unchanged (matches
    Control.mirror() in the Android robot app) rather than swapping the
    two channels as a tank-drive (left, right) flip would.

    The coin must be drawn with a tf op, not numpy. This runs inside a
    tf.data.map, so a numpy draw is evaluated once while the graph is traced and
    frozen into it as a constant - making the augmentation all-or-nothing across
    the entire dataset rather than random per sample. On a course driven in a
    single direction that is not a subtle problem: the "all" case trains the model
    on a wholly mirrored dataset and it learns to steer the wrong way.
    """
    do_flip = tf.random.uniform([]) < 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    # Only left/right (-1/1) mirror; straight (0) and reverse (2) stay as they are.
    flipped_cmd = tf.where(tf.equal(tf.abs(cmd), 1.0), -cmd, cmd)
    cmd = tf.cond(do_flip, lambda: flipped_cmd, lambda: cmd)
    label = tf.cond(
        do_flip,
        lambda: tf.stack([-label[0], label[1]]),
        lambda: tf.stack([label[0], label[1]]),
    )
    return img, cmd, label


def flip_sample_mem(img, cmd, mem, label):
    """
    flip_sample for a memory model's sample, with its memory
    [s(t-N), t(t-N), ..., s(t-1), t(t-1)] mirrored on the same coin: every steering
    negated, the throttles left as they are, as in the label. Flipping the image but
    not the memory would pair a left turn with a memory of turning right.
    """
    do_flip = tf.random.uniform([]) < 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    flipped_cmd = tf.where(tf.equal(tf.abs(cmd), 1.0), -cmd, cmd)
    cmd = tf.cond(do_flip, lambda: flipped_cmd, lambda: cmd)
    # Even entries are steering, odd ones throttle. A constant, so mem keeps its
    # static shape.
    sign = tf.constant([-1.0, 1.0] * (mem.shape[0] // 2))
    mem = tf.cond(do_flip, lambda: mem * sign, lambda: mem)
    label = tf.cond(
        do_flip,
        lambda: tf.stack([-label[0], label[1]]),
        lambda: tf.stack([label[0], label[1]]),
    )
    return img, cmd, mem, label


# Memory augmentation (--mem_aug): how unreliable a memory model's memory is made in
# training, in normalized units (0.1 ~ 25 raw).
MEM_DROP_PROB = 0.3  # whole memory zeroed, the state the phone starts every run in
MEM_STEER_OFFSET = 0.1  # sd of one offset added to every steering in the memory
MEM_THROTTLE_OFFSET = 0.05  # the same for the throttles
MEM_JITTER = 0.02  # sd of the noise per value on top


def augment_mem(mem):
    """
    Makes a memory model's memory [s(t-N), t(t-N), ..., s(t-1), t(t-1)] less
    trustworthy, so the model has to look at the image. Driving, the memory holds the
    model's own outputs, and their error is a bias rather than noise: a model that
    steers too little remembers steering too little every frame. So:
      - with MEM_DROP_PROB the memory is zeroed, as the phone starts every run, and the
        image is all there is;
      - otherwise one random offset is added to all steerings and one to all
        throttles, plus a little noise per value.
    The label stays the recorded control, so where image and memory disagree the model
    learns to follow the image.
    """
    pairs = mem.shape[0] // 2
    offset = tf.random.normal([2]) * [MEM_STEER_OFFSET, MEM_THROTTLE_OFFSET]
    noisy = mem + tf.tile(offset, [pairs]) + tf.random.normal(mem.shape) * MEM_JITTER
    noisy = tf.clip_by_value(noisy, -1.0, 1.0)
    drop = tf.random.uniform([]) < MEM_DROP_PROB
    return tf.where(drop, tf.zeros_like(mem), noisy)
