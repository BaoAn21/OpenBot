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


def augment_cmd(cmd):
    """
    Command augmentation

    Args:
      cmd: input command

    Returns:
      cmd: augmented command
    """
    if not (cmd > 0 or cmd < 0):
        coin = np.random.default_rng().uniform(low=0.0, high=1.0, size=None)
        if coin < 0.25:
            cmd = -1.0
        elif coin < 0.5:
            cmd = 1.0
    return cmd


def flip_sample(img, cmd, label):
    """
    Mirrors an (image, steering, throttle) sample horizontally.

    label is (steering, throttle) in Ackermann convention, so a horizontal
    flip negates steering and leaves throttle unchanged (matches
    Control.mirror() in the Android robot app) rather than swapping the
    two channels as a tank-drive (left, right) flip would.
    """
    coin = np.random.default_rng().uniform(low=0.0, high=1.0, size=None)
    if coin < 0.5:
        img = tf.image.flip_left_right(img)
        cmd = -cmd
        label = tf.stack([-label[0], label[1]])
    return img, cmd, label
