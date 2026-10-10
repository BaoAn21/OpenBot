# Created by Matthias Mueller - Intel Intelligent Systems Lab - 2020

import tensorflow as tf

"""
Constructors for standard MLPs and CNNs
"""


def create_cnn(
    width,
    height,
    depth,
    cnn_filters=(8, 12, 16, 20, 24),
    kernel_sz=(5, 5, 5, 3, 3),
    stride=(2, 2, 2, 2, 2),
    padding="same",
    activation="relu",
    conv_dropout=0,
    mlp_filters=(64, 16),
    mlp_dropout=0.2,
    bn=False,
):
    # define input shape, channel dimension (tf convention: channels last) and img input
    inputShape = (height, width, depth)
    channelDim = -1
    inputs = tf.keras.Input(shape=inputShape, name="img_input")

    # build the cnn layer by layer
    for i, f in enumerate(cnn_filters):
        # set the input if it is the first layer
        if i == 0:
            x = inputs

        # build one block with conv, activation and optional bn and dropout
        x = tf.keras.layers.Conv2D(
            f,
            (kernel_sz[i], kernel_sz[i]),
            strides=(stride[i], stride[i]),
            padding=padding,
            activation=activation,
        )(x)
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=channelDim)(x)
        if conv_dropout > 0:
            x = tf.keras.layers.Dropout(conv_dropout)(x)

    # flatten output of the cnn and build the mlp
    x = tf.keras.layers.Flatten()(x)
    # build the mlp layer by layer
    for i, f in enumerate(mlp_filters):
        x = tf.keras.layers.Dense(f, activation=activation)(x)
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=channelDim)(x)
        if mlp_dropout > 0:
            x = tf.keras.layers.Dropout(mlp_dropout)(x)

    # assemble the model
    model = tf.keras.Model(inputs, x)

    # return the model
    return model


def create_mlp(in_dim, hidden_dim, out_dim, activation="relu", dropout=0.2, name="cmd"):
    model = tf.keras.Sequential(name="MLP")
    model.add(
        tf.keras.layers.Dense(
            hidden_dim, input_dim=in_dim, activation=activation, name=name
        )
    )
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(out_dim, activation=activation))
    return model


def pilot_net(img_width, img_height, bn=False, policy="autopilot"):
    if policy == "autopilot":
        mlp = create_mlp(1, 1, 1, dropout=0, name="cmd")
    elif policy == "point_goal_nav":
        mlp = create_mlp(3, 16, 16, dropout=0, name="goal")
    else:
        raise Exception("Unknown policy")

    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(24, 36, 48, 64, 64),
        kernel_sz=(5, 5, 5, 3, 3),
        stride=(2, 2, 2, 1, 1),
        padding="valid",
        activation="relu",
        mlp_filters=(1164, 100),
        mlp_dropout=0,
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(50, activation="relu")(combinedInput)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(10, activation="relu")(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(name="pilot_net", inputs=(cnn.input, mlp.input), outputs=x)
    return model


def _pilot_net_head(cnn, mlp, name):
    """The pilot_net output stack, shared by the single-frame and sequence variants."""
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    x = tf.keras.layers.Dense(50, activation="relu")(combinedInput)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(10, activation="relu")(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    return tf.keras.Model(name=name, inputs=(cnn.input, mlp.input), outputs=x)


def _autopilot_mlp(policy):
    if policy == "autopilot":
        return create_mlp(1, 1, 1, dropout=0, name="cmd")
    elif policy == "point_goal_nav":
        return create_mlp(3, 16, 16, dropout=0, name="goal")
    raise Exception("Unknown policy")


def pilot_net_seq(img_width, img_height, bn=False, policy="autopilot", seq_len=5):
    """pilot_net over a stack of seq_len frames fused at the input (early fusion).

    The only structural difference from pilot_net is the input depth: the frames are
    concatenated along the channel axis, so the first convolution sees all of them at
    once and can pick up apparent motion. Everything after that is unchanged, which
    keeps the op set (and the inference cost) essentially identical to the
    single-frame model - only the first conv kernel grows.
    """
    cnn = create_cnn(
        img_width,
        img_height,
        3 * seq_len,
        cnn_filters=(24, 36, 48, 64, 64),
        kernel_sz=(5, 5, 5, 3, 3),
        stride=(2, 2, 2, 1, 1),
        padding="valid",
        activation="relu",
        mlp_filters=(1164, 100),
        mlp_dropout=0,
        bn=bn,
    )
    return _pilot_net_head(cnn, _autopilot_mlp(policy), "pilot_net_seq")


def cil_mobile(img_width, img_height, bn=True, policy="autopilot"):
    if policy == "autopilot":
        mlp = create_mlp(1, 16, 16, dropout=0.5, name="cmd")
    elif policy == "point_goal_nav":
        mlp = create_mlp(3, 16, 16, dropout=0.5, name="goal")
    else:
        raise Exception("Unknown policy")

    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 64, 96, 128, 256),
        kernel_sz=(5, 3, 3, 3, 3),
        stride=(2, 2, 2, 2, 2),
        padding="same",
        activation="relu",
        conv_dropout=0.2,
        mlp_filters=(128, 64),
        mlp_dropout=0.5,
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(64, activation="relu")(combinedInput)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(name="cil_mobile", inputs=(cnn.input, mlp.input), outputs=x)

    return model


def cil_mobile_fast(img_width, img_height, bn=True, policy="autopilot"):
    if policy == "autopilot":
        mlp = create_mlp(1, 16, 16, name="cmd")
    elif policy == "point_goal_nav":
        mlp = create_mlp(3, 16, 16, name="goal")
    else:
        raise Exception("Unknown policy")

    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 32, 64, 64, 128),
        kernel_sz=(5, 3, 3, 3, 2),
        stride=(2, 2, 2, 2, 2),
        padding="valid",
        activation="relu",
        conv_dropout=0.2,
        mlp_filters=(512, 512),
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(64, activation="relu")(combinedInput)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(
        name="cil_mobile_fast", inputs=(cnn.input, mlp.input), outputs=x
    )

    return model


def cil(img_width, img_height, bn=True, policy="autopilot"):
    if policy == "autopilot":
        mlp = create_mlp(1, 64, 64, dropout=0.5, name="cmd")
    elif policy == "point_goal_nav":
        mlp = create_mlp(3, 64, 64, dropout=0.5, name="goal")
    else:
        raise Exception("Unknown policy")

    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 32, 64, 64, 128, 128, 256, 256),
        kernel_sz=(5, 3, 3, 3, 3, 3, 3, 3),
        stride=(2, 1, 2, 1, 2, 1, 1, 1),
        padding="valid",
        activation="relu",
        conv_dropout=0.2,
        mlp_filters=(512, 512),
        mlp_dropout=0.5,
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(256, activation="relu")(combinedInput)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(name="cil", inputs=(cnn.input, mlp.input), outputs=x)

    return model


def donkey_memory(img_width, img_height, bn=False, policy="autopilot", mem_len=3):
    """donkeycar's "memory" pilot (default_memory in donkeycar/parts/keras.py), as is.

    The current image plus the last mem_len (steering, throttle) pairs: recorded ones
    in training, the model's own predictions when driving. Kept faithful on purpose,
    so it is not pilot_net with a memory input bolted on: donkeycar's CNN with dropout
    and no batch norm (bn is ignored), no cmd input, and tanh steering / sigmoid
    throttle heads - which means it cannot predict reverse. The two heads are joined
    into the single (steering, throttle) output every other model here has.
    """
    if policy != "autopilot":
        raise Exception("donkey_memory only supports the autopilot policy")

    drop = 0.2
    drop2 = 0.1

    img_in = tf.keras.Input(shape=(img_height, img_width, 3), name="img_input")
    x = img_in
    # core_cnn_layers: valid padding, dropout after every convolution.
    for filters, kernel, stride in (
        (24, 5, 2),
        (32, 5, 2),
        (64, 5, 2),
        (64, 3, 1),
        (64, 3, 1),
    ):
        x = tf.keras.layers.Conv2D(
            filters, (kernel, kernel), strides=(stride, stride), activation="relu"
        )(x)
        x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Flatten()(x)

    # The memory branch narrows from 2 * mem_len down to 2 (mem_depth = 0, the default).
    mem_in = tf.keras.Input(shape=(2 * mem_len,), name="mem_input")
    y = mem_in
    for i in range(1, mem_len):
        y = tf.keras.layers.Dense(2 * (mem_len - i), activation="relu")(y)
        y = tf.keras.layers.Dropout(drop2)(y)

    x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.Dropout(drop)(x)

    steering = tf.keras.layers.Dense(1, activation="tanh", name="steering")(x)
    throttle = tf.keras.layers.Dense(1, activation="sigmoid", name="throttle")(x)
    x = tf.keras.layers.concatenate([steering, throttle])

    return tf.keras.Model(name="donkey_memory", inputs=(img_in, mem_in), outputs=x)
