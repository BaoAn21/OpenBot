#import "../figures/architectures.typ": cil-arch, rnn-arch

= Driving Policy Models <ch:training>

This chapter describes the two driving policies studied in this thesis and how they are
trained. The first, CIL-Mobile-Fast, is the fastest conditional model provided by OpenBot
and serves as the single-frame baseline. The second is a recurrent model adapted from
DonkeyCar's RNN pilot, which sees a short sequence of frames instead of a single image; it
was extended in this work with a high-level command input and trained with OpenBot's
turn-weighted loss. Both models are trained separately for forward and for backward
driving, on the corresponding parts of the dataset described in @ch:dataset.

== Problem formulation <sec:formulation>
A driving policy $pi$ maps the camera view and a high-level command to a control command:

$ (hat(s)_t, hat(t)_t) = pi(bold(I)_t, c_t), $

where $bold(I)_t$ is the camera image at time $t$, $c_t in {-1, 0, 1, 2}$ is the command
(left, straight, right, reverse), and $hat(s)_t$, $hat(t)_t$ are the predicted steering
and throttle, normalised to $[-1, 1]$ as described in @ch:dataset. The images are RGB
frames of $256 times 96$ pixels, as stored by the data logger. The policy is trained by
behavioural cloning: its parameters are chosen to minimise a loss between its predictions
and the commands of the human driver over the recorded frames.

For the recurrent model, $bold(I)_t$ is replaced by a sequence of $K = 3$ frames,
$(bold(I)_(t-6), bold(I)_(t-3), bold(I)_t)$. Consecutive frames are about 33 ms apart, so
the sequence spans about 200 ms of driving.

== Single-frame model: CIL-Mobile-Fast <sec:cil>
CIL-Mobile-Fast is OpenBot's lightweight variant of conditional imitation learning
@codevilla2018cil, designed to run in real time on a phone @muller2021openbot. Its
architecture is shown in @fig:cil-arch and its layers are listed in @tab:cil-layers. The image passes through five
convolutional layers with 32, 32, 64, 64 and 128 filters, each with a stride of 2 and
followed by batch normalisation and dropout with a rate of 0.2. The resulting
$2 times 7 times 128$ feature map is flattened into 1,792 features and reduced by two
fully connected layers of 512 units.

The command is processed by a small separate network of two fully connected layers with
16 units each. Its output is concatenated with the image features and passed through two
fully connected layers of 64 units. The raw command value is concatenated again before
each of these layers, so that the command can directly influence the decision until the
output. The output layer has two linear units for the steering and the throttle. Because
the output is linear rather than bounded, the throttle can take negative values, which
allows the same architecture to be used as the backward model.

The architecture is used as provided by OpenBot. The only difference is the meaning of the
two outputs: in OpenBot they are the left and right wheel speeds of a differential-drive
robot, while here they are the steering and the throttle (@sec:app-control). The model has
1,324,564 parameters.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, right),
    table.header([*Layer*], [*Output shape*], [*Parameters*]),
    [Image input], [$96 times 256 times 3$], [--],
    [Conv $5 times 5$, 32, stride 2], [$46 times 126 times 32$], [2,432],
    [Conv $3 times 3$, 32, stride 2], [$22 times 62 times 32$], [9,248],
    [Conv $3 times 3$, 64, stride 2], [$10 times 30 times 64$], [18,496],
    [Conv $3 times 3$, 64, stride 2], [$4 times 14 times 64$], [36,928],
    [Conv $2 times 2$, 128, stride 2], [$2 times 7 times 128$], [32,896],
    [Flatten, FC 512, FC 512], [512], [1,180,672],
    [Command MLP (16, 16)], [16], [304],
    [Concatenate, FC 64], [64], [33,856],
    [Concatenate command, FC 64], [64], [4,224],
    [Concatenate command, FC 2 (linear)], [2], [132],
  ),
  caption: [Layers of CIL-Mobile-Fast. Every convolutional and 512-unit layer is followed
    by batch normalisation and dropout; the batch normalisation parameters are included
    in the totals.],
) <tab:cil-layers>

#figure(
  cil-arch,
  caption: [Architecture of CIL-Mobile-Fast. The command network's output is concatenated
    with the image features, and the raw command $c$ is concatenated with the input of each
    later layer.],
  placement: auto,
) <fig:cil-arch>

== Recurrent model <sec:rnn>
A single image shows where the car is relative to the track, but not how it is moving: it
does not show the car's speed, or whether it is already turning. The recurrent model gives
the policy this information by processing a short sequence of frames. It is based on the
RNN pilot of DonkeyCar @donkeycar, which applies the same small convolutional network to
every frame and combines the per-frame features with recurrent layers.

Its architecture is shown in @fig:rnn-arch and its layers are listed in @tab:rnn-layers. Each of the three frames
passes through the same convolutional network, applied to each frame with shared
weights: four convolutional layers with 24, 32, 32 and 32 filters, each followed by
dropout with a rate of 0.3, a $2 times 2$ max-pooling layer, and a fully connected layer
that reduces each frame to 100 features. The three feature vectors are then processed in
order, oldest first, by two LSTM layers @hochreiter1997lstm of 128 units, each followed by
dropout with a rate of 0.1. Because the motion is read from features of whole frames, the
model can relate the position of the track in one frame to its position in the next.

This work makes three changes to DonkeyCar's RNN pilot:

+ *Command input.* DonkeyCar's pilot has no command input and cannot be told which way to
  go at a junction. A command branch identical to that of CIL-Mobile-Fast was added: a
  16-16 network whose output is concatenated with the LSTM output, and the raw command
  value concatenated again before each of the following fully connected layers of 128, 64
  and 10 units and the linear output layer.
+ *Frame spacing.* Instead of consecutive frames, the sequence uses every third frame, so
  that three frames cover about 200 ms. Over a span of only 66 ms the car moves too little
  for the difference between frames to be informative.
+ *Loss function.* DonkeyCar trains its pilots with the mean squared error. The model in
  this work is trained with OpenBot's turn-weighted loss (@sec:loss), so that the rare
  frames in turns are not outweighed by the many frames on straight sections.

The model has 495,552 parameters, fewer than CIL-Mobile-Fast even though it processes
three frames, because each frame is reduced to only 100 features before the recurrent
layers. It does not use batch normalisation. The three frames are given to the model as one
$96 times 256 times 9$ tensor, with the frames stacked along the colour channels, and are
separated inside the model, so that the same input format is used in training and on the
phone.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, right),
    table.header([*Layer*], [*Output shape*], [*Parameters*]),
    [Image input (3 stacked frames)], [$96 times 256 times 9$], [--],
    [Separate into frames], [$3 times 96 times 256 times 3$], [--],
    [Conv $5 times 5$, 24, stride 2 #super[a]], [$3 times 46 times 126 times 24$], [1,824],
    [Conv $5 times 5$, 32, stride 2 #super[a]], [$3 times 21 times 61 times 32$], [19,232],
    [Conv $3 times 3$, 32, stride 2 #super[a]], [$3 times 10 times 30 times 32$], [9,248],
    [Conv $3 times 3$, 32, stride 1 #super[a]], [$3 times 8 times 28 times 32$], [9,248],
    [Max pool $2 times 2$, flatten #super[a]], [$3 times 1792$], [--],
    [FC 100 #super[a]], [$3 times 100$], [179,300],
    [LSTM 128 (sequence)], [$3 times 128$], [117,248],
    [LSTM 128 (last step)], [128], [131,584],
    [Command MLP (16, 16)], [16], [304],
    [Concatenate, FC 128], [128], [18,560],
    [Concatenate command, FC 64], [64], [8,320],
    [Concatenate command, FC 10], [10], [660],
    [Concatenate command, FC 2 (linear)], [2], [24],
  ),
  caption: [Layers of the recurrent model. #super[a] Applied to each of the three frames
    with shared weights.],
) <tab:rnn-layers>

#figure(
  rnn-arch,
  caption: [Architecture of the recurrent model. The same convolutional network (dashed box)
    processes each of the three frames, two LSTM layers combine the per-frame features in
    time order, and the command enters as in CIL-Mobile-Fast.],
  placement: auto,
) <fig:rnn-arch>

@tab:model-compare summarises the differences between the two models.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: left,
    table.header([], [*CIL-Mobile-Fast*], [*Recurrent model*]),
    [Origin], [OpenBot, used unchanged], [DonkeyCar's RNN pilot, extended in this work],
    [Image input], [1 frame], [3 frames, about 200 ms],
    [Temporal processing], [None], [Two LSTM layers],
    [Command input], [16-16 MLP, raw command before later layers], [Same as CIL-Mobile-Fast],
    [Normalisation], [Batch normalisation], [None],
    [Loss], [Turn-weighted (@sec:loss)], [Turn-weighted (@sec:loss)],
    [Parameters], [1,324,564], [495,552],
  ),
  caption: [Comparison of the two driving models.],
) <tab:model-compare>

== Loss function <sec:loss>
On the track used in this work, as on most tracks, the car drives straight far more often
than it turns. Under a plain mean squared error, the frames in turns contribute little to
the total loss, and a model can achieve a low loss while steering too little in turns.
OpenBot addresses this with a loss that weights each sample by the sharpness of its turn,
and the same loss is used for both models in this work.

The loss is defined on OpenBot's left and right wheel speeds. The steering and throttle
labels are therefore first converted to that form, $l = t + s$ and $r = t - s$, and the
turn is measured by the angle $a = r - l$. For a batch of $B$ samples, the loss is

$ cal(L) = 1 / B sum_(i=1)^B w_i^2 [1 / B sum_(j=1)^B (a_j - hat(a)_j)^2 + 1 / 2 ((l_i - hat(l)_i)^2 + (r_i - hat(r)_i)^2)], $ <eq:loss>

with the per-sample weight

$ w_i = |a_i + 0.05|, $ <eq:loss-weight>

where hats denote predictions. The first term inside the brackets is the mean squared
error of the angle and the second is the mean squared error of the two wheel speeds. In
terms of the steering and throttle errors $Delta s$ and $Delta t$, the angle error is
$-2 Delta s$, and the second term equals $Delta s^2 + Delta t^2$, so steering errors
are weighted more heavily than throttle errors.

The weight $w_i = |0.05 - 2 s_i|$ grows with the magnitude of the target steering. A
sample at full steering lock is weighted about 1,600 times more than a sample driving
straight, and the constant 0.05 keeps the weight of straight samples above zero. The loss
is implemented exactly as in OpenBot, so that the results remain comparable with it. This
includes two properties of the original implementation: the weight is slightly asymmetric
between left and right turns, and the angle term is averaged over the whole batch, so that
only the wheel-speed term is weighted per sample.

== Training <sec:training-setup>
Both models are trained with the Adam optimiser @kingma2015adam, a learning rate of
$3 times 10^(-4)$ and a batch size of 128, for at most 100 epochs. The test recordings of
the selected sessions (@ch:dataset) serve as the validation set. Training stops early when
the validation loss has not improved for 20 epochs, and the checkpoint with the lowest
validation loss is kept for deployment. Training was performed on an NVIDIA RTX 3050 GPU.

The training images are augmented to make the models less sensitive to lighting:

- *Colour jitter.* Random changes of hue (up to $plus.minus 0.08$), saturation (factor 0.6
  to 1.6), brightness (up to $plus.minus 0.05$) and contrast (factor 0.7 to 1.3). For the
  recurrent model, one random change is applied to all three frames of a sequence;
  changing each frame independently would introduce differences between the frames that
  resemble motion.
- *Horizontal flip.* With a probability of 0.5, the image is mirrored, the steering label
  is negated and a left command is exchanged for a right command and vice versa. This
  doubles the variety of turns and balances left and right steering. Both models are
  trained with this augmentation; for the recurrent model, all three frames of a sequence
  are mirrored together.

== Deployment on the phone <sec:deployment>
The trained models are converted to TensorFlow Lite and run on the phone's CPU. For the
recurrent model, the conversion required special care. TensorFlow Lite's LSTM operator
keeps its internal state from one call to the next, whereas the model is trained to start
from a zero state for every sequence of three frames. Converted directly, the model on the
phone therefore carried its state from frame to frame and its predictions matched the
trained model only on the first call. The model is therefore exported with its LSTM layers
unrolled into ordinary operations over the three time steps, with the same weights and no
state kept between calls. Every exported model is checked automatically against the trained
model on real test frames, by feeding them one after another as the phone does and
verifying that the outputs agree and that nothing carries over between calls.

On the phone, every camera frame is stored in a short frame history, independently of how
long inference takes. When the model runs, the frames whose ages are closest to those used
in training (0, about 100 and about 200 ms) are taken from this history, so that the model
sees the same temporal spacing as in training even when the inference rate is lower than
the camera's frame rate.
