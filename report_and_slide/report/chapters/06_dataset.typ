= Dataset <ch:dataset>

All data were recorded on an indoor track marked with blue tape (@fig:track). Recordings
were made at different times of day, in the morning, in the afternoon under dim daylight
and in the evening under room lighting, so that the dataset covers a range of lighting
conditions.

#figure(
  grid(
    columns: 4,
    gutter: 6pt,
    image("../figures/track_01.jpg"),
    image("../figures/track_03.jpg"),
    image("../figures/track_04.jpg"),
    image("../figures/track_06.jpg"),
  ),
  caption: [Sections of the indoor track used for data collection and evaluation.],
) <fig:track>

== Driving task
The car is trained to perform the manoeuvres shown in @fig:track-paths. When driving
forward, it follows the corridor between the wall and the central block in either
direction, or turns right into one of the two side branches. When driving backward, it
reverses out of a side branch into the corridor. The manoeuvre at a junction is selected
by the high-level command given to the model: 0 to continue straight, 1 to turn right, and
2 while reversing. Forward and backward driving are handled by two separate models.

#figure(
  placement: auto,
  grid(
    columns: (32%, 32%),
    gutter: 16pt,
    image("../figures/forward_diagram.png"),
    image("../figures/backward_diagram.png"),
    [(a) Forward manoeuvres], [(b) Backward manoeuvres],
  ),
  caption: [Paths driven on the track. Black lines mark the track boundaries and the
    rectangle is the central block; arrows show the direction of travel.],
) <fig:track-paths>

TODO: track dimensions (lane width, total length).

== Data collection
TODO: recording with the phone's data logger while driving with the gamepad; the
indicator buttons set the high-level command (straight, right turn, reverse); recording
sessions and lighting conditions.

== Data processing
During a recording, the phone stores every camera frame with its timestamp and logs the
steering and throttle commands and the indicator state separately, each with its own
timestamp. After recording, each frame is matched to the control command and indicator
state closest to it in time, which gives one labelled sample per frame
(@fig:labels-raw). Control values are stored in the device units of the serial protocol,
in the range $[-255, 255]$. The column names _left_ and _right_ are inherited from the
original OpenBot pipeline, which was designed for differential-drive robots; in this work
they hold the steering and the throttle respectively.

#figure(
  image("../figures/dataset_processed.png", width: 85%),
  caption: [Excerpt of a processed label file: timestamp in nanoseconds, cropped camera
    frame, steering (_left_), throttle (_right_) and high-level command.],
) <fig:labels-raw>

Each recording is then split by the sign of the throttle. Contiguous runs with positive
throttle form the forward data, and all frames with negative throttle form the backward
data, whose command is set to 2. Frames with zero throttle, which occur while the ESC
switches between forward and reverse, are discarded. A forward run is also split wherever
two consecutive frames are more than 0.5 s apart. The recordings are grouped into numbered
sessions, and 20% of the recordings in each session are held out as test data. The split
is made per recording rather than per frame, so that no test frame has an almost identical
neighbouring frame in the training set. Finally, steering and throttle are divided by 255,
which gives training targets in $[-1, 1]$ (@fig:labels-processed).

#figure(
  image("../figures/dataset_processed_with_images.png", width: 100%),
  caption: [Training samples after processing: each camera frame with its time,
    steering, throttle and command in device units, and the normalised targets.],
) <fig:labels-processed>
