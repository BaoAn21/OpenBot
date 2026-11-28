= Software Applications <ch:app>

This chapter describes the two applications developed in this work: the Android
application that runs on the phone mounted on the car, and a web application on the
training computer that manages the recorded data, the training runs and the deployment of
models to the phone.

== Android application

=== Base: the OpenBot robot app
The Android application is a fork of the OpenBot robot app @muller2021openbot. OpenBot
already provides most of what a smartphone-driven robot needs: access to the camera,
on-device inference with TensorFlow Lite, a data logger that records camera frames together
with the control commands and the phone's sensors, support for Bluetooth gamepads, and
communication with the microcontroller over USB serial. These parts were reused with minor
changes, which allowed the work to concentrate on what this vehicle and task require.
@tab:app-changes summarises which components were reused and which were added or modified,
and @fig:app-screens (a) shows the main screen of the application.

#figure(
  placement: auto,
  table(
    columns: (auto, 1fr),
    align: left,
    table.header([*Component*], [*Status in this work*]),
    [Camera pipeline, TensorFlow Lite inference], [Reused from OpenBot],
    [Data logger (frames, controls, phone sensors)], [Reused; the indicator log is used as
      the high-level command],
    [Gamepad support], [Reused; indicator buttons toggle like a car's turn-signal stalk,
      with a third button for reverse],
    [Upload of recordings and download of models], [Reused; served by the web application
      of @sec:webapp],
    [Vehicle control and serial protocol], [Modified: steering and throttle instead of
      left and right wheel speeds (@sec:app-control)],
    [Indicator states], [Modified: a reverse state added to left, off and right],
    [Autopilot Session screen], [Added (@sec:app-session)],
    [Autopilot Memory and Autopilot Seq screens], [Added, for models that take their own
      previous outputs or a short sequence of frames as input],
  ),
  caption: [Components of the Android application reused from OpenBot and those added or
    modified in this work.],
) <tab:app-changes>

=== Ackermann control and serial protocol <sec:app-control>
OpenBot was designed for differential-drive robots, which steer by driving their left and
right wheels at different speeds; its control commands are therefore pairs of left and
right wheel speeds. The car used in this work has Ackermann steering, with a steering servo
and a single drive motor (@ch:hardware), so the application's control command was changed
to a pair of steering and throttle values. Both are expressed in device units in the range
$[-255, 255]$: positive steering turns right and positive throttle drives forward. They are
sent to the microcontroller as the message `c<steering>,<throttle>` (@tab:protocol).

Several parts reused from OpenBot, including its model interface and its phone and web
controllers, still produce a normalised left/right pair $(l, r) in [-1, 1]^2$. These are
converted at a single point in the code,

$ s = (l - r) / 2 dot 255, quad t = (l + r) / 2 dot 255, $

where $s$ is the steering and $t$ the throttle, so the rest of the reused code did not need
to change.

The indicator state, which in OpenBot only drives the turn-signal LEDs, has four values in
this work: $-1$ (left), $0$ (off), $1$ (right) and $2$ (reverse). It is sent to the
microcontroller, whose display shows the turn signals and the reverse light, and it is also
the high-level command given to the driving models (@ch:training).

=== Autopilot Session screen <sec:app-session>
The Autopilot Session screen (@fig:app-screens (b)) runs a complete autonomous run on a
track that is driven in both directions. Two models are selected on it, one for driving
forward and one for driving backward. Both are loaded when the screen opens, so that
switching between them is immediate, but only the active one is evaluated on each camera
frame. The active model is chosen by the indicator: in the reverse state the backward model
drives, otherwise the forward model. The indicator can be changed with the on-screen
direction button, with the gamepad or from a remote controller, and its value is also the
command input of the active model, exactly as it was logged during data collection.

On every change of direction the car stops for 0.5 s before the other model takes over.
The throttle is then constrained to be non-negative when driving forward and non-positive
when driving backward. This is necessary because some models have a linear throttle
output, and with the ESC in forward/reverse mode a small throttle of the wrong sign would
reverse the car in the middle of a run. The direction is therefore decided by the
indicator, and the model only decides the speed.

Three throttle modes can be selected. In the _model_ mode, the model controls both the
steering and the throttle. In the _fixed_ mode, every model drives at the same constant
speed, adjustable from the gamepad, so that runs with different models compare their
steering alone. In the _manual_ mode, the person holding the gamepad controls the throttle
while the model steers. In every mode, moving the gamepad's steering stick while the
autopilot is active hands control to the person; the models resume driving 0.3 s after the
stick has returned to the centre.

When logging is enabled, the run is recorded in the same format as the data logger's
recordings, so that the training pipeline can read it. In addition, a log is written with
one row per processed frame, stating who drove it (the model, the person during a
takeover, or the pause after a change of direction) together with the model's prediction.
These logs are used in the evaluation of @ch:results, and the segments driven by the
person can be extracted as new training data.

#figure(
  placement: auto,
  grid(
    columns: (32%, 32%),
    gutter: 24pt,
    image("../figures/OpenBot_mainscreen.jpg"),
    image("../figures/autopilot_session_screen.jpg"),
    [(a) Main screen], [(b) Autopilot Session],
  ),
  caption: [The Android application. (a) The main screen, with the screens added in this
    work (Autopilot Seq, Autopilot Memory and Autopilot Session) next to those reused from
    OpenBot. (b) The Autopilot Session screen with a forward and a backward model selected;
    the label at the top shows the current direction and command.],
) <fig:app-screens>

== Web application for data and training <sec:webapp>
Training a driving model involves many repeated steps: copying recordings from the phone,
splitting them into forward and backward data, building the training records, running the
training, inspecting the results and copying the selected model back to the phone.
Performing these steps by hand from the command line was slow and error-prone, so a web
application was developed on the training computer to perform them from a browser. It does
not implement any new processing of its own: each action runs the same command-line
scripts that are described in @ch:dataset and @ch:training, as a separate background process, so that a
training run continues even if the web server is restarted.

The application is organised into the following pages:

- *Inbox.* Recordings uploaded by the phone arrive here. Each recording can be played back
  frame by frame together with its steering, throttle and command (@fig:web-visualize),
  which makes it possible to discard faulty recordings before they are used for training.
  From this page the recordings are split into a new session.
- *Sessions.* Lists every session with its recordings, their assignment to the training or
  test set and their forward and backward parts (@fig:web-sessions).
- *Train.* Selects the direction, the sessions, the model and its options, and shows the
  exact command that will be run. Training runs are queued and executed one at a time on
  the GPU, with the loss curves updated live.
- *Models.* Shows the metrics, loss curves and evaluation plots of each trained model, and
  publishes a selected checkpoint to the phone, from which the Android application
  downloads it (@fig:web-models).
- *Autopilot runs.* Receives the logs of the Autopilot Session screen, reports the number
  of takeovers per minute for each model, and extracts the person's takeovers as new
  recordings. Each run can be played back with the model's predictions next to the
  commands that were sent to the car, with the takeovers marked (@fig:web-takeover).

#figure(
  image("../figures/data_visualize_web.png", width: 95%),
  caption: [Playback of a recording in the web application: the camera frame, the steering
    and throttle at that moment, the command, and the steering and throttle over the whole
    recording.],
  placement: auto,
) <fig:web-visualize>

#figure(
  image("../figures/data_train_test_web.png", width: 95%),
  caption: [A session in the web application: each recording with its assignment to the
    training or test set and the forward runs it contains.],
  placement: auto,
) <fig:web-sessions>

#figure(
  image("../figures/training_model_web.png", width: 95%),
  caption: [A trained model in the web application: publishing it to the phone, its
    metrics, its training curves and its evaluation on the test set.],
  placement: auto,
) <fig:web-models>

#figure(
  image("../figures/takeover_screen.png", width: 95%),
  caption: [Playback of an autopilot run in the web application. Solid lines are the
    steering and throttle sent to the car and dashed lines the model's predictions; the
    shaded interval is a takeover, in which the person drove.],
  placement: auto,
) <fig:web-takeover>
