= Background and Related Work <ch:background>

This chapter reviews the two open-source projects this thesis builds on, OpenBot and
DonkeyCar, and the learning methods behind their driving policies. It ends by placing
this work relative to both.

== Smartphones as robot brains

=== OpenBot <sec:openbot>
OpenBot @muller2021openbot proposes the smartphone as the brain of a small robot. A
modern phone already has a good camera, an IMU, GPS, wireless connectivity and a
processor that can run neural networks in real time, so the robot body only has to
provide wheels, motors and a microcontroller. The reference body is 3D-printed, costs
about \$50, and drives four wheels as a differential-drive vehicle: the left and right
wheels are set independently, and the robot turns by driving them at different speeds.
An Arduino Nano on the body drives the motors, reads the wheel encoders, sonar and
battery voltage, and talks to the phone over a USB serial link.

The software stack has three parts:
- An *Android app* that connects to the body, shows the camera feed, records driving
  data while a person drives with a game controller, and runs trained models with
  TensorFlow Lite. Besides autonomous driving it supports person following with an
  object detector.
- A *training pipeline* in Python and TensorFlow that turns the recorded sessions into
  training data, trains a driving policy and exports it to TensorFlow Lite.
- *Firmware* for the microcontroller, with a simple text protocol to the phone.

The driving policy is learned by imitation: it maps a camera image and a high-level
command (turn left, go straight, turn right) to the two wheel speeds, in the style of
conditional imitation learning (@sec:e2e). The paper shows that such a policy, trained on
data from a few hours of driving, runs on a range of phones and navigates indoor
environments.

This thesis reuses most of that stack: the app, the data format, the training pipeline,
its `cil_mobile_fast` network and the idea of its turn-weighted loss. OpenBot's design
does not, however, fit a car directly. Its control is differential (left/right), which
an Ackermann-steered car with a steering servo and a single drive motor cannot execute,
and its autopilot only drives forward.

== Small-scale self-driving cars

=== DonkeyCar <sec:donkeycar>
DonkeyCar @donkeycar is an open-source Python platform for self-driving RC cars,
usually at 1:16 or 1:10 scale. A Raspberry Pi or an NVIDIA Jetson Nano with a
wide-angle camera sits on an off-the-shelf RC car and drives its steering servo and
electronic speed controller through PWM, so unlike OpenBot it controls a real
Ackermann-steered car. A person drives the car around a track from a web page or a
gamepad while images and controls are recorded; a network is then trained on a
computer and copied back to the car to drive it. The project grew out of the DIY Robocars
races and is widely used for teaching and small-car racing.

DonkeyCar offers several Keras network types on the same data: the basic *linear*
model regresses steering and throttle from one image, the *categorical* model predicts
them as bins, and others add sequences of frames (recurrent or 3D-convolutional
models), an IMU, or a behaviour input. Its *memory* model is the one most relevant here:
besides the current image, it is given the last three (steering, throttle) pairs. These
are the recorded controls during training and the model's own previous outputs while
driving, which gives the policy a short memory of what it was doing. The memory models
in this thesis start from this network (`donkey_memory`). In my previous internship I
also used DonkeyCar's colour-based line-tracking method as a classical baseline.

Other small-car platforms take different routes: F1TENTH @okelly2020f1tenth uses 1:10
cars with lidar and embedded GPUs for racing research, and AWS DeepRacer
@balaji2020deepracer trains policies with reinforcement learning in simulation and
transfers them to a 1:18 car. Both rely on dedicated onboard computers rather than a
phone.

== End-to-end driving <sec:e2e>
End-to-end driving learns the mapping from sensor input to control commands directly,
instead of chaining hand-designed perception, planning and control stages. ALVINN
@pomerleau1989alvinn already steered a vehicle with a small neural network in 1989.
NVIDIA's PilotNet @bojarski2016pilotnet revived the idea with a convolutional network
that maps a front camera image to a steering angle, trained on about 72 hours of human
driving. Both OpenBot's and DonkeyCar's networks follow this pattern: a stack of
convolutional layers followed by fully connected layers that output the controls.

This form of imitation learning, also called behavioural cloning, is supervised
regression on recorded (image, control) pairs. Its main weakness is distribution shift
@ross2011dagger: when driving, the policy sees the consequences of its own small
errors, states the human driver rarely produced, and the errors can compound. Offline
metrics on recorded data therefore do not fully predict how well a policy drives, and
policies have to be evaluated closed loop, on the vehicle.

*Conditional imitation learning.* Image-to-control alone cannot decide which way to go
at a junction. Codevilla et al. @codevilla2018cil add a high-level command as an extra
input and train the network to follow it; in their architecture each command selects
its own output branch. OpenBot uses a lighter form: the command passes through a small
network that is fused with the image features, and is fed again into the final layers.
In this thesis the command is the car's indicator state (left, right or reverse), set by
the driver during recording and by the user at run time.

*Temporal context.* A single image does not show how fast the car is moving or which
way it is already turning. Policies can be given that context by stacking several
frames, by recurrent layers, or, as in DonkeyCar's memory model, by feeding back their
own recent outputs. More history is not automatically better: a policy given its past
actions can learn to copy them instead of reading the image, the "copycat" problem
@wen2020copycat, and behaviour cloning in general tends to pick up spurious
correlations, such as staying still because the car was already still
@codevilla2019limitations. The same problem appears in this thesis, where the memory
model relies on its memory instead of the indicator (@ch:results).

== Comparison and positioning of this work
@tab:related compares the two platforms with the system built in this thesis.

#figure(
  {
  set par(justify: false)
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    table.header([], [*OpenBot*], [*DonkeyCar*], [*This work*]),
    [Onboard computer], [Smartphone], [Raspberry Pi / Jetson Nano], [Smartphone],
    [Vehicle], [3D-printed, differential drive], [RC car, Ackermann steering],
      [RC car, Ackermann steering],
    [Low-level controller], [Arduino Nano], [PWM driver board], [ESP32-S3],
    [Control output], [Left/right wheel speeds], [Steering, throttle],
      [Steering, throttle],
    [Command input], [Left, straight, right], [—], [Indicator: left, right, reverse],
    [Memory of past controls], [—], [Memory model], [Memory models],
    [Learned reverse driving], [—], [—], [Separate backward model],
  )
  },
  caption: [OpenBot, DonkeyCar and this work.],
) <tab:related>

OpenBot brings the phone as the onboard computer and a complete app and training
pipeline, but drives a differential-drive body. DonkeyCar drives real RC cars and
offers a memory model, but needs a dedicated onboard computer and has no command input.
This thesis combines the two: it moves OpenBot onto an Ackermann RC car, with steering
and throttle throughout the app, the serial protocol and new firmware; it brings
DonkeyCar's memory model into OpenBot's pipeline and app and conditions it on the
indicator; and it adds learned backward driving with a separate model.
