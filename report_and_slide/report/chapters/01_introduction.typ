= Introduction <ch:intro>

== Motivation
Autonomous driving has advanced rapidly over the past decade, yet the vehicles used in
both industry and academic research remain expensive to build and operate. A typical
research vehicle perceives its surroundings through a suite of complementary sensors,
including several LiDAR units, radars, surround-view cameras and a GNSS/IMU unit, and is
often supported by high-definition maps @yurtsever2020survey @sun2020waymo. Processing
these data streams in real time further requires high-performance onboard computers with
substantial power budgets @lin2018architectural. The resulting cost in hardware, energy
and integration effort places full-scale autonomous driving research beyond the reach of
most university laboratories and individual students. Even small-scale research
platforms such as F1TENTH @okelly2020f1tenth rely on LiDAR and embedded GPUs.

The camera, by contrast, is the least expensive automotive sensor while carrying the
richest information about the scene, and it is the primary sense through which humans
drive. Whether cameras alone are sufficient for autonomous driving remains an open
question in industry: Waymo, which operates driverless robotaxis, combines LiDAR, radar
and cameras @waymo2024sixth, whereas Tesla has removed radar and ultrasonic sensors from
its vehicles and relies on eight cameras, without LiDAR or high-definition maps, for its
supervised driver-assistance system @tesla_vision @karpathy2021cvpr. In academic
research, early work on end-to-end learning demonstrated that a single front-facing camera
is sufficient to learn a steering policy directly from human demonstrations, first with
ALVINN @pomerleau1989alvinn and later with PilotNet @bojarski2016pilotnet. At the same
time, modern smartphones integrate a high-quality camera, an inertial measurement unit
and a processor capable of running neural networks in real time, which allows a phone to
serve as the complete sensing and computing unit of a small robot @muller2021openbot. A
smartphone mounted on a radio-controlled (RC) car therefore offers a low-cost platform on
which camera-only autonomous driving can be studied without dedicated sensors or
computers.

In the author's previous internship (Internship 2, December 2025), a smartphone was used
to drive a robot along a line with classical computer vision implemented in OpenCV:
the line was extracted by HSV colour thresholding or by Canny edge detection followed by
a Hough transform, and a set of hand-written rules converted its position into motion
commands. This approach proved ineffective in practice. The thresholds were sensitive to
illumination, shadows and cluttered backgrounds, so they had to be re-tuned for every
track and still failed under changing conditions, and the rule-based control made the
robot stop and rotate in place to realign with the line instead of steering smoothly.
These limitations motivate replacing hand-crafted perception and control with a policy
learned end to end from human demonstrations @bojarski2016pilotnet.

Motivated by these observations, this thesis investigates whether a low-cost
Ackermann-steered RC car, which perceives its environment solely through the monocular
camera of a smartphone and performs all computation on that phone, can learn to drive
around a track from human demonstrations. Rather than tuning rules by hand, the vehicle
learns to predict steering and throttle directly from camera images, conditioned on
high-level commands, for both forward and reverse driving.

== Objectives
The aim of this thesis is to develop a low-cost RC car that uses a smartphone as its only
sensor and computer and learns to drive around a track from camera images through
imitation learning. To achieve this aim, the following objectives are defined:

+ *Vehicle platform.* Convert an off-the-shelf Ackermann-steered RC car into an
  OpenBot-compatible robot, with a smartphone mount and an ESP32-S3 microcontroller that
  receives commands from the phone and drives the steering servo and the electronic
  speed controller (ESC).
+ *Application adaptation.* Modify the OpenBot Android application, which was designed
  for differential-drive robots, to control the car by steering and throttle instead of
  left and right wheel speeds, and extend its command set with turn indicators and
  reverse driving.
+ *Data collection.* Prepare a test track and record driving demonstrations on it with
  the application's data-logging function, pairing each camera image with the steering,
  throttle and high-level command given by the human driver.
+ *Policy learning.* Train end-to-end driving policies by imitation learning, with the
  conditional imitation learning network `cil_mobile_fast` @muller2021openbot
  @codevilla2018cil as the main model, supported by a small web application that
  simplifies running the training pipeline.
+ *Deployment and evaluation.* Deploy the trained models on the smartphone with
  TensorFlow Lite and evaluate them in closed loop on the track, in addition to offline
  metrics.

== Contributions
The main contributions of this thesis are:

- *An Ackermann-steering extension of OpenBot.* The OpenBot Android application and its
  serial protocol are extended from differential drive to steering-and-throttle control,
  with turn-indicator and reverse commands. Together with new firmware for an ESP32-S3
  board that drives a steering servo and an ESC, this allows an off-the-shelf RC car to
  be driven by a smartphone, with the phone as the only sensor and computer on the
  vehicle.
- *A camera-only driving dataset and training pipeline.* Driving demonstrations recorded
  on a purpose-built track, organised into sessions, and a pipeline that turns them into
  training data for end-to-end policies, accompanied by a lightweight web interface for
  running it.
- *A closed-loop evaluation of camera-only imitation learning on a real vehicle.*
  The conditional imitation learning model `cil_mobile_fast` is trained on this data,
  deployed on the phone and evaluated on the physical track, showing TODO: main result.

== Thesis outline
@ch:background reviews related work. @ch:hardware and @ch:firmware describe the vehicle,
@ch:app the phone app, @ch:training the learning pipeline, and @ch:results the evaluation.
