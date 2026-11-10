= Introduction <ch:intro>

== Motivation
Self-driving research usually needs expensive sensors and compute. A modern smartphone
already carries a camera, an IMU and a capable processor, so it can serve as the "brain"
of a small robot @muller2021openbot. TODO: why this matters for you.

== Objectives
+ Build a low-cost RC car platform controlled by a smartphone.
+ Collect driving data and train an end-to-end driving policy @bojarski2016pilotnet.
+ Evaluate the policy on a real track.

== Thesis outline
@ch:background reviews related work. @ch:hardware and @ch:firmware describe the vehicle,
@ch:app the phone app, @ch:training the learning pipeline, and @ch:results the evaluation.
