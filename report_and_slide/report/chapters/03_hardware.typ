= Hardware Platform <ch:hardware>

This chapter describes the vehicle used throughout this thesis: an off-the-shelf RC car
whose stock electronics were partly replaced so that it can be driven by a smartphone.
@sec:base-car introduces the base vehicle, @sec:mods the modifications made to it, and
@sec:electronics the control electronics and their wiring. The firmware running on the
microcontroller is described separately in @ch:firmware.

#figure(
  image("../figures/rc-truck-banner.jpg", width: 85%),
  caption: [The modified RC car with the smartphone mounted (placeholder photo).],
) <fig:car>

== Base vehicle <sec:base-car>
The platform is a WLtoys 144001, a 1:14-scale four-wheel-drive RC car with a brushed
DC motor and Ackermann steering. It was chosen because it is inexpensive, widely
available with spare parts, and, unlike the differential-drive robots OpenBot was
designed for, steers like a full-size car. In its stock configuration, however, the car
is controlled only by its 2.4 GHz handheld transmitter through an integrated receiver
board, which offers no interface for an external controller. The drive and steering
electronics were therefore modified as described below.

== Modifications <sec:mods>

=== Electronic speed controller
The stock receiver/ESC board was replaced by a standalone brushed ESC (AJL 6860, 60 A),
which accepts the standard 50 Hz RC servo signal (1000–2000 µs, 1500 µs = neutral) and
can therefore be driven directly by a microcontroller pin. The ESC is set to
forward/reverse mode so that the car can reverse without a braking step, and its
built-in battery eliminator circuit (BEC, 6 V / 3 A) powers the steering servo.

=== Steering
TODO: what was replaced and why (e.g. the stock servo replaced by a standard three-wire
servo, model ...). Because the servo arm rarely sits exactly centred on its spline, the
pulse widths for the centre and the two steering limits were measured on the car
(1320 µs centre, 1550 µs and 1080 µs at the left and right stops) and are set in the
firmware.

=== 3D-printed parts
TODO: list the printed parts (e.g. phone mount, microcontroller holder, ...) with a
photo or render of each, and the material and printer used.

== Control electronics <sec:electronics>
A LilyGO T-Display-S3, an ESP32-S3 development board with a built-in 1.9-inch display,
connects the phone to the actuators. The phone powers the board and exchanges serial
messages with it over a single USB-C cable. The board generates the PWM signals for the
ESC (GPIO 1) and the steering servo (GPIO 10), while the drive battery powers the ESC
and, through its BEC, the servo; all grounds are shared. The display faces backwards and
acts as the car's turn signals and reverse light.

TODO: wiring diagram (phone → T-Display-S3 → ESC / servo, battery → ESC).

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    table.header([*Part*], [*Role*], [*Cost*]),
    [WLtoys 144001 (brushed)], [Chassis, drive motor, battery], [TODO],
    [AJL 6860 brushed ESC], [Drive motor control, BEC for the servo], [TODO],
    [Steering servo (TODO model)], [Steering], [TODO],
    [LilyGO T-Display-S3], [PWM generation, turn-signal display], [TODO],
    [3D-printed parts], [Phone mount, TODO], [TODO],
    [Android phone], [Camera, inference, logging], [—],
  ),
  caption: [Bill of materials.],
) <tab:bom>
