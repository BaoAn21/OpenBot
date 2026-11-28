= Hardware Platform <ch:hardware>

This chapter describes the vehicle used throughout this thesis: an off-the-shelf RC car
whose stock electronics were partly replaced so that it can be driven by a smartphone.
@sec:base-car introduces the base vehicle, @sec:mods the modifications made to it, and
@sec:electronics the control electronics and their wiring. The firmware running on the
microcontroller is described separately in @ch:firmware.

#figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image("../figures/car_front.jpg"),
    image("../figures/car_rear.jpg"),
    [(a) Front view], [(b) Top rear three-quarter view],
  ),
  caption: [The modified RC car with the smartphone, the microcontroller and its display
    mounted on the 3D-printed plate.],
) <fig:car>

== Base vehicle <sec:base-car>
The platform is a WLtoys 144001 (@fig:wltoys), a 1:14-scale four-wheel-drive RC car with
a brushed DC motor and Ackermann steering. It was chosen because it is inexpensive,
widely available with spare parts, and, unlike the differential-drive robots OpenBot was
designed for, steers like a full-size car. In its stock configuration, however, the car
is controlled only by its 2.4 GHz handheld transmitter through an integrated receiver
board, which offers no interface for an external controller. The drive and steering
electronics were therefore modified as described below.

#figure(
  image("../figures/wltoys_car.jpg", width: 50%),
  caption: [The WLtoys 144001 in its stock configuration (manufacturer's product photo).],
) <fig:wltoys>

== Modifications <sec:mods>

=== Electronic speed controller
The stock receiver/ESC board was replaced by a standalone brushed ESC (AJL 6860, 60 A,
@fig:esc-servo a), which accepts the standard 50 Hz RC servo signal (1000–2000 µs,
1500 µs = neutral) and can therefore be driven directly by a microcontroller pin. The
ESC is set to forward/reverse mode so that the car can reverse without a braking step,
and its built-in battery eliminator circuit (BEC, 6 V / 3 A) powers the steering servo.

=== Steering
On the stock car the steering servo is driven directly by the receiver board, so it
does not accept a standard servo signal and had to be replaced together with that
board. It was replaced by an MG996R (@fig:esc-servo b), a standard three-wire
metal-gear servo that is controlled by the same 50 Hz pulse signal as the ESC. Because
the servo arm rarely sits exactly centred on its spline, the pulse widths for the
centre and the two steering limits were measured on the car (1320 µs centre, 1550 µs
and 1080 µs at the left and right stops) and are set in the firmware.

#figure(
  grid(
    columns: 2,
    gutter: 8pt,
    align: horizon,
    image("../figures/esc_brushed.jpg", height: 4.2cm),
    image("../figures/servo_RC.jpg", height: 4.2cm),
    [(a) Brushed ESC], [(b) MG996R steering servo],
  ),
  caption: [Replacement drive and steering actuators (product photos).],
) <fig:esc-servo>

=== 3D-printed parts
The car's body shell was removed and replaced by a flat 3D-printed plate that carries
all added components (@fig:assembly). The plate is fastened to the chassis at the
original body mounts through a front and a rear adapter, so that the car itself
needed no further modification. On the plate sit a base for the phone clamp, which
holds the phone upright at the front so that its camera looks along the direction of
travel, and an inclined stand for the microcontroller, which tilts its display towards
the rear of the car. The microcontroller is enclosed in a two-part case. The individual
parts are shown in @fig:printed-parts.

#figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image("../figures/full_assembly_iso.jpg"),
    image("../figures/full_assembly_side.jpg"),
    [(a) Isometric view], [(b) Side view],
  ),
  caption: [CAD model of the 3D-printed mounting assembly with the phone and the
    microcontroller case in place.],
) <fig:assembly>

#figure(
  grid(
    columns: 3,
    gutter: 6pt,
    image("../figures/plate.jpg"),
    image("../figures/front_adapter.jpg"),
    image("../figures/rear_adapter.jpg"),
    [(a) Mounting plate], [(b) Front adapter], [(c) Rear adapter],
    image("../figures/phone_clamp_base.jpg"),
    image("../figures/display_stand.jpg"),
    image("../figures/tdisplay_case_bottom.jpg"),
    [(d) Phone clamp base], [(e) Display stand], [(f) Case, bottom],
  ),
  caption: [Individual 3D-printed parts.],
) <fig:printed-parts>

== Control electronics <sec:electronics>
A LilyGO T-Display-S3 (@fig:tdisplay), an ESP32-S3 development board with a built-in
1.9-inch display, connects the phone to the actuators. The phone powers the board and
exchanges serial messages with it over a single USB-C cable. The board generates the PWM
signals for the ESC (GPIO 1) and the steering servo (GPIO 10), while the drive battery
powers the ESC and, through its BEC, the servo; all grounds are shared. The display
faces backwards and acts as the car's turn signals and reverse light (@fig:car-rear).

#figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image("../figures/t_display_s3_board.jpg"),
    image("../figures/t_display_s3_board_screen.jpg"),
    [(a) Component side], [(b) Display side],
  ),
  caption: [The LilyGO T-Display-S3 development board (CAD model).],
) <fig:tdisplay>

#figure(
  image("../figures/car_back.jpg", width: 45%),
  caption: [Rear view of the car. The microcontroller's display, mounted on the
    inclined stand, shows the turn signals; the phone is connected to it by a single
    USB-C cable.],
) <fig:car-rear>

TODO: wiring diagram (phone → T-Display-S3 → ESC / servo, battery → ESC).

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    table.header([*Part*], [*Role*], [*Cost*]),
    [WLtoys 144001 (brushed)], [Chassis, drive motor, battery], [TODO],
    [AJL 6860 brushed ESC], [Drive motor control, BEC for the servo], [TODO],
    [MG996R servo], [Steering], [TODO],
    [LilyGO T-Display-S3], [PWM generation, turn-signal display], [TODO],
    [3D-printed parts], [Mounting plate, adapters, phone clamp base, display stand, case], [TODO],
    [Android phone], [Camera, inference, logging], [—],
  ),
  caption: [Bill of materials.],
) <tab:bom>
