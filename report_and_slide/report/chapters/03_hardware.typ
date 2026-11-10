= Hardware Platform <ch:hardware>

The vehicle is built on an off-the-shelf RC car with Ackermann steering
(@fig:car). The phone sits on top and talks to a LilyGO T-Display-S3 over USB.

#figure(
  image("../figures/rc-truck-banner.jpg", width: 85%),
  caption: [The RC car platform (placeholder photo).],
) <fig:car>

== Components
#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    table.header([*Part*], [*Role*], [*Cost*]),
    [RC car (AJL 6860 RTR)], [Chassis, ESC, steering servo], [TODO],
    [LilyGO T-Display-S3], [Motor control, turn-signal screen], [TODO],
    [Android phone], [Camera, inference, logging], [—],
  ),
  caption: [Bill of materials.],
) <tab:bom>
