= Firmware <ch:firmware>

The ESP32-S3 receives text commands from the phone over serial at 115200 baud
(@tab:protocol). Driving commands use steering and throttle, not left/right wheel speeds.

#figure(
  table(
    columns: (auto, 1fr),
    align: left,
    table.header([*Message*], [*Meaning*]),
    [`h<ms>`], [Heartbeat; stop if no message within this time],
    [`c<steering>,<throttle>`], [Drive command, each in $[-255, 255]$],
    [`i<left>,<right>[,<reverse>]`], [Indicators / reverse light],
    [`f`], [Request firmware info],
  ),
  caption: [Serial protocol between phone and microcontroller.],
) <tab:protocol>

A typical exchange (@lst:serial):

#figure(
  ```text
  h500        // heartbeat timeout 500 ms
  i0,0        // indicators off
  c-40,120    // steer slightly left, drive forward
  ```,
  caption: [Example serial messages.],
) <lst:serial>
