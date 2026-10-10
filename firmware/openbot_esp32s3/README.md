# OpenBot firmware for LilyGO T-Display-S3

Firmware for an Ackermann-style OpenBot car (steering servo + one drive motor) using the
[LilyGO T-Display-S3](https://github.com/Xinyuan-LilyGO/T-Display-S3) (ESP32-S3) as the MCU.
The phone connects to the board's USB-C port. The built-in screen is mounted facing backwards
and works as the car's turn signals and reverse light, with steering/throttle gauges in the
middle.

This is a separate sketch from `../openbot/openbot.ino`: it speaks the steering/throttle
protocol (`c<steering>,<throttle>`) that the robot app sends, not the older left/right one.

## Wiring

The drive motor goes through either an RC car ESC (default) or an H-bridge, selected with
`MOTOR_DRIVER` at the top of `openbot_esp32s3.ino`. The steering servo is the same for both.

### ESC (`MOTOR_DRIVER ESC`, default)

Tested with an AJL 6860 RTR (60A brushed, 6 V / 3 A BEC). Any RC car ESC that takes a
standard 50 Hz servo signal should work.

| T-Display-S3 | Connects to | Notes |
|---|---|---|
| GPIO 1 | ESC signal (white) | 50 Hz, 1000–1500–2000 µs, 1500 = stop |
| GPIO 10 | Steering servo signal | 50 Hz, range set by `SERVO_*_US` |
| GND | ESC black, servo GND | All grounds must be connected |
| USB-C | Phone (OTG cable) | Serial at 115200 baud; also powers the board |

```
T-Display-S3        ESC 3-wire lead      Servo
GPIO 1  ─────────── white (signal)
GPIO 10 ──────────────────────────────── signal
GND     ─────────── black ────────────── GND
                    red (BEC 6 V) ────── +
```

- **Don't connect the ESC's red wire to the board.** It is the ESC's BEC output (6 V), used to
  power the servo; the board is powered by the phone over USB-C.
- **Set the ESC to F/R mode** (forward/reverse, no brake step). In F/B there is no reverse,
  and in F/B/R the first reverse command only brakes, which the app and the driving model
  don't expect.
- **Power-on order:** board first (phone/USB-C), then the ESC switch. The ESC learns neutral
  when it is switched on, and the board sends neutral (1500 µs) from boot and whenever the
  link is lost.

### H-bridge (`MOTOR_DRIVER HBRIDGE`)

| T-Display-S3 | Connects to | Notes |
|---|---|---|
| GPIO 1 | H-bridge IN1 | PWM 20 kHz, high = forward |
| GPIO 2 | H-bridge IN2 | PWM 20 kHz, high = reverse |
| GPIO 10 | Steering servo signal | 50 Hz, range set by `SERVO_*_US` |
| GND | H-bridge GND, servo GND, battery − | All grounds must be connected |
| USB-C | Phone (OTG cable) | Serial at 115200 baud; also powers the board |

```
T-Display-S3        H-bridge            Servo
GPIO 1  ─────────── IN1
GPIO 2  ─────────── IN2
GPIO 10 ──────────────────────────────── signal
GND     ─────────── GND ──────────────── GND
                    motor battery +      5-6 V from a BEC/regulator
```

- **Power the servo from a 5–6 V BEC or regulator**, not from the board's 3V3 pin. A servo can
  draw more current than the board supplies and make it reset.
- The H-bridge is driven with two PWM inputs (IN1/IN2). This works with:
  - L298N: keep the ENA jumper on
  - DRV8833, MX1508: connect directly
  - TB6612: tie PWMA and STBY high
  - BTS7960: GPIO 1 → RPWM, GPIO 2 → LPWM, tie R_EN and L_EN high

### Free pins for later

Only these GPIOs are free on the headers: **1, 2, 3, 10, 11, 12, 13, 16, 17, 18, 21, 43, 44**.
1 and 10 are used above (and 2 with the H-bridge); 43/44 are UART0 (TX/RX). GPIO 3 is a strapping pin, so avoid it
for anything that is pulled high or low at boot.

Do not use these, they are taken by the board itself:

| GPIO | Used by |
|---|---|
| 5, 6, 7, 8, 9 | Display control (RST, CS, DC, WR, RD) |
| 39, 40, 41, 42, 45, 46, 47, 48 | Display data bus (D0–D7) |
| 15 | Display power (must be HIGH) |
| 38 | Display backlight |
| 4 | Battery voltage sense |
| 0, 14 | Buttons (BOOT and KEY) |

To use different pins, change `PIN_STEERING_SERVO`, `PIN_ESC`, `PIN_MOTOR_IN1` and
`PIN_MOTOR_IN2` at the top of `openbot_esp32s3.ino`.

## Settings to tune

The steering values are measured per car, because the servo arm rarely sits exactly centred
on its spline. To measure them, temporarily set `SERVO_LEFT_US 1000`, `SERVO_CENTER_US 1500`,
`SERVO_RIGHT_US 2000`, then hold a steering value `s` by sending `h1000` and `c<s>,0` over
serial a few times a second; that gives a pulse of `1500 + s * 500 / 255` µs. Note the pulses
where the wheels are straight and where they just reach each stop. Either side may be the
larger number: that is what sets the steering direction (there is no invert flag).

All at the top of `openbot_esp32s3.ino`:

| Setting | Default | When to change |
|---|---|---|
| `SERVO_CENTER_US` | 1320 | Pulse with the wheels straight |
| `SERVO_LEFT_US` / `SERVO_RIGHT_US` | 1550 / 1080 | Pulses with the wheels just at the left / right stop |
| `THROTTLE_INVERT` | false | The car drives backwards when it should go forward |
| `MOTOR_DRIVER` | `ESC` | `HBRIDGE` for a two-input H-bridge |
| `ESC_FORWARD_MAX_US` | 2000 | Cap forward top speed (e.g. 1700 for first tests) |
| `ESC_REVERSE_MAX_US` | 1000 | Cap reverse top speed (e.g. 1300) |
| `ESC_NEUTRAL_US` | 1500 | The ESC doesn't stop at throttle 0 (it learns neutral at switch-on, so this rarely needs changing) |
| `ESC_DEADBAND_US` | 0 | The motor only starts well past neutral: skip that dead zone (try 30–80) |
| `MOTOR_MIN_PWM` | 0 | H-bridge: the motor hums but doesn't move at low throttle (try 60–100) |
| `MOTOR_BRAKE` | false | H-bridge: `true` = brake at zero throttle, `false` = coast |
| `HAS_GAMEPAD` | false | `true` to drive from an Xbox controller when no phone is attached |
| `GAMEPAD_MAX_THROTTLE` | 255 | Cap speed when driving with the Xbox controller (e.g. 150) |
| `GAMEPAD_DEADZONE` | 0.08 | The car creeps or steers with the stick/triggers at rest: raise it |

## Driving without the phone (Xbox controller)

With no phone attached, the board connects to an Xbox controller over Bluetooth LE and drives
from it. The phone always has priority: while its heartbeat is arriving, the controller is
ignored.

Only BLE controllers work, because the ESP32-S3 has no Bluetooth Classic: Xbox Series X|S
controllers, and Xbox One controllers (model 1708) with updated firmware. PlayStation and
Switch controllers do not work.

| Xbox control | Action |
|---|---|
| Left stick, left/right | Steering |
| RT | Forward (proportional) |
| LT | Reverse (proportional); throttle = RT − LT |
| X / B / Y | Left signal / right signal / reverse light; press again to turn off |

Pairing: switch the controller on and hold its pair button (top, next to the USB port) until
the Xbox logo blinks fast. The board connects to the first Xbox controller it finds and
reconnects to it automatically after that.

- If the controller goes out of range, the car keeps its last command until Bluetooth gives up
  on the link (about 2–6 s). Let go of the triggers before walking away.
- While a controller is connecting, the main loop can stall for up to about a second.

## Build and flash

Needs `arduino-cli`, the ESP32 core (tested with 3.3.12), the "GFX Library for Arduino"
library (tested with 1.6.8) and "XboxSeriesXControllerESP32_asukiaaa" (tested with 1.1.3; pulls
in NimBLE-Arduino 2.5.1):

```bash
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "GFX Library for Arduino"
arduino-cli lib install XboxSeriesXControllerESP32_asukiaaa
```

Flash from the repo root:

```bash
arduino-cli compile --upload -p /dev/ttyACM0 \
  --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc,USBMode=hwcdc,FlashSize=16M,PSRAM=opi,PartitionScheme=app3M_fat9M_16MB" \
  firmware/openbot_esp32s3
```

- `CDCOnBoot=cdc` is required, otherwise `Serial` does not go to the USB-C port and the phone
  can't talk to the board.
- On Linux the user needs to be in the group that owns `/dev/ttyACM0` (`uucp` on Arch,
  `dialout` on Debian/Ubuntu), then log out and back in.
- If the upload can't connect: hold BOOT, tap RST, release BOOT, upload again, then tap RST.

In the Arduino IDE, pick "ESP32S3 Dev Module" with USB CDC On Boot = Enabled, Flash Size =
16MB, PSRAM = OPI PSRAM.

## Serial protocol

115200 baud, one message per line (`\n`):

| Message | Meaning |
|---|---|
| `c<steering>,<throttle>` | Drive. Both in [-255, 255]; positive steering = right, positive throttle = forward |
| `h<ms>` | Heartbeat. If the next one doesn't arrive within `<ms>`, the car stops |
| `i<left>,<right>[,<reverse>]` | Indicators, each 0 or 1. The third field is optional |
| `f` | Feature request. The board replies `fTDISPLAY_S3:` |

The robot app's indicator values map to these messages: -1 → `i1,0` (left), 0 → `i0,0` (off),
1 → `i0,1` (right), 2 → `i0,0,1` (reverse).

Safety: until the first heartbeat arrives, and whenever the heartbeat times out, the board
falls back to the Xbox controller if one is connected. Otherwise steering is centred, the motor
stops, and all indicators and the reverse light are switched off; the screen then shows NO LINK.

## Screen

- Left and right edges: three amber chevrons per side that light up one after another
  (sequential turn signal); both sides together = hazard lights.
- Top of the centre panel: REVERSE badge, solid white while reversing.
- Centre: steering and throttle values with bars (throttle bar is green forward, orange in
  reverse).
- Bottom: who is driving: PHONE (green), XBOX (blue) or NO LINK (red).
