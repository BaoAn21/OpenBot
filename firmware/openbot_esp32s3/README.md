# OpenBot firmware for LilyGO T-Display-S3

Firmware for an Ackermann-style OpenBot car (steering servo + one drive motor) using the
[LilyGO T-Display-S3](https://github.com/Xinyuan-LilyGO/T-Display-S3) (ESP32-S3) as the MCU.
The phone connects to the board's USB-C port. The built-in screen is mounted facing backwards
and works as the car's turn signals and reverse light, with steering/throttle gauges in the
middle.

This is a separate sketch from `../openbot/openbot.ino`: it speaks the steering/throttle
protocol (`c<steering>,<throttle>`) that the robot app sends, not the older left/right one.

## Wiring

| T-Display-S3 | Connects to | Notes |
|---|---|---|
| GPIO 1 | H-bridge IN1 | PWM 20 kHz, high = forward |
| GPIO 2 | H-bridge IN2 | PWM 20 kHz, high = reverse |
| GPIO 10 | Steering servo signal | 50 Hz, 1000–1500–2000 µs |
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
1, 2 and 10 are used above; 43/44 are UART0 (TX/RX). GPIO 3 is a strapping pin, so avoid it
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

To use different pins, change `PIN_STEERING_SERVO`, `PIN_MOTOR_IN1` and `PIN_MOTOR_IN2` at the
top of `openbot_esp32s3.ino`.

## Settings to tune

All at the top of `openbot_esp32s3.ino`:

| Setting | Default | When to change |
|---|---|---|
| `SERVO_MIN_US` / `SERVO_MAX_US` | 1000 / 2000 | The servo pushes against the steering end stops: move closer to 1500 |
| `SERVO_CENTER_US` | 1500 | The car doesn't drive straight at steering 0 |
| `STEERING_INVERT` | false | The wheels turn the wrong way |
| `THROTTLE_INVERT` | false | The car drives backwards when it should go forward |
| `MOTOR_MIN_PWM` | 0 | The motor hums but doesn't move at low throttle (try 60–100) |
| `MOTOR_BRAKE` | false | `true` = brake at zero throttle, `false` = coast |
| `MOTOR_DRIVER` | `HBRIDGE` | `ESC` is not implemented yet |

## Build and flash

Needs `arduino-cli`, the ESP32 core (tested with 3.3.12) and the "GFX Library for Arduino"
library (tested with 1.6.8):

```bash
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "GFX Library for Arduino"
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

Safety: until the first heartbeat arrives, and whenever the heartbeat times out, steering is
centred, the motor stops, and all indicators and the reverse light are switched off. The screen
then shows NO LINK.

## Screen

- Left and right edges: three amber chevrons per side that light up one after another
  (sequential turn signal); both sides together = hazard lights.
- Top of the centre panel: REVERSE badge, solid white while reversing.
- Centre: steering and throttle values with bars (throttle bar is green forward, orange in
  reverse).
- Bottom: LINK / NO LINK.
