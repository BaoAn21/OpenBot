# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

OpenBot turns smartphones into robot brains. This is a monorepo spanning hardware, firmware, and several independent software stacks that communicate over a serial link (phone <-> MCU) or WiFi (phone <-> controller/policy tooling). There is no single build for "the project" — always work inside the relevant subdirectory, each of which has its own toolchain:

- `android/` — Kotlin/Java Android apps (Gradle). Contains two apps built from one project: the **robot** app (`android/robot`) that runs on the phone mounted on the vehicle, and the **controller** app (`android/controller`) for driving it. Shared code lives in `android/comlib`.
- `firmware/openbot/` — Arduino/ESP32 C++ firmware (`openbot.ino`) that bridges the phone and the robot body's motors/sensors over serial. Upstream multi-board sketch; still speaks the old left/right protocol.
- `firmware/openbot_esp32s3/` — our own firmware for the LilyGO T-Display-S3 (steering servo + H-bridge, screen as turn signals/reverse light). This is the one that matches the current robot app. Pinout and protocol in its `README.md`.
- `policy/` — Python/TensorFlow driving-policy training (imitation learning) plus a `frontend/` (React) and `openbot/server/` for visualizing training in-browser.
- `python/` — embedded-Linux alternative to the phone: runs the driving policy or joystick control directly from a Linux computer with a camera.
- `controller/` — four independent remote-control clients for the robot: `node-js/` (browser, Vite + Express), `web-server/` (cloud/WebRTC variant of the node-js controller), `python/` (keyboard/joystick over the terminal), `flutter/` (mobile controller app for Android/iOS).
- `open-code/` — "Playground" web app (React + Blockly) for building robot instruction programs visually; uses Firebase.
- `body/` — 3D-printable chassis designs and hardware BOMs (no code to build).

Because these stacks are independent, always `cd` into the specific subdirectory before running any build/lint/test command below.

## Commands by component

### Android apps (`android/`)

Primarily built through Android Studio (open the `android/` directory, select the `robot` or `controller` run configuration). From the CLI:

```bash
cd android
./gradlew assembleDebug          # build both apps
./gradlew :robot:assembleDebug   # build just the robot app
./gradlew :controller:assembleDebug
./gradlew checkStyle             # verify Java formatting (google-java-format), cross-platform via JavaExec
./gradlew checkStyleUnix         # same check via utils/checkStyle.sh (Unix only)
./gradlew applyStyle             # auto-fix Java formatting
```

The default `java` on this machine is JDK 27, which Gradle 7.6 can't run on ("Unsupported class file major version 71"). Prefix gradle calls with JDK 17:

```bash
export JAVA_HOME=~/.local/share/mise/installs/java/temurin-17; export PATH=$JAVA_HOME/bin:$PATH
```

Compile SDK 33 / target SDK 32, min API 21. Version compatibility issues between Android Studio and AGP are common — see `android/README.md` troubleshooting section if Gradle sync fails.

### Firmware for T-Display-S3 (`firmware/openbot_esp32s3/`)

The active firmware for this robot. Built and flashed with `arduino-cli` (ESP32 core 3.3.x + "GFX Library for Arduino"); the board shows up as `/dev/ttyACM0` and the user is in `uucp`, so flashing works from here:

```bash
arduino-cli compile --upload -p /dev/ttyACM0 \
  --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc,USBMode=hwcdc,FlashSize=16M,PSRAM=opi,PartitionScheme=app3M_fat9M_16MB" \
  firmware/openbot_esp32s3
```

`CDCOnBoot=cdc` is required or `Serial` won't reach the USB-C port. To exercise it without the phone, write the protocol messages (`h500`, `c<steering>,<throttle>`, `i...`, `f`) to `/dev/ttyACM0` at 115200 from a short Python script; the screen shows the result, but the user has to look at it. Pins, settings and protocol are documented in `firmware/openbot_esp32s3/README.md` — keep that README in sync when changing pins or messages. Motor driver: only `HBRIDGE` is implemented; `ESC` (brushed motor) is planned next.

### Firmware, upstream (`firmware/openbot/`)

Edited/flashed via the Arduino IDE, not a CLI build. Before compiling, set the hardware config macro at the top of `openbot.ino` (e.g. `OPENBOT DIY`, `OPENBOT PCB_V2`, `OPENBOT RTR_TT`, `OPENBOT RC_CAR`, `OPENBOT LITE`, `OPENBOT RTR_520`, `OPENBOT MTV`, `DIY_ESP32`) and the relevant feature flags (`HAS_VOLTAGE_DIVIDER`, `HAS_INDICATORS`, `HAS_SPEED_SENSORS_FRONT/BACK`, `HAS_SONAR`, `HAS_BUMPER`, `HAS_OLED`, `HAS_LEDS_*`, `BLUETOOTH`) — disabled features are compiled out to save flash/RAM.

### Policy training (`policy/`)

Requires a conda environment (`environment_linux.yml` / `_mac.yml` / `_win.yml`), not pip alone — the env files handle the TensorFlow/CUDA pinning that plain `pip install` gets wrong.

```bash
cd policy
conda env create -f environment_linux.yml   # once
conda activate openbot
pip install -r requirements.txt
jupyter notebook policy_learning.ipynb       # interactive training/eval
./dev.sh                                     # runs `adev runserver openbot/server` (live-reload training dashboard)
```

**Training runs in the local conda env, never Docker.** This machine is the GPU
box (RTX 3050, NVIDIA driver + CUDA already set up), so training is a direct
`conda activate openbot` + `python -m openbot.train ...` from `policy/` (paths
like `dataset_dir` in `openbot/utils.py` are relative to the working directory):

```bash
cd policy
conda activate openbot
python -m openbot.train --create_tf_record --model pilot_net \
    --batch_size 128 --num_epochs 100 --batch_norm
```

`Dockerfile`, `docker-run-train.sh`, `openbot-train.tar.gz` and
`TRAINING_ON_PC.md` are leftovers from an earlier setup where training had to be
shipped from a laptop without a GPU. Don't suggest them. Those runs also left
root-owned files behind in `policy/dataset` and `policy/models` (the container
ran as root through a bind mount); a `PermissionError` while writing
`matched_frame_ctrl.txt` or `train_preview.png` means another one surfaced, and
the fix is `sudo chown -R sim:sim policy/dataset policy/models`.

`policy/frontend` is a separate CRA app (own `package.json`): `npm start`, `npm run build`, `npm test` inside `policy/frontend`.

### Embedded-Linux control (`python/`)

```bash
cd python
pip install -r requirements.txt
python run.py --mode debug --dataset_path tests/test_data/logs1 --policy_path <path>   # replay a policy against recorded data, no camera/joystick needed
python run.py --mode inference --policy_path <path> --inference_backend tf   # live camera + policy
python run.py --mode joystick --control_mode dual                            # manual driving / data collection
pytest tests/                                # run all tests
pytest tests/test_infer.py                   # run a single test file
```

`--mode debug` is the fast way to exercise the policy-inference code path without hardware. `tests/` includes a `get_test_data.sh` helper for fetching fixture data/models used by the test suite (test_data, test_models for tf/tflite/openvino).

### Controllers (`controller/`)

Each has its own toolchain:

```bash
cd controller/node-js
npm install
npm start          # runs server + Vite client together (run-p dev:server dev:start-client)
npm run lint        # eslint --fix on server/*.js and client/*.js
npm test            # node --test

cd controller/web-server
npm install
npm start          # server + client via run-p

cd controller/flutter
flutter pub get
flutter run

cd controller/python
pip install -r requirements.txt
python keyboard-pygame.py   # or keyboard-click.py
```

### Playground (`open-code/`)

CRA app; requires Firebase project setup first (`open-code/src/services/README.md`).

```bash
cd open-code
npm install   # or yarn install
npm start
npm test
npm run build
```

## Architecture notes

**Data flow across stacks.** The robot app (`android/robot`) talks to the firmware (`firmware/openbot/openbot.ino`) over a serial (or Bluetooth, ESP32 boards only) link using a simple text protocol: driving commands and indicator state go one way, wheel-tick/voltage/sonar sensor readings come back. The same firmware protocol is what `python/joystick.py` and `python/realsense.py` speak when the embedded-Linux stack (`python/`) is used instead of a phone — so changes to the serial protocol need to stay in sync across `firmware/openbot/openbot.ino`, the Android robot app's serial-handling code, and `python/`.

**Two families of "controller".** Don't confuse `android/controller` (a native Android/companion-app controller, part of the Gradle multi-project build) with `controller/` at the repo root (four standalone web/desktop/mobile controller clients: node-js, web-server, python, flutter). They serve the same purpose (drive the robot remotely) but are unrelated codebases.

**Policy training vs. on-device inference.** `policy/` is where models are trained (TensorFlow, `policy/openbot/train.py`, driven from `policy_learning.ipynb`); the resulting model gets exported (e.g. `python/export_openvino.py`) and consumed either by the Android robot app (on-phone inference) or by `python/infer.py` (on-Linux inference via `run.py --mode inference`, selectable backend: `tf`, `tflite`, `openvino`). Data collected in joystick mode (`run.py --mode joystick` or the Android app's data-collection mode) is the training data format `policy/` expects.

**Android layouts are duplicated for landscape.** `android/robot/src/main/res/layout-land/`
holds landscape variants of 11 layouts (`fragment_autopilot`, `fragment_free_roam`,
`fragment_logger`, `fragment_object_nav`, `fragment_robot_info`,
`fragment_controller_mapping`, `fragment_edit_profile`, `fragment_blockly_executing`,
`fragment_bar_code_scanner`, `dialog_sensors`, `layout_bottom_sheet`). A view added to
only one variant makes view binding generate the field as `@Nullable` instead of
`@NonNull`, so it compiles and runs in one orientation and NPEs on rotation. Always add
a new view to both files, then confirm the field is `@NonNull` in the generated binding
(`robot/build/.../FragmentXBinding.java`). Note the two variants can differ
structurally - the landscape autopilot sheet is 400dp pinned to `end`, so a top-center
overlay that works in portrait lands behind it.

**Localization convention.** Every README, CONTRIBUTING, and DISCLAIMER file exists in multiple language variants (`.de-DE.md`, `.es-ES.md`, `.fr-FR.md`, `.ko-KR.md`, `.zh-CN.md`) alongside the English original. When updating docs, the English file is the source of truth; translations are maintained separately and are not expected to be updated in the same change.

**Driving control is Ackermann-style (steering + throttle), not differential (left/right).** `android/robot/src/main/java/org/openbot/vehicle/Control.java` and `Vehicle.java` represent commands as a `steering`/`throttle` pair in raw device units `[-255, 255]` (`Control.MAX`), not as independent left/right wheel speeds. `Control.fromLeftRight(left, right)` is the conversion point for callers that still produce a normalized differential-drive pair (TFLite models, the object tracker, phone/web controllers) — it maps `(left, right)` in `[-1, 1]` to `(steering, throttle)`. `Vehicle.getLeftSpeed()/getRightSpeed()` no longer exist; use `getSteering()`/`getThrottle()`. The serial protocol to the firmware changed accordingly: `sendControl()` now sends `c<steering>,<throttle>\n` instead of `c<left>,<right>\n`. `firmware/openbot_esp32s3/` parses this format; `firmware/openbot/openbot.ino` still parses `c<left>,<right>` and would misread it. Keep `python/` in sync too if it speaks the same protocol.

**Indicator values: -1 left, 0 off, 1 right, 2 reverse.** (`Enums.VehicleIndicator`; an earlier test build used 1 = reverse — that is gone.) `Vehicle.setIndicator()` sends them as `i1,0`, `i0,0`, `i0,1` and `i0,0,1`; the third field of `i<left>,<right>[,<reverse>]` is optional, so two-field messages still work. The indicator value is also the model's `cmd` input (`tflite/Autopilot*.java`) and is logged to `indicatorLog.txt` as the training `cmd`, so reverse shows up as `cmd 2`. Before training on such data, check `policy/openbot/data_augmentation.py`: `flip_sample` negates `cmd` and `augment_cmd` randomly turns 0 into ±1, which is wrong for a reverse value. On the gamepad the indicators toggle like a car stalk (`ControlsFragment.toggleIndicatorButton`): square = left, circle = right, triangle = reverse; pressing the active one again turns it off. The phone/web controller apps still use their left/right/stop commands.

**Pending:** Espressif's USB vendor ID (`12346`) is not in `android/robot/src/main/res/xml/device_filter.xml` yet, so the app doesn't auto-launch when the T-Display-S3 is plugged in.

When chaging code, remember to show me first.
