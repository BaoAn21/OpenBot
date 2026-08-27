# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

OpenBot turns smartphones into robot brains. This is a monorepo spanning hardware, firmware, and several independent software stacks that communicate over a serial link (phone <-> MCU) or WiFi (phone <-> controller/policy tooling). There is no single build for "the project" — always work inside the relevant subdirectory, each of which has its own toolchain:

- `android/` — Kotlin/Java Android apps (Gradle). Contains two apps built from one project: the **robot** app (`android/robot`) that runs on the phone mounted on the vehicle, and the **controller** app (`android/controller`) for driving it. Shared code lives in `android/comlib`.
- `firmware/openbot/` — Arduino/ESP32 C++ firmware (`openbot.ino`) that bridges the phone and the robot body's motors/sensors over serial.
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

Compile SDK 33 / target SDK 32, min API 21. Version compatibility issues between Android Studio and AGP are common — see `android/README.md` troubleshooting section if Gradle sync fails.

### Firmware (`firmware/openbot/`)

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

**Localization convention.** Every README, CONTRIBUTING, and DISCLAIMER file exists in multiple language variants (`.de-DE.md`, `.es-ES.md`, `.fr-FR.md`, `.ko-KR.md`, `.zh-CN.md`) alongside the English original. When updating docs, the English file is the source of truth; translations are maintained separately and are not expected to be updated in the same change.
