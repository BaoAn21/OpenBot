# Training the driving policy on the PC (GPU machine)

This laptop has no GPU, so training runs on the PC (Omarchy/Arch, RTX 3050)
instead. The training code + environment is packaged as a Docker image so
the PC never needs conda/TensorFlow/CUDA set up by hand — only the NVIDIA
driver, Docker, and `nvidia-container-toolkit`.

Files involved (all in `policy/`):

- `Dockerfile` — builds the training image (Ubuntu 22.04 + Miniconda +
  the `openbot` conda env from `environment_linux.yml` + `openbot/` code).
  `dataset/` and `models/` are NOT baked in.
- `.dockerignore` — keeps `dataset/`, `models/`, `frontend/` etc. out of
  the image build.
- `docker-run-train.sh` — one-command wrapper that mounts `dataset/` and
  `models/` as volumes and runs the container with `--gpus all`.
- `dist/openbot-train.tar.gz` — the built image, saved + gzipped, ready to
  copy to the PC without needing to rebuild there.

## 1. Get the image onto the PC

From this laptop, copy the saved image over (LAN, USB, whatever's easiest):

```bash
scp /home/sim/Work/OpenBot/policy/dist/openbot-train.tar.gz <pc-user>@<pc-host>:~/openbot-train.tar.gz
```

If the image needs rebuilding later (e.g. after code changes), rebuild and
re-save on the laptop:

```bash
cd /home/sim/Work/OpenBot/policy
docker build -t openbot-train .
docker save openbot-train:latest | gzip > dist/openbot-train.tar.gz
```

## 2. One-time setup on the PC (Omarchy/Arch)

```bash
sudo pacman -S docker nvidia-container-toolkit
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # then log out/in (or `newgrp docker`)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# sanity check that the GPU is visible inside a container:
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see the RTX 3050 listed. If this fails, the driver or
`nvidia-container-toolkit` config is the problem — fix that before going
further (training will just silently fall back to CPU otherwise).

## 3. Load the image on the PC

```bash
docker load < ~/openbot-train.tar.gz
docker images openbot-train   # confirm it's there
```

## 4. Get the code onto the PC

The repo is a git fork (`origin` = `https://github.com/BaoAn21/OpenBot.git`,
branch `my-version`). On the laptop, commit and push the training code
changes (Ackermann-style loss/metric/flip-augmentation fixes, plus the
`Dockerfile` / `.dockerignore` / `docker-run-train.sh`), then on the PC:

```bash
git clone https://github.com/BaoAn21/OpenBot.git
cd OpenBot
git checkout my-version
git pull
```

`dataset/` and `models/` are gitignored, so they won't come through git —
see next step.

## 5. Get the dataset onto the PC

Sync `policy/dataset/` separately (rsync, USB drive, or re-download the
same Google Drive zips and extract them, same as we did on the laptop).

The training code expects this layout inside `policy/`:

```
dataset/
  train_data/
    <session_name_1>/
      <recording_1>/images/...
      <recording_1>/sensor_data/...
      <recording_2>/...
    <session_name_2>/...
  test_data/
    <session_name>/...
```

(See `README.md`'s "Data Collection" section for the full explanation —
each recording is one extracted zip from the phone/app, grouped into named
session folders, split ~80/20 between `train_data` and `test_data`.)

## 6. Run training

```bash
cd OpenBot/policy
./docker-run-train.sh --create_tf_record --model pilot_net --batch_size 128 --num_epochs 100 --batch_norm --flip_aug
```

Notes on flags (full list: `docker run --rm openbot-train --help`):

- `--create_tf_record` — only needed the first time, or after the dataset
  changes (converts `dataset/train_data` + `dataset/test_data` into
  tfrecords). Omit it on reruns to reuse the existing tfrecords and start
  faster.
- `--model pilot_net --batch_size 128 --num_epochs 100 --batch_norm` — the
  README's recommended settings for a deployment-quality model.
- `--flip_aug` — random horizontal-flip augmentation. Now fixed to
  correctly negate steering and keep throttle unchanged for our
  Ackermann-style control data (previously would have swapped
  steering/throttle, which is wrong for this control scheme).
- `--resume` — continue from the last checkpoint instead of retraining
  from scratch.

## 7. Get the trained model

`docker-run-train.sh` bind-mounts `models/` from the host, so output lands
directly on the PC's filesystem — nothing is trapped inside the container:

```
policy/models/<model_name>/checkpoints/
  best-train.tflite
  best.tflite / best-val.tflite
  last.tflite
```

Pick one (usually `best.tflite` or `last.tflite`), copy it to this laptop
or straight to the phone, rename it to `autopilot_float.tflite`, and drop
it into the Android app at:

```
android/robot/src/main/assets/networks/autopilot_float.tflite
```

Then recompile/reinstall the robot app in Android Studio.

## Known caveat: steering data is one-sided

Across the `28_08_session` dataset collected so far, the `steering` column
only ever ranges from -255 to 0 (never positive) — see `ctrlLog.txt` in
any session's `sensor_data/` folder. If that wasn't intentional, the model
has no examples of turning right and won't learn to. Worth collecting a
balanced session (both left and right turns) before a serious training
run.
