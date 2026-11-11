# Autopilot Session data: record, split, train

How to go from new recordings to the two Autopilot Session models:

- **fwd**: `donkey_memory` (default) or `cil_mobile_fast_memory`, drives forward (model
  type CMDNAV_MEM on the phone)
- **bwd**: `cil_mobile_fast` (default), drives backward (model type CMDNAV, gets cmd 2), or
  `cil_mobile_fast_memory` or `donkey_memory_reverse` (model type CMDNAV_MEM)

`cil_mobile_fast_memory` is cil_mobile_fast plus donkey_memory's memory of its last 3
outputs. `donkey_memory_reverse` is donkey_memory with a tanh throttle instead of the
sigmoid, so it can reverse (plain donkey_memory can't). Pick either with `--model`.

Everything runs from `policy/` in the conda env:

```bash
cd policy
conda activate openbot
```

## 1. Copy the recordings

Copy the recordings from the phone (named like `20261025_110811`, the folders or the
`.zip` files) into:

```
policy/dataset/fwdbwd/inbox/
```

Drive forward and backward in the same recording as you like; the split sorts it out.
`split_direction.py` unpacks the zips into folders and deletes them, also on `--dry_run`.
A broken zip (cut off on the phone) is reported and deleted.

## 2 + 3. Split forward/backward and train/test

```bash
python split_direction.py --dry_run                       # look first, changes nothing
python split_direction.py --note "what this batch is"     # makes a new session
```

- Each run of the script turns everything in the inbox into **one session**
  (`session_1`, `session_2`, ...). The note is what `--list` shows later, so say what is
  special: the date, the light, "trial only, not for the final model".
- To add recordings to an existing session instead: `python split_direction.py --session 3`.
- 20% of the recordings go to test, picked at random (`--test_fraction 0.25` to change).
  A recording goes to train or test as a whole, forward and backward part together.
- Recordings with no frames or controls go to `parked/empty/`; recordings the autopilot
  drove go to `parked/autopilot/`. Nothing else is ever deleted.

## 4. Train

```bash
python train_fwdbwd.py --list                        # what each session holds
python train_fwdbwd.py fwd                           # forward model, all sessions
python train_fwdbwd.py bwd                           # backward model, all sessions
python train_fwdbwd.py fwd --model cil_mobile_fast_memory   # forward, cil with memory
python train_fwdbwd.py bwd --model cil_mobile_fast_memory   # backward, cil with memory
python train_fwdbwd.py bwd --model donkey_memory_reverse    # backward, donkey with memory
python train_fwdbwd.py fwd --sessions 1 3 4          # only these sessions
python train_fwdbwd.py bwd --sessions 1 3 --num_epochs 60   # other options go to openbot.train
```

- It builds the training records for your choice of sessions first. They are only rebuilt
  when the choice or the data changes, so a second run starts training right away.
  `--build_only` builds without training.
- Settings used: `donkey_memory --color_aug --mem_len 3`, `cil_mobile_fast --batch_norm
  --flip_aug`, `cil_mobile_fast_memory --batch_norm --flip_aug --mem_len 3`,
  `donkey_memory_reverse --color_aug --mem_len 3`, all
  `--batch_size 128 --learning_rate 0.0003 --num_epochs 100 --early_stop 20`.
- Training stops once val loss hasn't improved for 20 epochs (`--early_stop 20`), so
  100 epochs is only the upper limit. `best.tflite` is the best-val checkpoint either way.
  `--early_stop 0` trains all epochs; `--early_stop 30` waits longer (the cil models'
  val loss is noisier).
  The flip mirrors the memory too (its steering negated), so it stays consistent.
- `--flip_aug` also works for `donkey_memory` and `donkey_memory_reverse` (off by default,
  e.g. `python train_fwdbwd.py fwd --flip_aug`); the folder gets `_flip`. It assumes a
  mirrored frame is still a valid one: phone mounted near the centre, no keeping to one
  side of the track.
- `--mem_aug` (any memory model, e.g. `python train_fwdbwd.py fwd --mem_aug`) makes the
  memory unreliable in training - sometimes zeroed, otherwise offset at random - so the
  model has to look at the image more. No records rebuild; the folder gets `_memaug`.
  Train loss goes up and val may get a little worse; compare it with a run without it
  on the same sessions by the closed-loop numbers and on the robot.
- The model lands in `models/<fwd|bwd>_s<sessions>_<settings>/`, for example
  `models/fwd_s1-2-3_donkey_memory_lr0.0003_bz128_mem3_color/`, with a
  `data_manifest.txt` listing every data folder it was trained on.
- If that model folder already exists, training stops. Rename the old folder, or pass
  `--resume` to continue it.
- Don't use `--create_tf_record` for these models: it builds from `dataset/train_data/`,
  which these models don't use.

## 5. Check and use the model

- **Memory models** (donkey_memory, donkey_memory_reverse, cil_mobile_fast_memory): judge them by `logs/eval_rollout.png` (closed loop) and on
  the robot. Their val numbers use the recorded memory and look better than they drive, so
  don't compare them with the plain bwd model's. Don't compare loss values between
  donkey_memory (mse) and cil_mobile_fast_memory (turn-weighted loss) either.
- On the phone, cil_mobile_fast_memory and donkey_memory_reverse are added as CMDNAV_MEM; the
  Autopilot Session backward spinner offers CMDNAV and CMDNAV_MEM models. The phone clamps
  throttle to >= 0 forward and <= 0 backward.
- Put `checkpoints/best.tflite` on the phone (the checkpoint with the best val loss).
- Note the run in `policy/EXPERIMENTS.md`: sessions, metrics, robot result.

## 6. Autopilot runs: how the models drive, and your takeovers as data

The Autopilot Session screen's **Logging** switch (or the gamepad's x) records a run to
`Documents/OpenBot/autopilot_<date>.zip` on the phone: every frame the models drove, who
drove it (the model, or you when you take over with the stick) and what the model
predicted, also while you drove. Copy the zips into `dataset/fwdbwd/autopilot_runs/`.

```bash
python autopilot_log.py report                 # every run: takeovers per minute, plots
python autopilot_log.py extract --dry_run      # what extract would take
python autopilot_log.py extract                # your takeovers -> inbox/<run>_takeover/
python split_direction.py --note "takeovers of fwd_s1-2-3, 11-06"
```

- `report` prints per run and per model how long the model and you drove, the takeovers
  per minute the model drove (the number to compare models by on the robot), and how far
  the model was from you while you drove. It saves `autopilot_report.png` in the run's
  folder: the model's prediction and the command over time, your takeovers shaded.
- `extract` keeps only the frames you drove, each takeover minus its first 0.3 s (your
  reaction, the car still doing what the model did) and its last 0.3 s before you let go
  (`--trim_start`, `--trim_end`). The result is a normal recording for the inbox.
- Never put a run itself in the inbox: its labels are mostly the model's own output.
  `split_direction.py` parks it in `parked/autopilot/` if you do.
- Give the takeovers their own session, so you choose whether a model trains on them
  (`train_fwdbwd.py fwd --sessions 1 2 3 4`). They are all corrections, so train them
  together with normal driving, not alone.

## Where things are

```
dataset/fwdbwd/
  inbox/                          new recordings waiting for split_direction.py
  autopilot_runs/                 Autopilot Session logs, for autopilot_log.py
  raw/session_<N>/<recording>/    the recordings as recorded (plus matched_* files)
  raw/session_<N>/note.txt        your note
  raw/session_<N>/test_recordings.txt   which recordings went to test
  train/fwd/session_<N>/<recording>_fNN/   one folder per forward run
  train/bwd/session_<N>/<recording>/       the backward frames, cmd 2
  test/fwd/session_<N>/...
  test/bwd/session_<N>/...
  parked/empty/, parked/autopilot/
dataset/tfrecords_mem3/           fwd records + manifest.txt (what they hold)
dataset/tfrecords/                bwd records + manifest.txt
dataset/tfrecords_bwd_mem3/       bwd memory model records + manifest.txt
```

How the split works:

- Forward: every stretch of throttle > 0 becomes its own folder (`_f00`, `_f01`, ...). A
  stretch ends at a reverse or a pause over 0.5 s, so the memory model never sees memory
  from before a reverse.
- Backward: all throttle < 0 frames of a recording in one folder, cmd set to 2, because
  that is what the phone gives the backward model.
- Stopped frames (throttle 0) go to neither.
- The bwd memory records cut the memory where two backward frames are over 0.5 s
  apart, so a reverse never remembers the previous one (the phone clears it on a switch).
- Images are hard links to the raw recording, so splitting costs no disk space.
