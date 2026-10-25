# Autopilot Session data: record, split, train

How to go from new recordings to the two Autopilot Session models:

- **fwd**: `donkey_memory`, drives forward (model type CMDNAV_MEM on the phone)
- **bwd**: `cil_mobile_fast`, drives backward (model type CMDNAV, gets cmd 2)

Everything runs from `policy/` in the conda env:

```bash
cd policy
conda activate openbot
```

## 1. Copy the recordings

Copy the recording folders from the phone (named like `20261025_110811`) into:

```
policy/dataset/fwdbwd/inbox/
```

Drive forward and backward in the same recording as you like; the split sorts it out.

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
python train_fwdbwd.py fwd --sessions 1 3 4          # only these sessions
python train_fwdbwd.py bwd --sessions 1 3 --num_epochs 60   # other options go to openbot.train
```

- It builds the training records for your choice of sessions first. They are only rebuilt
  when the choice or the data changes, so a second run starts training right away.
  `--build_only` builds without training.
- Settings used: fwd `--model donkey_memory --color_aug`, bwd `--model cil_mobile_fast
  --batch_norm --flip_aug`, both `--batch_size 128 --learning_rate 0.0003 --num_epochs 100`.
- The model lands in `models/<fwd|bwd>_s<sessions>_<settings>/`, for example
  `models/fwd_s1-2-3_donkey_memory_lr0.0003_bz128_mem3_color/`, with a
  `data_manifest.txt` listing every data folder it was trained on.
- If that model folder already exists, training stops. Rename the old folder, or pass
  `--resume` to continue it.
- Don't use `--create_tf_record` for these models: it builds from `dataset/train_data/`,
  which these models don't use.

## 5. Check and use the model

- **fwd**: judge it by `logs/eval_rollout.png` (closed loop) and on the robot. Its val
  numbers use the recorded memory and look better than it drives.
- Put `checkpoints/best.tflite` on the phone (the checkpoint with the best val loss).
- Note the run in `policy/EXPERIMENTS.md`: sessions, metrics, robot result.

## Where things are

```
dataset/fwdbwd/
  inbox/                          new recordings waiting for split_direction.py
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
```

How the split works:

- Forward: every stretch of throttle > 0 becomes its own folder (`_f00`, `_f01`, ...). A
  stretch ends at a reverse or a pause over 0.5 s, so the memory model never sees memory
  from before a reverse.
- Backward: all throttle < 0 frames of a recording in one folder, cmd set to 2, because
  that is what the phone gives the backward model.
- Stopped frames (throttle 0) go to neither.
- Images are hard links to the raw recording, so splitting costs no disk space.
