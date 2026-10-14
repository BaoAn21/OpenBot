# Experiment log: driving policies

Running log of the datasets, trained models and results, kept for the thesis.
`policy/models/` and `policy/dataset/` are gitignored, so this file is the only
versioned record of what is in them. Add an entry for every new run (template at the
bottom) and keep the old ones: negative results are results too.

Last updated: 2026-10-14.

## Setup in short

- **Vehicle:** RC car, Ackermann steering. Labels are `(steering, throttle)` in raw device
  units `[-255, 255]`, divided by 255 for training (see `CLAUDE.md`).
- **Command (`cmd`)** = the indicator: `-1` left, `0` straight/off, `1` right, `2` reverse.
  **Sessions recorded before 2026-09-26 used `1` = reverse.** The current meaning was
  introduced in commit `d5bf8cdc` on 2026-09-26.
- **Scope:** forward driving only. Reverse was dropped from the model's scope on 2026-10-10
  (`make_forward_only.py`, commit `304689f3`).
- **Loss history:** before 2026-10-04 the autopilot models used the older loss, which folded
  throttle into the weight (`command_weight`). From commit `e03f2c11` on they use
  `losses.sq_weighted_mse_angle`. `donkey_memory` always uses plain MSE. **Loss values are
  not comparable across these groups; compare models by their metrics.**

## Datasets

| Folder (under `policy/dataset/`) | Recorded | Sessions train / test | Frames train / test | cmd | Notes |
|---|---|---|---|---|---|
| `_other_sessions/train_data/my_dataset_1`, `_other_sessions/test_data/my_dataset_2` | 2026-08-28, 09-18, 09-23 | 66 / 18 | 5,871 total after forward-only (5,397 reverse frames removed) | 0 only (was 1 = reverse) | Forward/reverse driving on the small track. The 09-23 sessions were reverse-only and are empty after `make_forward_only.py`. Originals kept as `*.with_reverse`; `make_forward_only.py --restore` brings them back. |
| `train_data/indicator_training`, `test_data/indicator_training` | 2026-09-27 | 25 / 6 | 2,209 / 685 | 0 and 1 (right) | Indicator dataset 1. 2 train sessions have no labels and are skipped. |
| `train_data/indicator_data_session_2`, `test_data/indicator_data_session_2` | 2026-10-13 evening, 2026-10-14 morning | 32 / 8 | 3,231 / 806 | 0 and 1 (right) | Indicator dataset 2. Test sessions: `20261013_203838`, `_203927`, `_204019`, `20261014_071641`, `_071735`, `_071828`, `_071921`, `_071941`. |
| `_other_sessions/empty_sessions/20261013_203750` | 2026-10-13 | n/a | 39 frames, no controls | n/a | Empty `ctrlLog.txt`; matching raises on it. |
| `policy/dataset_reverse/` | built 2026-09-09 | n/a | n/a | n/a | Reverse-only copy built by `build_reverse_dataset.py` for the reverse test runs. |

**There is no left-turn (cmd -1) data in any indicator set yet.**

### Test sets have changed over time

The metrics in each model's `logs/` were measured on whatever `test_data/` held when it was
trained, so they are only comparable within the same test set:

- **T1:** `my_dataset_2` (18 sessions), forward + reverse, before 2026-10-10.
- **T1-fwd:** `my_dataset_2`, forward-only (after 2026-10-10).
- **T2:** `indicator_training` test (6 sessions, 685 frames; 667 examples in the memory records).
- **T3:** T2 + `indicator_data_session_2` test (14 sessions, 1,491 frames; 1,449 memory examples).

For a fair comparison, re-evaluate the `.tflite` files on one fixed test set; see "How to
compare" below.

## Models

All under `policy/models/<folder>/checkpoints/`. `best.tflite` and `best-val.tflite` are
the checkpoint with the lowest val loss. Renamed copies (the names used on the phone) are
listed with the run they came from; identity was checked by md5.

### Indicator / memory models (current line of work)

| Folder | Date | Model and options | Trained on | Test set | Phone file | Best val epoch | val angle<0.1 / dir / throttle<0.1 / MAE | On the robot |
|---|---|---|---|---|---|---|---|---|
| `openbot_cil_mobile_fast_lr0.0003_bz128_bn` | 09-27 | cil_mobile_fast, BN, colour aug (always on), old loss | indicator 1 | T2 | `indicator_cil.tflite` (= best-val) | 89 | 0.599 / 0.914 / 0.549 / 0.121 | Good: the reference to beat. |
| `openbot_pilot_net_lr0.0003_bz128_bn` | 09-27 | pilot_net, BN, colour aug, old loss | indicator 1 | T2 | `indicator.tflite` (= best-val) | 64 | 0.466 / 0.736 / 0.359 / 0.195 | Not recorded. |
| `openbot_donkey_memory_lr0.0003_bz128_mem3` | 10-13 | donkey_memory, mem 3, no cmd, plain MSE, no augmentation | my_dataset_1, forward-only | T1-fwd | `donkey_mem.tflite` (= best-val) | 32 | 0.595 / 0.901 / 0.535 / 0.126 | Works on the small track (forward driving). |
| `openbot_donkey_memory_lr0.0003_bz128_mem3_cmd_rawcmd` | 10-13 | donkey_memory + `--mem_cmd`, **raw cmd** concatenated beside the 8000 image features (first version) | indicator 1 | T2 | `donkey_mem_indi.tflite` (= best-val) | 29 | 0.702 / 0.979 / 0.682 / 0.083 | Worse than cil. Folder renamed from `_mem3_cmd`. |
| `openbot_donkey_memory_lr0.0003_bz128_mem3_cmd` | 10-13 | donkey_memory + `--mem_cmd`, **cil-style cmd wiring** (16-16 MLP after Dense 100, raw cmd again before the heads) | indicator 1 | T2 | `donkey_cmd_new.tflite` | 28 | 0.697 / 0.960 / 0.711 / 0.088 | Tries to turn right, but late. Still worse than cil. |
| `openbot_donkey_memory_lr0.0003_bz128_mem3_cmd_color`, run A | 10-14 07:05 | as above + `--color_aug` | indicator 1 | T2 | `donkey_mem_color_aug_cmd.tflite` | overwritten by run B | overwritten by run B | Not recorded. |
| `openbot_donkey_memory_lr0.0003_bz128_mem3_cmd_color`, run B | 10-14 07:36 | as above + `--color_aug` | indicator 1 + 2 (5,275 / 1,449 examples) | T3 | `donkey_color_cmd_2.tflite` (= best-val) | 20 | 0.548 / 0.879 / 0.622 / 0.113 | **Drives much better; first memory model that works well with the indicator.** |

Run B's val numbers are on T3 (bigger, harder) and are not comparable with the T2 rows above.
Run A's checkpoints and logs were overwritten by run B; only its `.tflite` survives.

### Earlier experiments (before the indicator work)

These come from before the cmd meaning changed, and some from before the loss change. The
data they used is reconstructed from dates and file names, so treat the "Trained on" column
as **probable**, not certain.

| Folder | Date | Model and options | Trained on (probable) | Phone file(s) | Best val epoch | val angle / dir | Notes |
|---|---|---|---|---|---|---|---|
| `openbot_pilot_net_lr0.0003_bz128_bn_flip` | 09-06 | pilot_net, BN, flip aug | my_dataset (08-28 sessions) | `best.tflite` | 78 | 0.697 / 0.856 | Earliest run; logged before the throttle metrics existed. |
| `reverse_test_pilot_net_lr0.0003_bz128_bn` | 09-09 | pilot_net, 20 epochs | `dataset_reverse` | `best.tflite` | 19 | 0.951 / 0.951 | val_loss 0.0, a loss bug fixed in the next run. |
| `reverse_test_fixedloss_pilot_net_lr0.0003_bz128_bn` | 09-09 | pilot_net, 20 epochs, fixed loss | `dataset_reverse` | `reverse.tflite` (= best-val) | 18 | 0.718 / 0.718 | "Can it reverse" test. |
| `openbot_pilot_net_seq_lr0.0003_bz128_bn_flip_seq0-2-4-6-8_trim2` | 09-09 to 09-18 | pilot_net_seq, frames 0,2,4,6,8, flip, trim 2 | my_dataset (with reverse) | `seq.tflite` (= best-val) | 77 | 0.763 / 0.920 | Stacked frames to tell forward from reverse. |
| `openbot_pilot_net_seq_lr0.0003_bz128_bn_seq0-2-4-6-8_trim2` | 09-18 to 09-19 | as above, no flip | my_dataset (with reverse) | `seqnoflip.tflite` (09-18), `newseqnoflip.tflite` (= best-val, 09-19) | 90 | 0.728 / 0.888 | |
| `openbot_pilot_net_lr0.0003_bz128_bn`, earlier runs | 09-21 to 09-22 | pilot_net, BN | my_dataset with cmd 1 = reverse | `cmdModel.tflite`, `cmd50.tflite` (09-21), `cmdModelNew.tflite` (09-22) | overwritten | overwritten | Reverse-as-cmd experiments; logs overwritten by the 09-27 indicator run. |
| `openbot_pilot_net_seq_lr0.0003_bz128_bn_seq0-2-4_trim0` | 10-06 | pilot_net_seq, frames 0,2,4, trim 0, new loss | unknown, records built 10-06 | `pilot_seq_seq0-2-4.tflite` (= best-val) | 43 | 0.314 / 0.570 | Weak metrics; unclear why. Fill in if you remember. |

## Offline comparison on T2 (2026-10-13)

Done by running the deployed `.tflite` files on the 667 memory test examples of
indicator 1. "Closed loop" feeds the memory model its own predictions, starting from
zeros at the start of each session, the way the phone does.

| Model | Mode | Steering MAE | angle<0.1 | Direction | Throttle MAE |
|---|---|---|---|---|---|
| `indicator_cil.tflite` | n/a | 0.113 | 60.0% | 91.3% | 0.116 |
| `donkey_mem_indi.tflite` (raw cmd) | teacher-forced | 0.076 | 70.0% | 97.9% | 0.091 |
| `donkey_mem_indi.tflite` (raw cmd) | closed loop | 0.094 | 64.6% | 97.6% | 0.092 |
| `donkey_cmd_new.tflite` (cil-style cmd) | closed loop | 0.100 | 65.1% | 95.2% | 0.091 |

Steering change when cmd is flipped from 0 to 1 on the same frame (the data's gap between
cmd 1 and cmd 0 frames is +0.281):

| Model | Mean | Median |
|---|---|---|
| `indicator_cil.tflite` | +0.316 | +0.190 |
| `donkey_mem_indi.tflite` (raw cmd) | +0.093 | +0.044 |
| `donkey_cmd_new.tflite` (cil-style cmd) | +0.108 | +0.082 |

How much the raw-cmd memory model relies on its memory: with a random image from another
frame plus the real memory, it still had 86.7% direction accuracy. With the real image and
zero memory, its steering MAE was 0.119.

Not yet evaluated offline: `donkey_mem_color_aug_cmd.tflite`, `donkey_color_cmd_2.tflite`.

## Findings so far

1. **Teacher-forced validation flatters memory models.** Their val metrics use the recorded
   memory, which is almost the answer, since consecutive frames are 33 ms apart. The memory
   models beat cil offline, yet drove worse on the robot. Judge memory models by the closed
   loop (`logs/eval_rollout.png`) and by driving.
2. **The memory model learns to ignore cmd.** Mid-turn, the memory already explains the
   steering, so there is little training signal to use the indicator. A raw cmd value as 1
   of 8,003 inputs is barely used (+0.09 shift). cil-style wiring helped a little (median
   +0.044 to +0.082) but did not remove the shortcut.
3. **Offline replay did not reproduce the late turning.** On the recorded test drives the
   memory model started its turns on time, even when simulating the phone at 15 or 10 fps.
   The lateness came from the robot seeing its own, slightly different trajectory and
   lighting (distribution shift), where the model fell back on its memory.
4. **More data plus colour augmentation fixed it on the robot.** Indicator 2 (+3,231 frames,
   more right turns) and `--color_aug` together gave the first memory model that drives
   well with the indicator. Their separate contributions are not isolated yet.
5. **cil_mobile_fast is a strong baseline for indicator following.** It has a cmd MLP and
   feeds raw cmd into the last two layers, so one weight can shift the steering per command.
   The name is misleading: it is not the branched CIL architecture.

## Open questions / next steps

- [ ] Retrain `cil_mobile_fast` on indicator 1 + 2 for a fair comparison with run B.
      Its current model is on indicator 1 with the old loss.
- [ ] Separate the two effects: train `--mem_cmd` on indicator 1 + 2 **without**
      `--color_aug`.
- [ ] Record left-turn (cmd -1) sessions; nothing has been trained on them yet.
- [ ] Freeze a benchmark test set (for example T3) and stop changing it, so later models
      stay comparable.
- [ ] Save the offline comparison script in the repo. The one used above lived in a
      temporary folder and is gone; the method is described in "How to compare".
- [ ] Ideas not tried yet: the turn-weighted loss for donkey_memory, memory dropout or
      noise during training (against the memory shortcut), and a frame spacing for the
      memory if the phone's fps is well below 30.
- [ ] Measure the phone's real inference fps in the Autopilot Memory screen.

## How to compare models

1. Use the same test records for every model; for memory models, `dataset/tfrecords_mem3/test.tfrec`.
2. Run the `.tflite` file the phone actually uses, through `tf.lite.Interpreter`, finding
   inputs by name (`img_input`, `cmd_input`, `mem_input`).
3. For memory models, report both teacher-forced and closed-loop results.
4. Report per cmd: steering MAE, angle<0.1, direction, throttle MAE. Also report the cmd 0 to 1
   steering shift and the turn-start timing against the label.
5. The robot test is the final judge. Note the track, the lighting, how many laps or
   attempts, how many turns were taken correctly, and anything odd.

## Template for a new run

```
### <folder name>, <date>
- Command: python -m openbot.train ...
- Git commit: <hash> (and whether the working tree had uncommitted changes)
- Trained on: <dataset folders>, <train/test examples from the tfrecord log>
- Test set: T?
- Best val epoch and metrics (from logs/log.csv):
- Phone file name: <renamed .tflite>, which checkpoint it is (best-val / last)
- Offline comparison (same test set as the others):
- Robot test: track, lighting, attempts, result, notes or video link
- Conclusion / next step:
```
