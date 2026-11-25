# Experiment log: driving policies

Running log of the datasets, trained models and results, kept for the thesis.
`policy/models/` and `policy/dataset/` are gitignored, so this file is the only
versioned record of what is in them. Add an entry for every new run (template at the
bottom) and keep the old ones: negative results are results too.

Last updated: 2026-10-25.

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
  `losses.sq_weighted_mse_angle`. `donkey_memory` uses plain MSE unless trained with `--turn_loss` (`_turnloss` in the name). **Loss values are
  not comparable across these groups; compare models by their metrics.**

## Datasets

**2026-11-16: new base, camera tilted a little up at the front.** Everything recorded before
(the table below: `fwdbwd/` sessions 1-11, the autopilot runs, `_other_sessions/`) was moved
to `dataset/_archive/old_base_2026-11-16/` and the records (`tfrecords*/`) deleted;
`fwdbwd/` starts again at session 1. The models in `models/` up to this date were trained on
the old camera angle, so their numbers don't carry over to the new data.

| Folder (under `policy/dataset/`) | Recorded | Sessions train / test | Frames train / test | cmd | Notes |
|---|---|---|---|---|---|
| `_other_sessions/train_data/my_dataset_1`, `_other_sessions/test_data/my_dataset_2` | 2026-08-28, 09-18, 09-23 | 66 / 18 | 5,871 total after forward-only (5,397 reverse frames removed) | 0 only (was 1 = reverse) | Forward/reverse driving on the small track. The 09-23 sessions were reverse-only and are empty after `make_forward_only.py`. Originals kept as `*.with_reverse`; `make_forward_only.py --restore` brings them back. |
| `train_data/indicator_training`, `test_data/indicator_training` | 2026-09-27 | 25 / 6 | 2,209 / 685 | 0 and 1 (right) | Indicator dataset 1. 2 train sessions have no labels and are skipped. |
| `train_data/indicator_data_session_2`, `test_data/indicator_data_session_2` | 2026-10-13 evening, 2026-10-14 morning | 32 / 8 | 3,231 / 806 | 0 and 1 (right) | Indicator dataset 2. Test sessions: `20261013_203838`, `_203927`, `_204019`, `20261014_071641`, `_071735`, `_071828`, `_071921`, `_071941`. |
| `_other_sessions/empty_sessions/20261013_203750` | 2026-10-13 | n/a | 39 frames, no controls | n/a | Empty `ctrlLog.txt`; matching raises on it. |
| `policy/dataset_reverse/` | built 2026-09-09 | n/a | n/a | n/a | Reverse-only copy built by `build_reverse_dataset.py` for the reverse test runs. |
| `fwdbwd/` session 1 | 2026-10-17, 10-19, daylight | 71 recordings, test = every 5th (14) | fwd 9,534 / 2,455, bwd 3,405 / 736 | fwd 0, bwd 2 | Autopilot Session data: forward and backward driving, split by `split_direction.py` (see `FWDBWD.md`). Was `fwdbwd_session_1`. 2 empty recordings of 10-17 (`123514`, `123631`) are in `_other_sessions/empty_sessions/`. |
| `fwdbwd/` session 2 | 2026-10-20 evening | 47 recordings, test = every 5th (9) | fwd 3,638 / 1,199, bwd 2,010 / 532 | fwd 0, bwd 2 | 28 recordings (`203857`-`204510`) under warm yellow lamps (red/blue ~7, half the brightness), 19 neutral. Was `fwdbwd_session_2`. |
| `fwdbwd/` session 3 | 2026-10-25 late morning | 46 recordings, test = random 20%, seed 0 (9) | fwd 2,831 frames in 32 runs (2,236 / 595), bwd 898 / 302 | fwd 0, bwd 2 | Neutral light. |
| `fwdbwd/` session 8 | 2026-11-13 evening | 2 takeover recordings (`autopilot_log.py extract` of runs `213611` + `213634`), no test | fwd 338 frames in 16 runs, bwd 239 | fwd -1/0/1, bwd 2 | Recovery data: the human takeovers while `fwd_seq_turn_loss_13_11_seq0-3-6` (rnn, turn loss) and `bwd_seq0-3-6` (rnn) drove, 32 takeovers, trimmed 0.3 s each end. Meant to be trained together with 4-7, not alone. |
| `fwdbwd/` session 9 | 2026-11-14 morning | 18 forward recordings, test = random 20%, seed 0 (4) | fwd 1,416 / 408, no bwd | fwd -1 (1,110), 0 (286), 1 (428) | Clean laps, mostly with the left indicator on. |
| `fwdbwd/` session 10 | 2026-11-14 morning | 10 takeover recordings (`autopilot_log.py extract` of runs `072248`-`072703`, `072308` empty and skipped), `--test_fraction 0` | fwd 285 frames in 12 runs, no bwd | fwd -1 (197), 1 (88) | Recovery data: 13 takeovers while `turnlost_fwd_seq0-3-6` (rnn turn loss, s4-8) and `rnn_seq0-3-6` (plain rnn, s4-7) drove. Train together with 4-9. |

Frames for fwdbwd are images per split folder; the memory records drop the first 3 frames of
every forward run. Records for sessions 1 + 2 + 3 (2026-10-25): fwd 14,892 / 4,111 examples,
bwd 6,313 / 1,570.

**There is no left-turn (cmd -1) data in any indicator set yet.**

### Test sets have changed over time

The metrics in each model's `logs/` were measured on whatever `test_data/` held when it was
trained, so they are only comparable within the same test set:

- **T1:** `my_dataset_2` (18 sessions), forward + reverse, before 2026-10-10.
- **T1-fwd:** `my_dataset_2`, forward-only (after 2026-10-10).
- **T2:** `indicator_training` test (6 sessions, 685 frames; 667 examples in the memory records).
- **T3:** T2 + `indicator_data_session_2` test (14 sessions, 1,491 frames; 1,449 memory examples).
- **F1 / F12 / F123:** the fwdbwd test splits of sessions 1, 1 + 2 and 1 + 2 + 3, for the
  fwd and the bwd model alike. A model's `data_manifest.txt` lists its exact sessions.

For a fair comparison, re-evaluate the `.tflite` files on one fixed test set; see "How to
compare" below.

## Loss versions

- **Until 2026-10-04:** `command_weight` (steering or throttle in the weight), an early workaround.
- **2026-10-04 to 2026-11-21 (`e03f2c11`):** a per-sample rewrite of OpenBot's loss,
  `(2|s| + 0.05)^2 * (dt^2 + 5 ds^2)` per frame. Straight frames counted ~1/1,700 of full
  lock for steering. Every turn-weighted model below from that window used it.
- **From 2026-11-21:** OpenBot's `sq_weighted_mse_angle` exactly, run on `(left, right)`
  converted from the Ackermann labels (same numbers and gradients as upstream's code). Its angle
  term is the batch mean, so steering on straights is trained much more than in the previous
  version. Loss values aren't comparable across versions; compare by metrics and on the robot.

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

### Autopilot Session models (fwd donkey_memory + bwd cil_mobile_fast)

Trained on the fwdbwd sessions. From 2026-10-25 on, `train_fwdbwd.py` names the folders
after the sessions (`fwd_s1-2-3_...`) and puts a `data_manifest.txt` in them. Epochs are
as in `logs/log.csv` (counting from 0).

| Folder | Date | Model and options | Trained on | Test set | Best val epoch | val angle<0.1 / dir / throttle<0.1 / MAE | On the robot |
|---|---|---|---|---|---|---|---|
| `openbot_donkey_memory_lr0.0003_bz128_mem3_color_fwdbwd1` | 10-20 | donkey_memory, mem 3, `--color_aug`, no cmd | session 1 fwd (9,225 / 2,377 examples) | F1 | 52 | 0.485 / 0.894 / 0.571 / 0.131 | Not recorded. Closed-loop steering MAE 0.187 vs 0.151 teacher-forced; overfits after ~50. |
| `openbot_cil_mobile_fast_lr0.0003_bz128_bn_flip_fwdbwd1` | 10-20 | cil_mobile_fast, BN, `--flip_aug` (cmd 2 kept on flips) | session 1 bwd (3,405 / 736) | F1 | 80 | 0.519 / 0.769 / 0.865 / 0.197 | Not recorded. |
| `openbot_donkey_memory_lr0.0003_bz128_mem3_color` | 10-21 | as `_fwdbwd1` | sessions 1 + 2 fwd (12,734 / 3,534) | F12 | 17 | 0.532 / 0.900 / 0.636 / 0.115 | Not recorded. |
| `openbot_cil_mobile_fast_lr0.0003_bz128_bn_flip` | 10-21 | as `_fwdbwd1` | sessions 1 + 2 bwd (5,415 / 1,268) | F12 | 65 | 0.516 / 0.728 / 0.862 / 0.209 | Not recorded. |

### fwd_s4-5-6-7-8-9-10_donkey_3d_lr0.0003_bz128_bn_flip_cmd_seq0-3-6, 2026-11-14
- Command: `python train_fwdbwd.py fwd --model donkey_3d --sessions 4 5 6 7 8 9 10 --flip_aug --cmd_aug`
- Git commit: 80f682d2, with uncommitted changes (donkey_3d itself is uncommitted)
- Model: donkeycar's Keras3D_CNN (3D convs written as exact 2D equivalents for TFLite 2.4)
  plus cil_mobile_fast's cmd head; plain MSE; BN always.
- Trained on: fwd sessions 4-10 (`tfrecords_seq0-3-6`, shared with donkey_rnn). Test: the 4,768
  seq test frames of those sessions.
- Best val epoch 46 of 67 (early stop): val_loss 0.0362 (rnn same data 0.0467), val angle 0.459,
  direction 0.896, throttle<0.1 0.464, MAE 0.141. `best.tflite` passes check_tflite.
- Offline vs `fwd_s4-5-6-7-8-9-10_donkey_rnn_..._flip_cmd_seq0-3-6` (same test frames):
  steering MAE 38 vs 44 /255, corr 0.90 vs 0.86. Phantom turns (straight frames steered >100)
  3/0/1% vs 10/2/15% for indicator left/off/right. Under-steers hard turns more: mean |steer|
  145 vs rnn 164 vs labels 196. Indicator pull -9/+11 vs -32/+36 (`--cmd_aug` weakened it).
- Replay of the rnn's 9 takeovers of 2026-11-14 (stacks from the logged crops; the rnn replay
  matched the phone within 2): the 3D model would not have saved them - near-zero steering
  where the user went to +/-220, wrong side on 2.
- Robot test: not yet.
- Conclusion / next step: first model that is clearly better than the rnn offline, and cures the
  phantom turns; the remaining fault is under-steering at hard turns/obstacles, which is what
  `--turn_loss` targets.

### fwd_s4-5-6-7_cil_mobile_fast_cat_lr0.0003_bz128_bn_flip, 2026-11-15 (turn-weighted loss)
- Command: `python train_fwdbwd.py fwd --model cil_mobile_fast_cat --sessions 4 5 6 7`
- Git commit: 80f682d2, with uncommitted changes (cil_mobile_fast_cat itself is uncommitted)
- Model: cil_mobile_fast with steering as 21 softmax bins (decoded in the graph: argmax bin
  and its neighbours) and a linear throttle. Loss: Gaussian soft target (sigma 1 bin),
  cross-entropy + 10 x throttle^2, **times cil's turn weight w^2** (the first version of the
  loss, `sq_weighted_ce_angle`, since replaced by the unweighted `ce_angle`).
- Trained on: fwd sessions 4-7 (`tfrecords_fwd`), the same records and test frames (4,360) as
  `cil_13_11`.
- Best val epoch 45 of 66 (early stop): val angle 0.178, direction 0.544, throttle<0.1 0.465,
  MAE 0.247 (cil_13_11: 0.269 / 0.641 / 0.398 / 0.195). Val loss rises from epoch 45 while the
  metrics keep improving, so best-val is worse than last (last: 0.203 / 0.563 / 0.483 / 0.211).
- Offline vs `cil_13_11` (same test frames): steering MAE 0.364 best-val / 0.289 last vs 0.237;
  straights 0.496 / 0.399 vs 0.294; turns 0.231 / 0.194 vs 0.192; wrong-way errors > 0.5
  22% / 11% vs 7%; throttle MAE 0.130 / 0.134 vs 0.152 (better).
- Why: it **never predicts the three middle bins (-0.1, 0, +0.1)**, though 39% of the test frames
  (1,704) are labelled straight. w^2 is 0.0025 straight vs up to 4.2 on a hard turn, so the
  straight bins got no signal; regression can still average to ~0, classes can't.
- Robot test: not done.
- Conclusion / next step: failed because of the weight, not the bins (turns ~equal to cil).
  Retrain with the unweighted `ce_angle`, as donkeycar trains its categorical model. Folder
  renamed to `..._flip_turnweight`.

### fwd_s4-5-6-7_cil_mobile_fast_cat_lr0.0003_bz128_bn_flip, 2026-11-15 (unweighted loss)
- Command: `python train_fwdbwd.py fwd --model cil_mobile_fast_cat --sessions 4 5 6 7`
- Git commit: 80f682d2, with uncommitted changes (cil_mobile_fast_cat itself is uncommitted)
- Model: as the run above; loss `ce_angle`: soft-target cross-entropy + 10 x throttle^2, no
  turn weight.
- Trained on: fwd sessions 4-7 (`tfrecords_fwd`), same 4,360 test frames as `cil_13_11`.
- Best val epoch 37 of 58 (early stop): val angle 0.457, direction 0.870, throttle<0.1 0.507,
  MAE 0.161 (cil_13_11: 0.269 / 0.641 / 0.398 / 0.195). Last: 0.430 / 0.826 / 0.471 / 0.165.
- Offline vs `cil_13_11` (best-val / last vs cil): steering MAE 0.196 / 0.201 vs 0.237;
  straights 0.143 / 0.171 vs 0.294; turns (|s| >= 0.3) 0.228 / 0.215 vs 0.192; hard turns
  (>= 0.7) 0.227 / 0.194 vs 0.175, mean |steer| there 0.73 / 0.79 vs 0.79 (labels 0.91);
  phantom turns (straight frames steered > 0.4) 16% / 15% vs 25%; wrong-way > 0.5 4.8% / 6.2%
  vs 6.6%; throttle MAE 0.126 / 0.128 vs 0.152. The predicted bins now follow the label
  histogram, straight bin included.
- Ties: ~40% of frames have their top two bins within 0.02, but almost all are neighbours
  (top two > 3 bins apart: 2.3% best-val, 0.7% last), so a flip moves steering ~0.1.
  check_tflite passes on all four files.
- Robot test 2026-11-15 (`cate_please` = last.tflite, Session logs `autopilot_20261115_*`,
  with the bwd cat below): 26.1 takeovers/min fwd over all 20 runs (15.8/min in the one long
  run, 2m18s; the 19 short runs each start just before the same left turn). Nearly every
  takeover is that left turn under-steered: model -0..-100 before you take over, you -180..-255.
  Replaying the logged crops reproduces the phone (mean diff 4/255), so no export bug. Where you
  steered > 150 (964 frames, you mean 249) the model said 113: its probability peaks at the
  moderate bins (+0.4..+0.6), only 0.39 on the hard ones - it has hardly seen full lock
  recoveries, which you need because it arrives late. No better than the rnn runs (18-25/min).
- Conclusion / next step: best offline forward model so far on these sessions, mainly from
  straights; hard turns still a little behind cil. Judge by a logged Session run vs
  `cil_13_11` (takeovers per minute).

### fwd_s4-5-6-7-8-10-11_cil_mobile_fast_cat_lr0.0003_bz128_bn_flip, 2026-11-15
- Command: `python train_fwdbwd.py fwd --model cil_mobile_fast_cat --sessions 4 5 6 7 8 10 11`
- Git commit: 80f682d2, with uncommitted changes
- Trained on: fwd sessions 4-7 plus the takeover sessions 8, 10, 11 (11 = takeovers of the
  cat models on 2026-11-15, `--test_fraction 0`; 1,180 takeover frames, 5% of the data). Same
  4,360 test frames as before (takeover sessions have no test).
- Early stop after 28 epochs, best val epoch 7: val angle 0.422, direction 0.991, throttle<0.1
  0.191, MAE 0.255. Train loss still falling; val loss (cross-entropy) bottoms early for this
  model, so the early stop cut it short.
- Offline vs s4-7 cat last / cil_13_11: best-val predicts ~straight everywhere (hard turns |s|
  0.27, turn MAE 0.553) - unusable. last: steering MAE 0.262 (turns 0.388 vs 0.215), throttle
  bias **+67/255** (s4-7 cat +4): the takeover sessions' throttle is median 219-238, often 255
  (you speed up when you take over), vs ~120 in the laps.
- Pre-takeover frames of 2026-11-15 (313 model-driven frames in the 0.6 s before a takeover
  where you steered > 120, not in training; you then steered 184 on average): model |steering|
  cil_13_11 163, cat s4-7 last (the one on the car) 82, this run's last 146.
- Robot test: not done; not worth it as is (best-val straight, last too fast).
- Conclusion / next step: train to the end (`--early_stop 0`), and keep the takeovers' throttle
  from teaching speed. cil_13_11 already steers that turn twice as hard as the cat model did.

### bwd_s1-2-3-4-5-6-7-8_cil_mobile_fast_cat_lr0.0003_bz128_bn_flip, 2026-11-15
- Command: `python train_fwdbwd.py bwd --model cil_mobile_fast_cat` (all sessions, 1-8; a first
  attempt died at epoch 1 on a cuBLAS init error, GPU memory held elsewhere; folder removed)
- Git commit: 80f682d2, with uncommitted changes (cil_mobile_fast_cat itself is uncommitted)
- Model: as the fwd cat run, unweighted `ce_angle`.
- Trained on: bwd sessions 1-8 (`tfrecords`, rebuilt: 3,411 test frames, 1,570 of them from
  sessions 1-3). Sessions 4-8 reverse slower than 1-3 (test throttle median -222 vs -255).
- Best val epoch 44 of 65 (early stop): val angle 0.541, direction 0.913, throttle<0.1 0.410,
  MAE 0.189. Last: 0.534 / 0.918 / 0.423 / 0.199.
- Offline vs `bwd_25_10` (cil, sessions 1-3, best-val) - **not the same training data**. On the
  s1-3 test frames (both trained on s1-3): steering MAE cat best-val 0.259 vs cil 0.350;
  straights 0.323 vs 0.907 (cil steers near full lock on most frames); turns 0.225 vs 0.118,
  hard turns 0.200 vs 0.064 (mean |steer| there 0.83 vs 0.97, labels 0.98); wrong-way 13.9% vs
  27.8%; throttle MAE 0.124 vs 0.086. On s4-8 (cil never saw them): steering MAE 0.198 vs 0.681,
  wrong-way 2.3% vs 41%. last under-steers hard turns more than best-val (0.76 vs 0.81).
- Ties: 23% of frames with top two bins within 0.02, 6.0% of frames with them > 3 bins apart
  (fwd: 2.3%), so more left/right jumps possible than forward.
- Robot test 2026-11-15 (`bwd_cate_please` = best-val): 19.8 takeovers/min over 30 s of model
  driving; under-steers like fwd (you 240, model 162 where you steered > 150), with 0.20 of its
  probability on the wrong side there.
- Conclusion / next step: far better on straights and on the newer sessions, under-steers hard
  reverse turns where cil (always near full lock) doesn't. A cil_mobile_fast trained on s1-8
  would separate the effect of the bins from that of the data.

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
- Command: python train_fwdbwd.py ... (or python -m openbot.train ...)
- Git commit: <hash> (and whether the working tree had uncommitted changes)
- Trained on: <dataset folders>, <train/test examples from the tfrecord log>
- Test set: T?
- Best val epoch and metrics (from logs/log.csv):
- Phone file name: <renamed .tflite>, which checkpoint it is (best-val / last)
- Offline comparison (same test set as the others):
- Robot test: track, lighting, attempts, result, notes or video link
- Conclusion / next step:
```
