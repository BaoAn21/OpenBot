"""The fwdbwd web app: the Autopilot Session data and training flow in the browser.

  /                 the app (openbot/server/fwdbwd_web/), /legacy/ the old trainer UI
  POST /upload      phone (and browser) uploads: recordings into dataset/fwdbwd/inbox/,
                    autopilot_*.zip into dataset/fwdbwd/autopilot_runs/ (or the old
                    dataset/uploaded/, a setting)
  /fb/api/...       JSON for the app
  /fb/file/...      files under dataset/, /fb/modelfile/... files under models/

Everything that changes data or trains runs the existing scripts (split_direction.py,
train_fwdbwd.py, train_both.py, autopilot_log.py, check_tflite.py) as jobs: separate
processes in their own session, logging to dataset/fwdbwd/.webapp/jobs/<id>.log, so a
server restart (adev reloads it on every save) never kills a training run, and the app
picks the job up again from jobs.json. Exclusive jobs (training, a real split) run one
at a time from a queue, as there is one GPU; dry runs, reports and checks start at once.
"""

import asyncio
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile
import zlib
from bisect import bisect_right

from aiohttp import web

from .. import base_dir, dataset_dir, models_dir

if base_dir not in sys.path:
    sys.path.insert(0, base_dir)  # the scripts in policy/ are imported as modules

import autopilot_log  # noqa: E402
import split_direction  # noqa: E402
import train_both  # noqa: E402
import train_fwdbwd  # noqa: E402

ROOT = os.path.join(dataset_dir, "fwdbwd")
INBOX = os.path.join(ROOT, "inbox")
RUNS = os.path.join(ROOT, autopilot_log.RUNS)
PARKED = os.path.join(ROOT, "parked")
STATE_DIR = os.path.join(ROOT, ".webapp")
JOBS_DIR = os.path.join(STATE_DIR, "jobs")
SETTINGS = os.path.join(STATE_DIR, "settings.json")
PUBLISHED = os.path.join(models_dir, ".published.json")
WEB_DIR = os.path.join(os.path.dirname(__file__), "fwdbwd_web")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# What train_fwdbwd.py / train_both.py train, per direction (pilot_net_seq isn't offered).
DIRECTIONS = {
    "fwd": train_fwdbwd.MODELS["fwd"],
    "bwd": train_fwdbwd.MODELS["bwd"],
    "both": train_both.MODELS,
}
MEMORY_MODELS = ("donkey_memory", "donkey_memory_reverse", "cil_mobile_fast_memory")
SEQ_MODELS = ("donkey_rnn", "donkey_3d")
DONKEY_MODELS = ("donkey_memory", "donkey_memory_reverse")
MSE_MODELS = DONKEY_MODELS + SEQ_MODELS
# train.py refuses --steer_only for memory models and cil_mobile_fast_cat.
STEER_ONLY_MODELS = ("cil_mobile_fast", "pilot_net") + SEQ_MODELS
# Trained on the turn-weighted loss, so --turn_floor applies (MSE_MODELS only with
# --turn_loss; cil_mobile_fast_cat is on ce_angle).
TURN_FLOOR = 0.05
# Opt-in openbot.train flags per model, as train.py accepts them; flags the script
# already passes for a model (MODELS[direction][model]) show as fixed instead.
ALL_MODELS = tuple(sorted({m for models in DIRECTIONS.values() for m in models}))
OPTIONS = {
    "flip_aug": ("mirror images and steering", DONKEY_MODELS + SEQ_MODELS),
    "tilt_aug": ("rotate images up to ±5° (camera roll, not up/down pitch)", ALL_MODELS),
    "cmd_aug": ("noise on the indicator (cmd) input", tuple(m for m in ALL_MODELS if m not in DONKEY_MODELS)),
    "mem_aug": ("make the memory unreliable (zeroed or offset)", MEMORY_MODELS),
    "turn_loss": ("turn-weighted loss instead of plain MSE", MSE_MODELS),
    "mem_cmd": ("feed the indicator to the memory model", DONKEY_MODELS),
    "steer_only": ("train steering only; drive it with Fixed or Manual throttle", STEER_ONLY_MODELS),
}
# Which of the flags above (and of the scripts' fixed flags) are augmentations.
AUGMENTATIONS = ("flip_aug", "color_aug", "tilt_aug", "cmd_aug", "mem_aug")
COMMON = {"num_epochs": 100, "early_stop": 20, "batch_size": 128, "learning_rate": 0.0003}
KNOWN_MODELS = sorted(
    {m for models in DIRECTIONS.values() for m in models} | {"pilot_net_seq", "cil_mobile"},
    key=len,
    reverse=True,
)
PLOTS = (
    "eval_rollout.png",
    "eval_scatter.png",
    "loss.png",
    "eval_samples.png",
    "test_preview.png",
    "angle.png",
    "direction.png",
    "error.png",
)


# ---------------------------------------------------------------- small helpers


def rel_dataset(path):
    return os.path.relpath(path, dataset_dir)


def safe_join(root, rel):
    """root/rel, refusing anything that leaves root."""
    real = os.path.normpath(os.path.join(root, rel))
    if real != root and not real.startswith(root + os.sep):
        raise web.HTTPForbidden()
    return real


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return default


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def settings():
    return {"upload_to": "inbox", **read_json(SETTINGS, {})}


def csv_rows(path):
    """Data rows of a logger file (header skipped), [] if it's missing."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [r for r in reader if r]


def listdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        d
        for d in os.listdir(path)
        if not d.startswith(".") and os.path.isdir(os.path.join(path, d))
    )


def count_images(folder):
    images = os.path.join(folder, "images")
    return len(os.listdir(images)) if os.path.isdir(images) else 0


def raw_recordings():
    """Recording name -> session number, for every recording already split."""
    out = {}
    raw = os.path.join(ROOT, "raw")
    for s in listdirs(raw):
        if s.startswith("session_"):
            for r in listdirs(os.path.join(raw, s)):
                out[r] = int(s.split("_")[1])
    return out


# ---------------------------------------------------------------- recordings


def frame_images(rec):
    """Frame number -> image file name (the logger's <frame>_crop.jpeg)."""
    out = {}
    images = os.path.join(rec, "images")
    if os.path.isdir(images):
        for name in os.listdir(images):
            head = name.split("_")[0].split(".")[0]
            if head.isdigit() and (int(head) not in out or "crop" in name):
                out[int(head)] = name
    return out


def recording_frames(rec):
    """Per frame, columnar: time (s), image, steering, throttle, cmd; for an Autopilot
    Session log also who drove and the model's prediction. Commands are paired the data
    logger's way, the last one sent before the frame (display only: the record builder
    does its own matching)."""
    images = frame_images(rec)
    sensor = os.path.join(rec, "sensor_data")
    indicator = sorted(
        (int(r[0]), int(float(r[1])))
        for r in csv_rows(os.path.join(sensor, "indicatorLog.txt"))
        if len(r) >= 2
    )
    ind_ts = [r[0] for r in indicator]

    def cmd_at(ts):
        i = bisect_right(ind_ts, ts)
        return indicator[i - 1][1] if i else 0

    out = {k: [] for k in ("t", "img", "steering", "throttle", "cmd")}
    log = os.path.join(sensor, autopilot_log.LOG)
    if os.path.isfile(log):
        rows = [r for r in autopilot_log.read_log(rec) if r.frame in images]
        for k in ("source", "pred_s", "pred_t", "model", "direction"):
            out[k] = []
        t0 = rows[0].ts if rows else 0
        for r in rows:
            out["t"].append(round((r.ts - t0) / 1e9, 3))
            out["img"].append(images[r.frame])
            out["steering"].append(r.s)
            out["throttle"].append(r.t)
            out["cmd"].append(2 if r.direction == "bwd" else cmd_at(r.ts))
            out["source"].append(r.source)
            out["pred_s"].append(r.pred_s)
            out["pred_t"].append(r.pred_t)
            out["model"].append(r.model)
            out["direction"].append(r.direction)
        return out

    frames = sorted(
        (int(r[0]), int(r[1]))
        for r in csv_rows(os.path.join(sensor, "rgbFrames.txt"))
        if len(r) >= 2 and int(r[1]) in images
    )
    ctrl = sorted(
        (int(r[0]), float(r[1]), float(r[2]))
        for r in csv_rows(os.path.join(sensor, "ctrlLog.txt"))
        if len(r) >= 3
    )
    ctrl_ts = [c[0] for c in ctrl]
    t0 = frames[0][0] if frames else 0
    for ts, frame in frames:
        i = bisect_right(ctrl_ts, ts)
        s, t = (ctrl[i - 1][1], ctrl[i - 1][2]) if i else (0.0, 0.0)
        out["t"].append(round((ts - t0) / 1e9, 3))
        out["img"].append(images[frame])
        out["steering"].append(s)
        out["throttle"].append(t)
        out["cmd"].append(cmd_at(ts))
    return out


_summary_cache = {}


def recording_summary(rec):
    """What a recording list shows; cached until its files change."""
    sensor = os.path.join(rec, "sensor_data")
    stamp = tuple(
        os.path.getmtime(p) if os.path.exists(p) else 0
        for p in (rec, sensor, os.path.join(rec, "images"))
    )
    hit = _summary_cache.get(rec)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        fr = recording_frames(rec)
        error = None
    except Exception as e:  # a half-written or odd folder must not break the list
        fr, error = {"t": [], "img": [], "throttle": []}, str(e)
    n = len(fr["t"])
    secs = {"fwd": 0.0, "bwd": 0.0, "stop": 0.0}
    for i in range(n - 1):
        dt = min(fr["t"][i + 1] - fr["t"][i], split_direction.MAX_GAP_S)
        t = fr["throttle"][i]
        secs["fwd" if t > 0 else "bwd" if t < 0 else "stop"] += dt
    try:
        reason = split_direction.unusable_reason(rec)
    except Exception as e:
        reason = f"unreadable: {e}"
    summary = {
        "name": os.path.basename(rec),
        "path": rel_dataset(rec),
        "frames": n,
        "seconds": round(fr["t"][-1], 1) if n else 0,
        "fwd_s": round(secs["fwd"], 1),
        "bwd_s": round(secs["bwd"], 1),
        "stop_s": round(secs["stop"], 1),
        "thumb": rel_dataset(os.path.join(rec, "images", fr["img"][n // 2])) if n else None,
        "reason": reason,
        "autopilot": os.path.isfile(os.path.join(sensor, autopilot_log.LOG)),
        "error": error,
    }
    _summary_cache[rec] = (stamp, summary)
    return summary


def inbox_state():
    taken = raw_recordings()
    recordings = []
    for name in listdirs(INBOX):
        s = dict(recording_summary(os.path.join(INBOX, name)))
        s["duplicate_of"] = taken.get(name)
        recordings.append(s)
    zips = sorted(n for n in os.listdir(INBOX) if n.endswith(".zip")) if os.path.isdir(INBOX) else []
    parked = []
    for reason in listdirs(PARKED):
        for name in listdirs(os.path.join(PARKED, reason)):
            parked.append({"reason": reason, "name": name, "path": rel_dataset(os.path.join(PARKED, reason, name))})
    sessions = sorted(set(train_fwdbwd.sessions_in(ROOT, "fwd")) | set(train_fwdbwd.sessions_in(ROOT, "bwd")) | set(taken.values()))
    return {
        "recordings": recordings,
        "zips": zips,
        "parked": parked,
        "sessions": sessions,
        "next_session": max(sessions, default=0) + 1,
        "uploads": list(reversed(upload_events[-30:])),
        "settings": settings(),
    }


# ---------------------------------------------------------------- sessions


def sessions_state():
    raw = os.path.join(ROOT, "raw")
    numbers = sorted(
        set(train_fwdbwd.sessions_in(ROOT, "fwd"))
        | set(train_fwdbwd.sessions_in(ROOT, "bwd"))
        | {int(s.split("_")[1]) for s in listdirs(raw) if s.startswith("session_")}
    )
    out = []
    for n in numbers:
        sdir = os.path.join(raw, f"session_{n}")
        note_path = os.path.join(sdir, "note.txt")
        note = open(note_path).read().strip() if os.path.isfile(note_path) else ""
        test_path = os.path.join(sdir, "test_recordings.txt")
        test = set(open(test_path).read().split()) if os.path.isfile(test_path) else set()
        counts = {}
        parts = {}
        for split, _, _ in train_fwdbwd.SPLITS:
            for direction in ("fwd", "bwd"):
                total = 0
                for f in train_fwdbwd.folders(ROOT, split, direction, n):
                    path = os.path.join(ROOT, split, direction, f"session_{n}", f)
                    frames = count_images(path)
                    total += frames
                    rec = re.sub(r"_f\d+$", "", f) if direction == "fwd" else f
                    parts.setdefault(rec, []).append(
                        {"split": split, "direction": direction, "name": f, "path": rel_dataset(path), "frames": frames}
                    )
                counts[f"{direction}_{split}"] = total
        recordings = [
            {
                "name": r,
                "path": rel_dataset(os.path.join(sdir, r)),
                "split": "test" if r in test else "train",
                "parts": parts.get(r, []),
            }
            for r in listdirs(sdir)
        ]
        out.append({"number": n, "note": note, "counts": counts, "recordings": recordings, "test": len(test)})
    return out


# ---------------------------------------------------------------- models


def parse_model_name(name):
    direction = next((d for d in ("fwd", "bwd", "both") if name.startswith(d + "_")), "")
    model = next((m for m in KNOWN_MODELS if f"_{m}_lr" in name or name.startswith(m + "_lr")), "")
    sessions = re.search(r"_s(\d+(?:-\d+)*)_", name)
    phone = "CMDNAV_SEQ" if re.search(r"_seq\d", name) else "CMDNAV_MEM" if re.search(r"_mem\d", name) else "CMDNAV"
    return {
        "direction": direction,
        "model": model,
        "sessions": [int(s) for s in sessions[1].split("-")] if sessions else [],
        "phone_type": phone,
        "seq_tag": (re.search(r"_seq\d+(?:-\d+)*", name) or [None])[0],
        "steer_only": "_steeronly" in name,
    }


def read_log_csv(folder):
    path = os.path.join(folder, "logs", "log.csv")
    if not os.path.isfile(path):
        return [], []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header = rows[0]
    data = []
    for r in rows[1:]:
        try:
            data.append([float(v) for v in r])
        except ValueError:
            continue
    return header, data


_model_cache = {}


_hash_cache = {}


def file_hash(path):
    st = os.stat(path)
    key = (st.st_mtime, st.st_size)
    hit = _hash_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    with open(path, "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()
    _hash_cache[path] = (key, digest)
    return digest


def guess_origin(path):
    """For a .tflite copied into models/ by hand: the checkpoint with the same bytes."""
    size = os.path.getsize(path)
    for name in listdirs(models_dir):
        ckpt = os.path.join(models_dir, name, "checkpoints")
        if not os.path.isdir(ckpt):
            continue
        # best.tflite before best-val.tflite, its copy
        for f in sorted((f for f in os.listdir(ckpt) if f.endswith(".tflite")), key=lambda f: (f != "best.tflite", f)):
            p = os.path.join(ckpt, f)
            if os.path.getsize(p) == size and file_hash(p) == file_hash(path):
                return {"model": name, "checkpoint": f, "guessed": True}
    return {}


def published():
    """models/*.tflite (what the phone downloads), with where each came from."""
    info = read_json(PUBLISHED, {})
    out = []
    for p in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, p)
        if p.endswith(".tflite") and os.path.isfile(path):
            name = p[: -len(".tflite")]
            out.append({"name": name, "file": p, "mtime": os.path.getmtime(path), **(info.get(name) or guess_origin(path))})
    return out


def model_summary(name):
    folder = os.path.join(models_dir, name)
    log = os.path.join(folder, "logs", "log.csv")
    ckpt = os.path.join(folder, "checkpoints")
    # checkpoints/ too: the .tflite files are exported after the last log.csv line.
    stamp = (
        os.path.getmtime(folder),
        os.path.getmtime(log) if os.path.isfile(log) else 0,
        os.path.getmtime(ckpt) if os.path.isdir(ckpt) else 0,
    )
    hit = _model_cache.get(name)
    if hit and hit[0] == stamp:
        return hit[1]
    header, data = read_log_csv(folder)
    best = last = None
    if data and "val_loss" in header:
        vl = header.index("val_loss")
        best = dict(zip(header, min(data, key=lambda r: r[vl])))
        last = dict(zip(header, data[-1]))
    ckpt = os.path.join(folder, "checkpoints")
    s = {
        "name": name,
        "mtime": stamp[1] or stamp[0],
        **parse_model_name(name),
        "epochs": len(data),
        "best": best,
        "last": last,
        "tflite": sorted(f for f in os.listdir(ckpt) if f.endswith(".tflite")) if os.path.isdir(ckpt) else [],
        "has_manifest": os.path.isfile(os.path.join(folder, "data_manifest.txt")),
    }
    _model_cache[name] = (stamp, s)
    return s


def models_state():
    out = []
    for name in listdirs(models_dir):
        if os.path.isdir(os.path.join(models_dir, name, "checkpoints")) or os.path.isdir(
            os.path.join(models_dir, name, "logs")
        ):
            out.append(model_summary(name))
    out.sort(key=lambda m: m["mtime"], reverse=True)
    pubs = published()
    by_model = {}
    for p in pubs:
        if p.get("model"):
            by_model.setdefault(p["model"], []).append(p["name"])
    for m in out:
        m["published"] = by_model.get(m["name"], [])
    return {"models": out, "published": pubs}


def model_detail(name):
    folder = safe_join(models_dir, name)
    if not os.path.isdir(folder):
        raise web.HTTPNotFound()
    header, data = read_log_csv(folder)
    manifest = os.path.join(folder, "data_manifest.txt")
    notes = os.path.join(folder, "notes.txt")
    logs = os.path.join(folder, "logs")
    plots = [p for p in PLOTS if os.path.isfile(os.path.join(logs, p))]
    plots += sorted(
        p for p in (os.listdir(logs) if os.path.isdir(logs) else []) if p.endswith(".png") and p not in plots
    )
    return {
        **model_summary(name),
        "header": header,
        "rows": data,
        "manifest": open(manifest).read() if os.path.isfile(manifest) else "",
        "notes": open(notes).read() if os.path.isfile(notes) else "",
        "plots": plots,
        "published": [p for p in published() if p.get("model") == name],
    }


# ---------------------------------------------------------------- autopilot runs


def runs_state():
    runs = []
    zips = []
    for entry in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        path = os.path.join(RUNS, entry)
        if entry.endswith(".zip"):
            if not os.path.isdir(path[: -len(".zip")]):
                zips.append(entry)
            continue
        if not os.path.isfile(os.path.join(path, "sensor_data", autopilot_log.LOG)):
            continue
        runs.append(run_summary(path))
    return {"runs": list(reversed(runs)), "zips": zips}


_run_cache = {}


def run_summary(path):
    log = os.path.join(path, "sensor_data", autopilot_log.LOG)
    report = os.path.join(path, autopilot_log.PLOT)
    stamp = (os.path.getmtime(log), os.path.getmtime(report) if os.path.isfile(report) else 0)
    hit = _run_cache.get(path)
    if hit and hit[0] == stamp:
        s = dict(hit[1])
    else:
        rows = autopilot_log.read_log(path)
        stats = autopilot_log.model_stats(rows)
        per_model = []
        model_s = human_s = 0.0
        takeovers = 0
        for (direction, model), (ms, hs, n, *_rest) in sorted(stats.items()):
            per_model.append(
                {
                    "direction": direction,
                    "model": model,
                    "model_s": round(ms, 1),
                    "human_s": round(hs, 1),
                    "takeovers": n,
                    "per_min": round(n / (ms / 60), 1) if ms else None,
                }
            )
            model_s += ms
            human_s += hs
            takeovers += n
        s = {
            "name": os.path.basename(path),
            "path": rel_dataset(path),
            "seconds": round((rows[-1].ts - rows[0].ts) / 1e9, 1) if rows else 0,
            "frames": len(rows),
            "model_s": round(model_s, 1),
            "human_s": round(human_s, 1),
            "takeovers": takeovers,
            "per_min": round(takeovers / (model_s / 60), 1) if model_s else None,
            "per_model": per_model,
            "report": rel_dataset(report) if os.path.isfile(report) else None,
        }
        _run_cache[path] = (stamp, s)
        s = dict(s)
    taken = autopilot_log.extracted_already(ROOT, s["name"] + autopilot_log.SUFFIX)
    s["extracted"] = rel_dataset(taken) if taken else None
    return s


# ---------------------------------------------------------------- uploads

upload_events = []


def unpack_zip(zip_path, folder):
    """The phone zips a folder's contents; split_direction.unpack handles both layouts."""
    split_direction.unpack(zip_path, folder)


async def handle_upload(request):
    """The phone's Data Logger posts each recording's zip here when it stops logging; the
    browser posts here too. The phone deletes its copy on a 200, so answer 200 only once
    the recording is safely unpacked."""
    if settings()["upload_to"] == "legacy":
        from .api import handle_upload as legacy_upload

        return await legacy_upload(request)
    reader = await request.multipart()
    field = None
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            field = part
            break
    if field is None or not field.filename:
        return web.json_response({"error": "no file field"}, status=400)
    filename = os.path.basename(field.filename)
    if not filename.endswith(".zip") or not NAME_RE.match(filename):
        return web.json_response({"error": f"not a recording zip: {filename}"}, status=400)
    name = filename[: -len(".zip")]
    dest_dir = RUNS if name.startswith("autopilot_") else INBOX
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(dest_dir, f".upload_{name}_{os.getpid()}_{time.time_ns()}.zip")
    size = 0
    digest = hashlib.sha1()
    with open(tmp, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            size += f.write(chunk)
            digest.update(chunk)
    event = {"name": name, "to": rel_dataset(dest_dir), "size": size, "time": time.time(), "source": request.remote}
    folder = os.path.join(dest_dir, name)
    taken = raw_recordings().get(name)
    try:
        if os.path.exists(folder) or taken:
            # The phone retries when an answer got lost; the recording is already here.
            event["status"] = "already here" + (f" (session {taken})" if taken else "")
        else:
            await asyncio.to_thread(unpack_zip, tmp, folder)
            event["status"] = "ok"
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as e:
        event["status"] = f"broken: {e}"
        upload_events.append(event)
        return web.json_response({"error": event["status"]}, status=400)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    upload_events.append(event)
    del upload_events[:-200]
    return web.json_response({"filename": filename, "size": size, "hash": digest.hexdigest()})


# ---------------------------------------------------------------- jobs

# Runs argv[2:], then writes its exit code to argv[1]. Stop signals go to the whole
# process group; the wrapper ignores them (after starting the child, which keeps the
# default handlers) and waits for the child to end.
JOB_WRAPPER = """
import signal, subprocess, sys
p = subprocess.Popen(sys.argv[2:])
for s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(s, signal.SIG_IGN)
code = p.wait()
with open(sys.argv[1], "w") as f:
    f.write(str(code))
sys.exit(code if code >= 0 else 1)
"""


class Jobs:
    """Script runs as detached processes; state in jobs.json, output in <id>.log."""

    def __init__(self):
        os.makedirs(JOBS_DIR, exist_ok=True)
        self.path = os.path.join(JOBS_DIR, "jobs.json")
        self.jobs = read_json(self.path, [])
        self.procs = {}
        for j in self.jobs:
            if j["status"] == "running" and not self._alive(j):
                self._finish(j)

    def save(self):
        write_json(self.path, self.jobs[-300:])

    def get(self, job_id):
        return next((j for j in self.jobs if j["id"] == job_id), None)

    def log_path(self, j):
        return os.path.join(JOBS_DIR, f"{j['id']}.log")

    def submit(self, kind, label, cmd, exclusive, meta=None, gpu=False):
        j = {
            "id": time.strftime("%Y%m%d_%H%M%S") + f"_{len(self.jobs) % 1000:03d}",
            "kind": kind,
            "label": label,
            "cmd": cmd,
            "exclusive": exclusive,
            "gpu": gpu,
            "status": "queued",
            "created": time.time(),
            "started": None,
            "ended": None,
            "pid": None,
            "returncode": None,
            "meta": meta or {},
        }
        while self.get(j["id"]):
            j["id"] += "x"
        self.jobs.append(j)
        self.tick()
        return j

    def _exit_path(self, j):
        return os.path.join(JOBS_DIR, f"{j['id']}.exit")

    def _start(self, j):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if not j["gpu"]:
            env["CUDA_VISIBLE_DEVICES"] = "-1"  # checks and reports leave the GPU to training
        log = open(self.log_path(j), "ab")
        log.write(f"$ {shlex.join(j['cmd'])}\n".encode())
        log.flush()
        # Not /bin/sh: the conda env's libraries break it. The wrapper writes the exit
        # code to a file, so a restarted server still learns how the job ended.
        p = subprocess.Popen(
            [sys.executable, "-c", JOB_WRAPPER, self._exit_path(j), *j["cmd"]],
            cwd=base_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,  # outlives a server restart; its own process group
        )
        log.close()
        self.procs[j["id"]] = p
        j.update(status="running", started=time.time(), pid=p.pid)

    def _alive(self, j):
        p = self.procs.get(j["id"])
        if p is not None:
            return p.poll() is None
        if not j.get("pid"):
            return False
        try:
            os.kill(j["pid"], 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return not os.path.isfile(self._exit_path(j))

    def _finish(self, j):
        try:
            code = int(open(self._exit_path(j)).read().strip())
        except (FileNotFoundError, ValueError):
            code = None
        self.procs.pop(j["id"], None)
        j["returncode"] = code
        j["ended"] = time.time()
        if j.get("stop_requested"):
            j["status"] = "stopped"
        else:
            j["status"] = "done" if code == 0 else "failed"

    def tick(self):
        changed = False
        for j in self.jobs:
            if j["status"] == "running" and not self._alive(j):
                self._finish(j)
                changed = True
            if j["status"] == "running" and j["kind"] == "train" and not j["meta"].get("model"):
                found = find_model_folder(j)
                if found:
                    j["meta"]["model"] = found
                    changed = True
        busy = any(j["status"] == "running" and j["exclusive"] for j in self.jobs)
        for j in self.jobs:
            if j["status"] != "queued":
                continue
            if not j["exclusive"]:
                self._start(j)
                changed = True
            elif not busy:
                self._start(j)
                busy = changed = True
        if changed:
            self.save()
        return changed

    def stop(self, j):
        if j["status"] == "queued":
            j.update(status="stopped", ended=time.time())
        elif j["status"] == "running":
            j["stop_requested"] = True
            # SIGINT first: Keras stops between batches and the checkpoints stay whole.
            self._signal(j, signal.SIGINT)
            asyncio.get_event_loop().call_later(15, self._signal, j, signal.SIGTERM)
            asyncio.get_event_loop().call_later(40, self._signal, j, signal.SIGKILL)
        self.save()

    def _signal(self, j, sig):
        if j["status"] == "running" and j.get("pid"):
            try:
                os.killpg(j["pid"], sig)
            except (ProcessLookupError, PermissionError):
                pass

    def read_log(self, j, offset, limit=256 * 1024):
        path = self.log_path(j)
        if not os.path.isfile(path):
            return "", 0
        size = os.path.getsize(path)
        if offset < 0:  # the tail
            offset = max(0, size + offset)
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(limit)
        return data.decode("utf-8", "replace"), offset + len(data)


def find_model_folder(j):
    """The models/ folder a training job writes: the newest one with its run name and
    model, made after the job started."""
    prefix = j["meta"].get("run_name")
    model = j["meta"].get("model_type")
    best = None
    for name in listdirs(models_dir):
        if not name.startswith(f"{prefix}_{model}_lr"):
            continue
        folder = os.path.join(models_dir, name)
        log = os.path.join(folder, "logs", "log.csv")
        # A --resume run writes into an old folder: its log.csv still moves every epoch.
        mtime = max(os.path.getmtime(folder), os.path.getmtime(log) if os.path.isfile(log) else 0)
        if mtime >= (j["started"] or 0) - 5 and (best is None or mtime > best[0]):
            best = (mtime, name)
    return best[1] if best else None


jobs = None


def job_view(j):
    view = {k: v for k, v in j.items() if k != "cmd"}
    view["command"] = shlex.join(j["cmd"])
    if j["kind"] == "train" and j["meta"].get("model"):
        view["progress"] = model_summary(j["meta"]["model"]) if os.path.isdir(
            os.path.join(models_dir, j["meta"]["model"])
        ) else None
    return view


def python(script, *args):
    return [sys.executable, script, *[str(a) for a in args]]


# ---------------------------------------------------------------- training commands


def train_options():
    sessions = sessions_state()
    out = {}
    for direction, models in DIRECTIONS.items():
        out[direction] = {
            m: {
                "fixed": [a.lstrip("-") for a in flags],
                "options": {k: v[0] for k, v in OPTIONS.items() if m in v[1] and f"--{k}" not in flags},
                "augmentations": list(AUGMENTATIONS),
                # train.py jitters colour for every model but the donkey_memory ones by itself
                "builtin_color": m not in DONKEY_MODELS,
                "memory": m in MEMORY_MODELS,
                "seq": m in SEQ_MODELS,
                # "turn_loss": only once --turn_loss is ticked
                "turn_floor": False if m == "cil_mobile_fast_cat" else "turn_loss" if m in MSE_MODELS else True,
            }
            for m, flags in models.items()
            if m != "pilot_net_seq"
        }
    return {
        "directions": out,
        "common": COMMON,
        "turn_floor": TURN_FLOOR,
        "sessions": [{"number": s["number"], "note": s["note"], "counts": s["counts"]} for s in sessions],
    }


def train_command(p):
    direction = p.get("direction")
    if direction not in DIRECTIONS:
        raise web.HTTPBadRequest(text=f"direction must be one of {list(DIRECTIONS)}")
    model = p.get("model")
    if model not in DIRECTIONS[direction]:
        raise web.HTTPBadRequest(text=f"{direction} doesn't train {model}")
    sessions = sorted({int(s) for s in p.get("sessions") or []})
    if not sessions:
        raise web.HTTPBadRequest(text="pick at least one session")
    if direction == "both":
        cmd = python("train_both.py")
    else:
        cmd = python("train_fwdbwd.py", direction)
    cmd += ["--sessions", *map(str, sessions), "--model", model]
    fixed = DIRECTIONS[direction][model]
    for opt in p.get("options") or []:
        if opt not in OPTIONS or model not in OPTIONS[opt][1] or f"--{opt}" in fixed:
            raise web.HTTPBadRequest(text=f"--{opt} doesn't apply to {model}")
        cmd.append(f"--{opt}")
    if model in MEMORY_MODELS and p.get("mem_len") and int(p["mem_len"]) != 3:
        cmd += ["--mem_len", str(int(p["mem_len"]))]
    if model in SEQ_MODELS and p.get("seq_offsets") and p["seq_offsets"].replace(" ", "") != "0,3,6":
        offsets = p["seq_offsets"].replace(" ", "")
        if not re.fullmatch(r"\d+(,\d+)*", offsets):
            raise web.HTTPBadRequest(text="seq offsets look like 0,3,6")
        cmd += ["--seq_offsets", offsets]
    if p.get("turn_floor") not in (None, "") and float(p["turn_floor"]) != TURN_FLOOR:
        floor = float(p["turn_floor"])
        if model == "cil_mobile_fast_cat" or (model in MSE_MODELS and "--turn_loss" not in cmd):
            raise web.HTTPBadRequest(
                text="the turn floor needs the turn-weighted loss: "
                + ("tick --turn_loss" if model in MSE_MODELS else f"{model} doesn't use it")
            )
        if floor <= 0:
            raise web.HTTPBadRequest(text="turn floor must be > 0")
        cmd += ["--turn_floor", f"{floor:g}"]
    for key, default in COMMON.items():
        value = p.get(key)
        if value in (None, ""):
            continue
        value = float(value) if key == "learning_rate" else int(value)
        if value != default:
            cmd += [f"--{key}", str(value)]
    run_name = (p.get("run_name") or "").strip()
    if run_name:
        if not NAME_RE.match(run_name):
            raise web.HTTPBadRequest(text="run name: letters, digits, _ . - only")
        cmd += ["--run_name", run_name]
    else:
        run_name = f"{direction}_s" + "-".join(map(str, sessions))
    if p.get("resume"):
        cmd.append("--resume")
    label = f"{direction} {model} s" + "-".join(map(str, sessions))
    return cmd, label, {"run_name": run_name, "model_type": model, "direction": direction, "sessions": sessions}


# ---------------------------------------------------------------- gpu

_gpu = {"time": 0, "data": None}


def gpu_state():
    if time.time() - _gpu["time"] < 2:
        return _gpu["data"]
    data = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()
        if out:
            name, used, total, util = [v.strip() for v in out[0].split(",")]
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip().splitlines()
            data = {
                "name": name,
                "used_mb": int(used),
                "total_mb": int(total),
                "util": int(util),
                "processes": [
                    dict(zip(("pid", "name", "used_mb"), [v.strip() for v in a.split(",")])) for a in apps if a.strip()
                ],
            }
    except (OSError, subprocess.SubprocessError, ValueError):
        data = None
    _gpu.update(time=time.time(), data=data)
    return data


# ---------------------------------------------------------------- handlers


async def body(request):
    try:
        return await request.json()
    except ValueError:
        return {}


def ok(data=None):
    return web.json_response(data if data is not None else {"ok": True}, dumps=lambda o: json.dumps(o, default=str))


async def api_inbox(request):
    return ok(await asyncio.to_thread(inbox_state))


async def api_recording(request):
    path = safe_join(dataset_dir, request.query.get("path", ""))
    if not os.path.isdir(os.path.join(path, "sensor_data")):
        raise web.HTTPNotFound(text="not a recording")
    frames = await asyncio.to_thread(recording_frames, path)
    summary = await asyncio.to_thread(recording_summary, path)
    return ok({"path": rel_dataset(path), "summary": summary, "frames": frames})


async def api_inbox_remove(request):
    p = await body(request)
    src = safe_join(INBOX, p.get("name", ""))
    if src == INBOX or not os.path.exists(src):
        raise web.HTTPNotFound()
    dst_dir = os.path.join(PARKED, "removed")
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(os.path.join(dst_dir, os.path.basename(src))):
        raise web.HTTPConflict(text="parked/removed already has one of that name")
    shutil.move(src, dst_dir)
    return ok()


async def api_parked_restore(request):
    p = await body(request)
    src = safe_join(PARKED, os.path.join(p.get("reason", ""), p.get("name", "")))
    if not os.path.isdir(src) or os.path.dirname(os.path.dirname(src)) != PARKED:
        raise web.HTTPNotFound()
    if os.path.exists(os.path.join(INBOX, os.path.basename(src))):
        raise web.HTTPConflict(text="the inbox already has one of that name")
    os.makedirs(INBOX, exist_ok=True)
    shutil.move(src, INBOX)
    return ok()


async def api_inbox_unpack(request):
    def run():
        split_direction.unpack_inbox(INBOX)

    try:
        await asyncio.to_thread(run)
    except SystemExit as e:
        raise web.HTTPConflict(text=str(e))
    return ok()


async def api_settings(request):
    if request.method == "POST":
        p = await body(request)
        s = settings()
        if p.get("upload_to") in ("inbox", "legacy"):
            s["upload_to"] = p["upload_to"]
        write_json(SETTINGS, s)
    return ok(settings())


async def api_split(request):
    p = await body(request)
    dry = bool(p.get("dry_run"))
    cmd = python("split_direction.py")
    if p.get("session"):
        cmd += ["--session", int(p["session"])]
    if p.get("note") and not dry:
        cmd += ["--note", str(p["note"])]
    frac = p.get("test_fraction")
    if frac not in (None, "") and float(frac) != 0.2:
        if not 0 <= float(frac) < 1:
            raise web.HTTPBadRequest(text="test fraction is between 0 and 1")
        cmd += ["--test_fraction", float(frac)]
    if dry:
        cmd.append("--dry_run")
    cmd = [str(c) for c in cmd]
    target = f"session {p['session']}" if p.get("session") else "a new session"
    j = jobs.submit("split", ("preview: " if dry else "") + f"inbox -> {target}", cmd, exclusive=not dry)
    return ok(job_view(j))


async def api_sessions(request):
    return ok(await asyncio.to_thread(sessions_state))


async def api_session_note(request):
    n = int(request.match_info["n"])
    sdir = os.path.join(ROOT, "raw", f"session_{n}")
    if not os.path.isdir(sdir):
        raise web.HTTPNotFound()
    p = await body(request)
    with open(os.path.join(sdir, "note.txt"), "w") as f:
        f.write(str(p.get("note", "")).strip() + "\n")
    return ok()


async def api_train_options(request):
    return ok(await asyncio.to_thread(train_options))


async def api_train(request):
    p = await body(request)
    cmd, label, meta = train_command(p)
    if p.get("preview"):
        return ok({"command": shlex.join(cmd)})
    j = jobs.submit("train", label, cmd, exclusive=True, meta=meta, gpu=True)
    return ok(job_view(j))


async def api_jobs(request):
    jobs.tick()
    return ok([job_view(j) for j in reversed(jobs.jobs[-100:])])


async def api_job_log(request):
    j = jobs.get(request.match_info["id"])
    if not j:
        raise web.HTTPNotFound()
    text, offset = jobs.read_log(j, int(request.query.get("offset", 0)))
    return ok({"text": text, "offset": offset, "job": job_view(j)})


async def api_job_stop(request):
    j = jobs.get(request.match_info["id"])
    if not j:
        raise web.HTTPNotFound()
    jobs.stop(j)
    return ok(job_view(j))


async def api_models(request):
    return ok(await asyncio.to_thread(models_state))


async def api_model(request):
    return ok(await asyncio.to_thread(model_detail, request.match_info["name"]))


async def api_model_notes(request):
    folder = safe_join(models_dir, request.match_info["name"])
    if not os.path.isdir(folder):
        raise web.HTTPNotFound()
    p = await body(request)
    with open(os.path.join(folder, "notes.txt"), "w") as f:
        f.write(str(p.get("notes", "")))
    return ok()


async def api_model_publish(request):
    name = request.match_info["name"]
    folder = safe_join(models_dir, name)
    p = await body(request)
    checkpoint = p.get("checkpoint", "best.tflite")
    src = safe_join(os.path.join(folder, "checkpoints"), checkpoint)
    if not src.endswith(".tflite") or not os.path.isfile(src):
        raise web.HTTPNotFound(text=f"no {checkpoint}")
    target = (p.get("target") or "").strip()
    if target.endswith(".tflite"):
        target = target[: -len(".tflite")]
    if not NAME_RE.match(target):
        raise web.HTTPBadRequest(text="name: letters, digits, _ . - only")
    parsed = parse_model_name(name)
    seq = parsed["seq_tag"]
    if seq and not re.search(r"_seq\d", target):
        target += seq  # the phone reads the frame offsets from the file name
    if parsed["steer_only"] and "steeronly" not in target:
        target += "_steeronly"  # the phone won't drive it on its untrained throttle
    dst = os.path.join(models_dir, target + ".tflite")
    if os.path.exists(dst) and not p.get("overwrite"):
        raise web.HTTPConflict(text=f"{target}.tflite is already published")
    # copyfile, not copy2: a fresh mtime is what makes the phone download it again.
    shutil.copyfile(src, dst)
    info = read_json(PUBLISHED, {})
    info[target] = {"model": name, "checkpoint": checkpoint, "time": time.time()}
    write_json(PUBLISHED, info)
    return ok({"name": target})


async def api_unpublish(request):
    target = request.match_info["target"]
    path = safe_join(models_dir, target + ".tflite")
    if os.path.dirname(path) != models_dir or not os.path.isfile(path):
        raise web.HTTPNotFound()
    os.remove(path)
    info = read_json(PUBLISHED, {})
    info.pop(target, None)
    write_json(PUBLISHED, info)
    return ok()


async def api_model_check(request):
    name = request.match_info["name"]
    if not os.path.isdir(safe_join(models_dir, name)):
        raise web.HTTPNotFound()
    j = jobs.submit("check", f"check_tflite {name}", python("check_tflite.py", os.path.join("models", name)), exclusive=False, meta={"model": name})
    return ok(job_view(j))


async def api_runs(request):
    return ok(await asyncio.to_thread(runs_state))


async def api_runs_unpack(request):
    def run():
        for z in runs_state()["zips"]:
            path = os.path.join(RUNS, z)
            autopilot_log.unpack(path, path[: -len(".zip")])

    await asyncio.to_thread(run)
    return ok()


async def api_runs_report(request):
    p = await body(request)
    names = [n for n in p.get("runs") or [] if NAME_RE.match(n)]
    j = jobs.submit("report", "report " + (", ".join(names) if names else "all runs"), python("autopilot_log.py", "report", *names), exclusive=False)
    return ok(job_view(j))


async def api_runs_extract(request):
    p = await body(request)
    names = [n for n in p.get("runs") or [] if NAME_RE.match(n)]
    if not names:
        raise web.HTTPBadRequest(text="pick the runs to extract")
    dry = bool(p.get("dry_run"))
    cmd = python("autopilot_log.py", "extract", *(["--dry_run"] if dry else []), *names)
    j = jobs.submit("extract", ("preview: " if dry else "") + "extract " + ", ".join(names), cmd, exclusive=not dry)
    return ok(job_view(j))


async def api_status(request):
    jobs.tick()
    running = [job_view(j) for j in jobs.jobs if j["status"] == "running"]
    queued = sum(1 for j in jobs.jobs if j["status"] == "queued")
    inbox = len(listdirs(INBOX))
    return ok({"gpu": await asyncio.to_thread(gpu_state), "running": running, "queued": queued, "inbox": inbox})


async def file_from(root, request):
    path = safe_join(root, request.match_info["path"])
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    headers = {"Cache-Control": "public, max-age=86400"} if path.endswith(".jpeg") else {"Cache-Control": "no-cache"}
    return web.FileResponse(path, headers=headers)


async def handle_dataset_file(request):
    return await file_from(dataset_dir, request)


async def handle_model_file(request):
    return await file_from(models_dir, request)


async def handle_index(request):
    return web.FileResponse(os.path.join(WEB_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


async def handle_asset(request):
    path = safe_join(WEB_DIR, request.match_info["path"])
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Cache-Control": "no-cache"})


async def handle_legacy(request):
    """The old trainer UI: hash routing and absolute /static, /ws paths, so its index.html
    works from here; its assets are still served by frontend.py."""
    if request.path == "/legacy":
        raise web.HTTPFound("/legacy/")
    try:
        import openbot_frontend
    except ImportError:
        raise web.HTTPNotFound(text="the old frontend (openbot_frontend) isn't installed")
    return web.FileResponse(os.path.join(openbot_frontend.where(), "index.html"))


async def scheduler(app):
    while True:
        try:
            jobs.tick()
        except Exception as e:  # keep scheduling whatever one job did
            print("fwdbwd scheduler:", e)
        await asyncio.sleep(1)


async def init_fwdbwd(app: web.Application):
    """Registered before init_api and init_frontend, so / and POST /upload are ours."""
    global jobs
    jobs = Jobs()
    for d in (INBOX, RUNS):
        os.makedirs(d, exist_ok=True)
    r = app.router
    r.add_get("/", handle_index)
    r.add_get("/legacy", handle_legacy)
    r.add_get("/legacy/", handle_legacy)
    r.add_post("/upload", handle_upload)
    r.add_get("/fb/file/{path:.+}", handle_dataset_file)
    r.add_get("/fb/modelfile/{path:.+}", handle_model_file)
    r.add_get("/fb/api/status", api_status)
    r.add_get("/fb/api/inbox", api_inbox)
    r.add_post("/fb/api/inbox/remove", api_inbox_remove)
    r.add_post("/fb/api/inbox/unpack", api_inbox_unpack)
    r.add_post("/fb/api/parked/restore", api_parked_restore)
    r.add_get("/fb/api/recording", api_recording)
    r.add_route("*", "/fb/api/settings", api_settings)
    r.add_post("/fb/api/split", api_split)
    r.add_get("/fb/api/sessions", api_sessions)
    r.add_post("/fb/api/sessions/{n:\\d+}/note", api_session_note)
    r.add_get("/fb/api/train/options", api_train_options)
    r.add_post("/fb/api/train", api_train)
    r.add_get("/fb/api/jobs", api_jobs)
    r.add_get("/fb/api/jobs/{id}/log", api_job_log)
    r.add_post("/fb/api/jobs/{id}/stop", api_job_stop)
    r.add_get("/fb/api/models", api_models)
    r.add_get("/fb/api/models/{name}", api_model)
    r.add_post("/fb/api/models/{name}/notes", api_model_notes)
    r.add_post("/fb/api/models/{name}/publish", api_model_publish)
    r.add_post("/fb/api/models/{name}/check", api_model_check)
    r.add_post("/fb/api/published/{target}/delete", api_unpublish)
    r.add_get("/fb/api/runs", api_runs)
    r.add_post("/fb/api/runs/unpack", api_runs_unpack)
    r.add_post("/fb/api/runs/report", api_runs_report)
    r.add_post("/fb/api/runs/extract", api_runs_extract)
    r.add_get("/fb/{path:.+}", handle_asset)

    task = asyncio.get_event_loop().create_task(scheduler(app))

    async def cleanup(app):
        task.cancel()

    app.on_cleanup.append(cleanup)
