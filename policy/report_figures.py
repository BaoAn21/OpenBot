"""Figures and a table for the report's dataset section, made from dataset/fwdbwd/.

Reads only; nothing in the dataset is changed. Writes into --out:

  dataset_timeline.svg     one forward recording: steering/throttle over time, and a
                           1 s zoom of how camera frames are matched to the
                           (asynchronous) control log
  dataset_filmstrip.svg    frames into a turn with their labels, one fwd and one bwd
                           recording
  dataset_samples.svg      random training frames from different sessions
  dataset_labels.svg       steering and throttle label distributions, fwd vs bwd
  dataset_augmentation.svg one frame through the training augmentations
  dataset_sessions.typ     per-session table (recordings, frames, minutes), #include it

Usage (from policy/, conda env):
    python report_figures.py --out ../report_and_slide/report/figures
    python report_figures.py --sessions 15 16 17 --fwd_recording 20261122_221820
"""

import argparse
import glob
import json
import logging
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(POLICY_DIR, "dataset", "fwdbwd")
# The current track ("[line]" sessions); earlier ones were recorded on the old one.
DEFAULT_SESSIONS = list(range(15, 23))
SPLITS = ("train", "test")
DIRECTIONS = ("fwd", "bwd")
MAX = 255.0  # Control.MAX: labels are raw device units

# Reference categorical palette (light), text inks and recessive grid.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIR_COLOR = {"fwd": BLUE, "bwd": ORANGE}

plt.rcParams.update(
    {
        # Text stays text in the SVGs and Typst sets it in the report's font
        # (template.typ); matplotlib only uses the fallback to measure it.
        "font.family": ["New Computer Modern", "DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "svg.fonttype": "none",
        "image.interpolation": "none",  # frames embedded at their own resolution
    }
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ---------------------------------------------------------------- reading


def read_csv(path):
    """Rows of a sensor_data log as lists of strings, header dropped."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        lines = f.read().splitlines()[1:]
    return [l.split(",") for l in lines if l.strip()]


def samples(folder):
    """(timestamp ns, image path, steering, throttle, cmd) per matched frame.

    The frame column holds the path at matching time (often the inbox), so the image
    is looked up by name in this folder.
    """
    rows = read_csv(os.path.join(folder, "sensor_data", "matched_frame_ctrl_cmd_processed.txt"))
    out = []
    for ts, frame, steer, thr, cmd in rows:
        img = os.path.join(folder, "images", os.path.basename(frame))
        out.append((int(ts), img, float(steer), float(thr), float(cmd)))
    return out


def folders(split, direction, session):
    return sorted(glob.glob(os.path.join(ROOT, split, direction, f"session_{session}", "*/")))


def session_note(session):
    path = os.path.join(ROOT, "raw", f"session_{session}", "note.txt")
    if not os.path.isfile(path):
        return ""
    return " ".join(open(path).read().split())


def minutes(sample_list):
    """Recorded time covered by the frames: count x the folder's median frame gap."""
    if len(sample_list) < 2:
        return 0.0
    ts = np.array([s[0] for s in sample_list], dtype=np.float64)
    return len(ts) * np.median(np.diff(ts)) / 1e9 / 60


def pick_recording(sessions, direction):
    """The raw recording with the hardest turn among those driven in this direction.

    Recordings here are usually all forward or all reverse, so each direction gets its own.
    """
    best, best_score = None, -1
    for session in sessions:
        for rec in sorted(glob.glob(os.path.join(ROOT, "raw", f"session_{session}", "2026*/"))):
            rec = rec.rstrip("/")
            if not processed_folders_of(rec)[direction]:
                continue
            ctrl = read_csv(os.path.join(rec, "sensor_data", "ctrlLog.txt"))
            sign = 1 if direction == "fwd" else -1
            steer = [abs(float(r[1])) for r in ctrl if sign * float(r[2]) > 0]
            if steer and max(steer) > best_score:
                best, best_score = rec, max(steer)
    return best


def processed_folders_of(recording):
    """The fwd and bwd output folders split_direction.py made from a raw recording."""
    name = os.path.basename(recording)
    session = os.path.basename(os.path.dirname(recording))
    out = {}
    for d in DIRECTIONS:
        out[d] = sorted(
            glob.glob(os.path.join(ROOT, "*", d, session, name + "_f*/"))
            + glob.glob(os.path.join(ROOT, "*", d, session, name + "/"))
        )
    return out


# ---------------------------------------------------------------- figures


def fig_timeline(recording, out):
    sd = os.path.join(recording, "sensor_data")
    ctrl = np.array([[float(v) for v in r] for r in read_csv(os.path.join(sd, "ctrlLog.txt"))])
    frames = np.array([float(r[0]) for r in read_csv(os.path.join(sd, "rgbFrames.txt"))])
    matched = read_csv(os.path.join(sd, "matched_frame_ctrl.txt"))
    t0 = frames[0]
    sec = lambda ns: (np.asarray(ns, dtype=np.float64) - t0) / 1e9

    fig, (ax, az) = plt.subplots(
        2, 1, figsize=(6.3, 3.8), gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.8}
    )

    # (a) the whole recording; background shaded by how split_direction.py files it
    tc, steer, thr = sec(ctrl[:, 0]), ctrl[:, 1] / MAX, ctrl[:, 2] / MAX
    t_end = max(tc[-1], sec(frames[-1]))
    edges = np.append(tc, t_end)
    for i in range(len(tc)):
        if thr[i] != 0:
            c = DIR_COLOR["fwd" if thr[i] > 0 else "bwd"]
            ax.axvspan(edges[i], edges[i + 1], color=c, alpha=0.08, lw=0)
    ax.step(tc, steer, where="post", color=VIOLET, label="steering")
    ax.step(tc, thr, where="post", color=AQUA, label="throttle")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlim(0, t_end)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("command / 255")
    ax.set_xlabel("time [s]")
    ax.set_title(f"(a) recording {os.path.basename(recording)}", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.15),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.15),
    ]
    labels += ["→ fwd (throttle > 0)", "→ bwd (throttle < 0)"]
    ax.legend(handles, labels, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.08), fontsize=7)

    # (b) 1 s around the biggest steering change while driving: frame timestamps,
    # control timestamps, and which control each frame got
    change = np.abs(np.diff(steer, prepend=steer[0])) * (thr > 0)
    k = int(np.argmax(change))
    lo = max(0.0, tc[k] - 0.5)
    hi = lo + 1.0
    ax.axvspan(lo, hi, fill=False, ec=INK2, lw=0.8, ls="--")
    fr = sec(frames)
    fr = fr[(fr >= lo) & (fr <= hi)]
    cw = tc[(tc >= lo) & (tc <= hi)]
    for ts, off, *_ in matched:
        tf_ = sec(float(ts))
        if lo <= tf_ <= hi:
            az.plot([tf_, tf_ + float(off) / 1e9], [1, 0], color=GRID, lw=1, zorder=1)
    az.vlines(fr, 0.85, 1.15, color=INK, lw=1.2, label=f"camera frame ({len(fr)})")
    az.vlines(cw, -0.15, 0.15, color=AQUA, lw=1.2, label=f"control update ({len(cw)})")
    az.set_yticks([0, 1], ["ctrlLog", "rgbFrames"])
    az.set_ylim(-0.4, 1.4)
    az.set_xlim(lo, hi)
    az.grid(False)
    az.set_xlabel("time [s]")
    az.set_title("(b) 1 s zoom: each frame gets the latest control (grey lines)", loc="left")
    az.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.12), fontsize=7)

    fig.savefig(os.path.join(out, "dataset_timeline.svg"), bbox_inches="tight")
    plt.close(fig)


def label_text(s):
    _, _, steer, thr, cmd = s
    return f"s {steer / MAX:+.2f}\u2002t {thr / MAX:+.2f}\ncmd {int(cmd)}"  # en space: Typst collapses two


def image_grid(rows, out_path, row_labels=None, cols=6):
    """rows: list of lists of samples; one image per cell with its label under it."""
    n = len(rows)
    fig, axes = plt.subplots(
        n, cols, figsize=(6.3, n * 0.68), squeeze=False,
        gridspec_kw={"wspace": 0.05, "hspace": 0.55}
    )
    for r, row in enumerate(rows):
        for c in range(cols):
            a = axes[r][c]
            a.axis("off")
            if c >= len(row):
                continue
            a.imshow(Image.open(row[c][1]))
            a.text(0.5, -0.06, label_text(row[c]), transform=a.transAxes, ha="center",
                   va="top", fontsize=6.5, color=INK, linespacing=1.15)
        if row_labels:
            axes[r][0].text(-0.06, 0.5, row_labels[r], transform=axes[r][0].transAxes,
                            ha="right", va="center", fontsize=7, fontweight="bold",
                            color=list(DIR_COLOR.values())[r % 2] if row_labels[r] in DIR_COLOR else INK)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evenly(seq, n):
    if len(seq) <= n:
        return list(seq)
    return [seq[int(round(i))] for i in np.linspace(0, len(seq) - 1, n)]


def turn_window(ss, half=30):
    """About 2 s of frames centred where steering changes most (into or out of a turn)."""
    steer = np.array([s[2] for s in ss])
    if len(steer) <= 2 * half:
        return slice(0, len(steer))
    change = np.abs(steer[2 * half:] - steer[:-2 * half])
    k = int(np.argmax(change)) + half
    return slice(k - half, k + half)


def fig_filmstrip(recordings, out, cols=6):
    rows, labels = [], []
    for d in DIRECTIONS:
        if not recordings[d]:
            continue
        ss = [s for f in processed_folders_of(recordings[d])[d] for s in samples(f)]
        rows.append(evenly(ss[turn_window(ss)], cols))
        labels.append(d)
    image_grid(rows, os.path.join(out, "dataset_filmstrip.svg"), labels, cols)


def fig_samples(sessions, out, seed, n_rows=3, cols=6):
    rng = random.Random(seed)
    pool = {s: [x for f in folders("train", "fwd", s) for x in samples(f)] for s in sessions}
    pool = {s: v for s, v in pool.items() if v}
    picks = []
    order = sorted(pool)
    for i in range(n_rows * cols):  # round-robin over sessions for variety
        picks.append(rng.choice(pool[order[i % len(order)]]))
    rng.shuffle(picks)
    image_grid([picks[r * cols:(r + 1) * cols] for r in range(n_rows)],
               os.path.join(out, "dataset_samples.svg"), cols=cols)


def fig_labels(sessions, out):
    vals = {d: [x for s in sessions for f in folders("train", d, s) for x in samples(f)]
            for d in DIRECTIONS}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw={"wspace": 0.3})
    bins = np.linspace(-1, 1, 42)
    for ax, col, name in ((axes[0], 2, "steering"), (axes[1], 3, "throttle")):
        for d in DIRECTIONS:
            v = np.array([x[col] for x in vals[d]]) / MAX
            if not len(v):
                continue
            w = np.full(len(v), 100.0 / len(v))
            ax.hist(v, bins=bins, weights=w, histtype="step", lw=1.6, color=DIR_COLOR[d],
                    label=f"{d} ({len(v):,} frames)")
        ax.set_xlabel(f"{name} label / 255")
        ax.set_xlim(-1, 1)
    axes[0].set_ylabel("% of frames")
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=7)
    fig.savefig(os.path.join(out, "dataset_labels.svg"), bbox_inches="tight")
    plt.close(fig)


def fig_augmentation(recording, out, seed):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # a few images, no need for the GPU
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from openbot import data_augmentation

    tf.random.set_seed(seed)
    proc = processed_folders_of(recording)
    ss = [s for f in proc["fwd"] for s in samples(f)]
    s = max(ss, key=lambda x: abs(x[2]))  # a turning frame, so the flip shows
    img = tf.image.convert_image_dtype(tf.io.decode_jpeg(tf.io.read_file(s[1])), tf.float32)

    cells = [(img, "original", f"s {s[2] / MAX:+.2f}")]
    for i in range(3):
        cells.append((data_augmentation.augment_img(img), f"colour #{i + 1}", "always"))
    cells.append((tf.image.flip_left_right(img), "flip", f"s {-s[2] / MAX:+.2f}\u2002(opt-in)"))
    cells.append((data_augmentation.tilt_img(img), "tilt", "label kept\u2002(opt-in)"))

    fig, axes = plt.subplots(2, 3, figsize=(6.3, 2.0), gridspec_kw={"wspace": 0.05, "hspace": 0.45})
    for a, (im, title, sub) in zip(axes.flat, cells):
        a.imshow(np.clip(im.numpy(), 0, 1))
        a.axis("off")
        a.set_title(title, fontsize=7, color=INK, pad=2)
        a.text(0.5, -0.05, sub, transform=a.transAxes, ha="center", va="top", fontsize=6, color=INK2)
    fig.savefig(os.path.join(out, "dataset_augmentation.svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- table


def typ_str(s):
    return json.dumps(s, ensure_ascii=False)  # Typst string escapes match JSON's for ASCII


def table_sessions(sessions, out):
    header = ["Session", "Note", "Rec.", "Train fwd", "Train bwd", "Test fwd", "Test bwd", "Min."]
    rows, total = [], np.zeros(6)
    for s in sessions:
        raw = os.path.join(ROOT, "raw", f"session_{s}")
        recs = [d for d in os.listdir(raw) if os.path.isdir(os.path.join(raw, d))] if os.path.isdir(raw) else []
        counts, mins = [], 0.0
        for sp in SPLITS:
            for d in DIRECTIONS:
                n = 0
                for f in folders(sp, d, s):
                    ss = samples(f)
                    n += len(ss)
                    mins += minutes(ss)
                counts.append(n)
        line = [len(recs)] + counts + [mins]
        total += line
        rows.append([str(s), session_note(s)] + [f"{v:,}" for v in line[:-1]] + [f"{mins:.1f}"])
    rows.append(["Total", ""] + [f"{int(v):,}" for v in total[:-1]] + [f"{total[-1]:.1f}"])

    cells = ",\n  ".join(", ".join(typ_str(c) for c in r) for r in [header] + rows)
    with open(os.path.join(out, "dataset_sessions.typ"), "w") as f:
        f.write(
            "// Generated by policy/report_figures.py - edit the script, not this file.\n"
            "#table(\n"
            "  columns: (auto, 1fr, auto, auto, auto, auto, auto, auto),\n"
            "  align: (x, y) => if x >= 2 { right } else { left },\n"
            "  stroke: none,\n"
            "  table.hline(),\n"
            f"  table.header({', '.join(typ_str(c) for c in header)}),\n"
            "  table.hline(stroke: 0.5pt),\n"
            "  " + ",\n  ".join(", ".join(typ_str(c) for c in r) for r in rows[:-1]) + ",\n"
            "  table.hline(stroke: 0.5pt),\n"
            "  " + ", ".join(typ_str(c) for c in rows[-1]) + ",\n"
            "  table.hline(),\n"
            ")\n"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder (created if missing)")
    p.add_argument("--sessions", type=int, nargs="+", default=DEFAULT_SESSIONS)
    p.add_argument("--fwd_recording", help="raw recording for the timeline and fwd examples (default: picked)")
    p.add_argument("--bwd_recording", help="raw recording for the bwd examples (default: picked)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no_tf", action="store_true", help="skip the augmentation figure")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    recs = {}
    for d in DIRECTIONS:
        name = getattr(args, f"{d}_recording")
        if name:
            hits = glob.glob(os.path.join(ROOT, "raw", "session_*", name))
            if not hits:
                p.error(f"no raw recording {name}")
            recs[d] = hits[0]
        else:
            recs[d] = pick_recording(args.sessions, d)
        print(f"{d} example recording:", recs[d])
    if not recs["fwd"]:
        p.error("no forward recording in these sessions")

    fig_timeline(recs["fwd"], args.out)
    fig_filmstrip(recs, args.out)
    fig_samples(args.sessions, args.out, args.seed)
    fig_labels(args.sessions, args.out)
    table_sessions(args.sessions, args.out)
    if not args.no_tf:
        fig_augmentation(recs["fwd"], args.out, args.seed)
    print("written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
