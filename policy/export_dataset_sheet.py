"""The final dataset (dataset/fwdbwd/) as a spreadsheet, for looking at or a report screenshot.

Reads every matched_frame_ctrl_cmd_processed.txt of the chosen sessions (the labels the
records are built from) and writes one .xlsx; nothing in the dataset is changed. Sheets:

  processed fwd / bwd        one folder's matched_frame_ctrl_cmd_processed.txt exactly as it
                             is: its header, every row, values untouched; only the image
                             path is shortened to .../<folder>/images/<file>
  Preview fwd / Preview bwd  ~2 s of one recording going into a turn, with each frame's
                             thumbnail beside its label
  Summary                    frames per session, split and direction
  Frames                     every labelled frame (filter/sort in the sheet)

Usage (from policy/, conda env; needs openpyxl):
    python export_dataset_sheet.py --out ../report_and_slide/dataset.xlsx
    python export_dataset_sheet.py --out dataset.xlsx --sessions 15 16 --fwd_recording 20261122_221820
"""

import argparse
import glob
import io
import os

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from PIL import Image

from report_figures import (
    DEFAULT_SESSIONS,
    DIRECTIONS,
    MAX,
    ROOT,
    SPLITS,
    folders,
    pick_recording,
    processed_folders_of,
    samples,
    session_note,
    turn_window,
)

THUMB_W = 160  # px; frames are 256 x 96
HEAD_FONT = Font(bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="2A78D6")
TOTAL_FONT = Font(bold=True)
THIN = Side(style="thin", color="C8C7C2")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center")


def header(ws, names, widths):
    ws.append(names)
    for i, w in enumerate(widths, 1):
        c = ws.cell(row=1, column=i)
        c.font, c.fill, c.alignment, c.border = HEAD_FONT, HEAD_FILL, CENTER, BORDER
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def style_body(ws, formats):
    """Borders, centring and number formats ({column: format}) for every data row."""
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.border = BORDER
            c.alignment = CENTER
            if c.column in formats:
                c.number_format = formats[c.column]


def thumbnail(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((THUMB_W, round(im.height * THUMB_W / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    xl = XlImage(buf)
    return xl, im.height


def processed_sheet(wb, title, folder):
    """The processed label file of one folder, cell for cell."""
    path = os.path.join(folder, "sensor_data", "matched_frame_ctrl_cmd_processed.txt")
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    ws = wb.create_sheet(title)
    header(ws, lines[0].split(","), [20, 46, 8, 8, 6])
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    for line in lines[1:]:
        ts, frame, *values = line.split(",")
        short = ".../" + "/".join(frame.split("/")[-3:])
        # timestamp as text: 16 digits, more than a spreadsheet number keeps exactly
        ws.append([ts, short] + [int(v) for v in values])
    style_body(ws, {1: "@"})
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(horizontal="left", vertical="center")
    ws.cell(row=len(lines) + 2, column=1,
            value=f"{os.path.relpath(path, ROOT)} ({len(lines) - 1} rows). "
                  "left = steering, right = throttle, in [-255, 255]; "
                  "cmd: -1 left, 0 straight, 1 right, 2 reverse").font = Font(italic=True)


def preview_sheet(wb, title, recording, direction, half):
    ws = wb.create_sheet(title)
    ss = [s for f in processed_folders_of(recording)[direction] for s in samples(f)]
    ss = ss[turn_window(ss, half)]
    names = ["#", "frame", "image file", "timestamp [ns]", "time [s]",
             "steering", "throttle", "cmd", "steering / 255", "throttle / 255"]
    header(ws, names, [5, THUMB_W / 6.5, 18, 20, 9, 10, 10, 6, 14, 14])
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    t0 = ss[0][0]
    for i, (ts, img, steer, thr, cmd) in enumerate(ss, 1):
        ws.append([i, None, os.path.basename(img), ts, (ts - t0) / 1e9,
                   int(steer), int(thr), int(cmd), steer / MAX, thr / MAX])
        xl, h = thumbnail(img)
        ws.add_image(xl, f"B{i + 1}")
        ws.row_dimensions[i + 1].height = h * 0.75 + 8  # px -> pt, small margin
    style_body(ws, {4: "0", 5: "0.000", 9: "+0.00;-0.00;0.00", 10: "+0.00;-0.00;0.00"})
    ws.cell(row=len(ss) + 3, column=1,
            value=f"{os.path.relpath(recording, ROOT)} ({direction}), "
                  "from sensor_data/matched_frame_ctrl_cmd_processed.txt").font = Font(italic=True)


def summary_sheet(wb, sessions):
    ws = wb.create_sheet("Summary")
    cols = [f"{sp} {d}" for sp in SPLITS for d in DIRECTIONS]
    header(ws, ["session", "note", "recordings"] + cols + ["total frames"],
           [9, 38, 11] + [11] * len(cols) + [13])
    totals = [0] * (len(cols) + 2)
    for s in sessions:
        raw = os.path.join(ROOT, "raw", f"session_{s}")
        recs = len([d for d in os.listdir(raw) if os.path.isdir(os.path.join(raw, d))]) \
            if os.path.isdir(raw) else 0
        counts = [sum(len(samples(f)) for f in folders(sp, d, s)) for sp in SPLITS for d in DIRECTIONS]
        line = [recs] + counts + [sum(counts)]
        totals = [a + b for a, b in zip(totals, line)]
        ws.append([s, session_note(s)] + line)
    ws.append(["total", ""] + totals)
    style_body(ws, {})
    for c in ws[ws.max_row]:
        c.font = TOTAL_FONT
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(horizontal="left", vertical="center")


def frames_sheet(wb, sessions):
    ws = wb.create_sheet("Frames")
    names = ["session", "split", "direction", "folder", "image file", "timestamp [ns]",
             "steering", "throttle", "cmd", "steering / 255", "throttle / 255"]
    header(ws, names, [9, 7, 10, 24, 18, 20, 10, 10, 6, 14, 14])
    for s in sessions:
        for sp in SPLITS:
            for d in DIRECTIONS:
                for f in folders(sp, d, s):
                    name = os.path.basename(f.rstrip("/"))
                    for ts, img, steer, thr, cmd in samples(f):
                        ws.append([s, sp, d, name, os.path.basename(img), ts,
                                   int(steer), int(thr), int(cmd), steer / MAX, thr / MAX])
    style_body(ws, {6: "0", 10: "+0.00;-0.00;0.00", 11: "+0.00;-0.00;0.00"})
    ws.auto_filter.ref = ws.dimensions
    return ws.max_row - 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help=".xlsx file to write")
    p.add_argument("--sessions", type=int, nargs="+", default=DEFAULT_SESSIONS)
    p.add_argument("--fwd_recording", help="raw recording for Preview fwd (default: picked)")
    p.add_argument("--bwd_recording", help="raw recording for Preview bwd (default: picked)")
    p.add_argument("--preview_frames", type=int, default=24, help="rows per preview sheet")
    args = p.parse_args()

    wb = Workbook()
    wb.remove(wb.active)
    recs = {}
    for d in DIRECTIONS:
        name = getattr(args, f"{d}_recording")
        if name:
            hits = glob.glob(os.path.join(ROOT, "raw", "session_*", name))
            if not hits:
                p.error(f"no raw recording {name}")
            rec = hits[0]
        else:
            rec = pick_recording(args.sessions, d)
        if rec:
            recs[d] = rec
    for d, rec in recs.items():
        folder = processed_folders_of(rec)[d][0].rstrip("/")
        print(f"processed {d}: {folder}")
        processed_sheet(wb, f"processed {d}", folder)
    for d, rec in recs.items():
        print(f"Preview {d}: {rec}")
        preview_sheet(wb, f"Preview {d}", rec, d, args.preview_frames // 2)
    summary_sheet(wb, args.sessions)
    n = frames_sheet(wb, args.sessions)
    wb.save(args.out)
    print(f"{n:,} frames -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
