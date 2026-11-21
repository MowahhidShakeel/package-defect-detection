import os
import re
import glob
import json
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

from config import (
    TAPAL_OK_RAW_DIR, TAPAL_NOT_OK_RAW_DIR,
    RAW_FILTERED_DIR, YOLO_DATASET_DIR, SEED
)

# Keep all 184 OK and 162 Not_OK files (346 total)
DEDUPLICATE_AUG_COPIES = False


def parse_tapal_filename(filepath: Path):
    """
    Parses filenames like: augmented_0_111605_230315_1_0000008974.jpg
    Returns (copy_index, session_id, unique_frame_id)
    """
    stem = filepath.stem
    m = re.match(r"^augmented_(\d+)_(\d+_\d+_\d+)_(\d+)$", stem, re.IGNORECASE)
    if m:
        copy_idx = int(m.group(1))
        session_id = m.group(2)
        frame_num = m.group(3)
        unique_frame_id = f"{session_id}_{frame_num}"
        return copy_idx, session_id, unique_frame_id

    digits = re.findall(r"\d+", stem)
    session_id = digits[1] if len(digits) >= 2 else "session_default"
    return 0, session_id, stem


def filter_and_split_sessions():
    ok_files = sorted([
        Path(p) for ext in ("*.bmp", "*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(str(TAPAL_OK_RAW_DIR / ext))
    ])
    not_ok_files = sorted([
        Path(p) for ext in ("*.bmp", "*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(str(TAPAL_NOT_OK_RAW_DIR / ext))
    ])

    # Sort OK files by physical camera frame ID so copies of the same frame stay in the same split
    ok_real = sorted(ok_files, key=lambda p: parse_tapal_filename(p)[2])
    not_ok_real = sorted(not_ok_files, key=lambda p: parse_tapal_filename(p)[2])

    print(f"[Tapal Dataset] OK: {len(ok_real)} images | Not_OK: {len(not_ok_real)} images "
          f"(Total: {len(ok_real) + len(not_ok_real)})")

    # Group by unique frame ID first, then partition into 10 contiguous session blocks
    # so identical frames never leak across train/val/test
    unique_frames = []
    frame_to_paths = {}
    for p in ok_real:
        _, _, fid = parse_tapal_filename(p)
        if fid not in frame_to_paths:
            unique_frames.append(fid)
            frame_to_paths[fid] = []
        frame_to_paths[fid].append(p)

    n_unique = len(unique_frames)
    n_train = max(1, int(0.70 * n_unique))
    n_val = max(1, int(0.15 * n_unique))

    train_fids = set(unique_frames[:n_train])
    val_fids = set(unique_frames[n_train:n_train + n_val])

    manifest = {"train_ok": [], "val_ok": [], "test_ok": [], "test_not_ok": []}
    for fid in unique_frames:
        target = "train_ok" if fid in train_fids else ("val_ok" if fid in val_fids else "test_ok")
        for p in frame_to_paths[fid]:
            _, sid, _ = parse_tapal_filename(p)
            manifest[target].append({"path": str(p), "session": fid, "label": "OK"})

    for p in not_ok_real:
        _, sid, fid = parse_tapal_filename(p)
        manifest["test_not_ok"].append({"path": str(p), "session": fid, "label": "Not_OK"})

    manifest_path = RAW_FILTERED_DIR / "tapal_session_split.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Session Split Saved] Train OK: {len(manifest['train_ok'])}, "
          f"Val OK: {len(manifest['val_ok'])}, Test OK: {len(manifest['test_ok'])}, "
          f"Test Real Not_OK: {len(manifest['test_not_ok'])}")
    return manifest


def stage0_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Stage 0: Resize to 640x640 and percentile contrast stretch."""
    img = cv2.resize(img_bgr, (640, 640))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p2, p98 = np.percentile(l, (2, 98))
    if p98 > p2:
        l = np.clip((l.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def auto_detect_tapal_print_region(img_bgr: np.ndarray, ref_template: np.ndarray = None):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if ref_template is not None:
        res = cv2.matchTemplate(gray, ref_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.45:
            th, tw = ref_template.shape[:2]
            return [int(max_loc[0]), int(max_loc[1]), int(tw), int(th)]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 150 and bh > 100:
            rx1 = bx + int(bw * 0.08)
            rx2 = bx + int(bw * 0.88)
            ry1 = by + int(bh * 0.30)
            ry2 = by + int(bh * 0.72)

            sub_gray = gray[ry1:ry2, rx1:rx2]
            if sub_gray.size > 0:
                bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                blackhat = cv2.morphologyEx(sub_gray, cv2.MORPH_BLACKHAT, bh_kernel)
                _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
                closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k)

                ys, xs = np.where(closed > 0)
                if len(xs) > 50:
                    pad_x, pad_y = 12, 10
                    x1 = max(0, rx1 + int(np.min(xs)) - pad_x)
                    y1 = max(0, ry1 + int(np.min(ys)) - pad_y)
                    x2 = min(w, rx1 + int(np.max(xs)) + pad_x)
                    y2 = min(h, ry1 + int(np.max(ys)) + pad_y)
                    if (x2 - x1) > 80 and (y2 - y1) > 30:
                        return [x1, y1, x2 - x1, y2 - y1]

            return [rx1, ry1, rx2 - rx1, ry2 - ry1]

    return [110, 260, 395, 125]


class TkinterAnnotator:
    """
    Annotation states stored in saved_boxes[img_path]:
      - {"status": "has_print", "box": [x, y, w, h]}  -> Valid print_region (OK or defective print)
      - {"status": "missing_print", "box": [x, y, w, h]} -> Blank red flap (empty YOLO label, crop still tested in Stage 3b)
      - {"status": "wrong_view", "box": null} -> Wrong carton side / out of frame (empty YOLO label, skipped in Stage 3b crop)
    """
    def __init__(self, all_items, saved_boxes, annotations_file):
        self.all_items = all_items
        self.saved_boxes = saved_boxes
        self.annotations_file = annotations_file
        self.ref_template = None
        self.default_flap_box = [110, 260, 395, 125]
        self.idx = 0
        self.auto_ok_only = False

        self.root = tk.Tk()
        self.root.title("Tapal Print-Region Annotator")
        self.root.geometry("760x740")
        self.root.configure(bg="#222222")

        self.info_lbl = tk.Label(
            self.root,
            text="",
            fg="#00FF88", bg="#222222", font=("Arial", 10, "bold")
        )
        self.info_lbl.pack(pady=6)

        self.canvas = tk.Canvas(self.root, width=640, height=640, bg="black", cursor="cross")
        self.canvas.pack()

        btn_frame = tk.Frame(self.root, bg="#222222")
        btn_frame.pack(pady=8)

        tk.Button(btn_frame, text="Accept Box (SPACE)", command=self.accept_has_print,
                  bg="#2e7d32", fg="white", font=("Arial", 9, "bold"), padx=8).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Blank Flap / No Ink (M)", command=self.mark_missing_print,
                  bg="#e65100", fg="white", font=("Arial", 9, "bold"), padx=8).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Wrong Side / Out of Frame (X)", command=self.mark_wrong_view,
                  bg="#b71c1c", fg="white", font=("Arial", 9, "bold"), padx=8).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Auto-Accept Remaining OKs (A)", command=self.trigger_auto_ok,
                  bg="#1565c0", fg="white", font=("Arial", 9, "bold"), padx=8).pack(side=tk.LEFT, padx=4)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.root.bind("<space>", lambda e: self.accept_has_print())
        self.root.bind("<Return>", lambda e: self.accept_has_print())
        self.root.bind("m", lambda e: self.mark_missing_print())
        self.root.bind("M", lambda e: self.mark_missing_print())
        self.root.bind("x", lambda e: self.mark_wrong_view())
        self.root.bind("X", lambda e: self.mark_wrong_view())
        self.root.bind("a", lambda e: self.trigger_auto_ok())
        self.root.bind("A", lambda e: self.trigger_auto_ok())

        self.start_x = None
        self.start_y = None
        self.cur_box = [110, 260, 395, 125]
        self.cur_img_bgr = None
        self.tk_img = None
        self.rect_id = None

        self.load_next_unannotated()
        if self.idx < len(self.all_items):
            self.root.mainloop()

    def normalize_entry(self, entry):
        if isinstance(entry, list):
            return {"status": "has_print", "box": entry}
        return entry

    def load_next_unannotated(self):
        while self.idx < len(self.all_items):
            item, _ = self.all_items[self.idx]
            img_path = item["path"]
            img_raw = cv2.imread(img_path)
            if img_raw is None:
                self.idx += 1
                continue
            self.cur_img_bgr = stage0_preprocess(img_raw)

            if img_path in self.saved_boxes:
                entry = self.normalize_entry(self.saved_boxes[img_path])
                self.saved_boxes[img_path] = entry
                if entry["status"] == "has_print" and entry["box"]:
                    x, y, bw, bh = entry["box"]
                    if self.ref_template is None and bw > 20 and bh > 20:
                        self.ref_template = cv2.cvtColor(self.cur_img_bgr[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
                        self.default_flap_box = [x, y, bw, bh]
                self.idx += 1
                continue

            self.cur_box = auto_detect_tapal_print_region(self.cur_img_bgr, self.ref_template)

            # Auto-accept only applies to OK images so you can inspect Not_OK defects manually
            if self.auto_ok_only and item["label"] == "OK":
                self.save_entry("has_print", self.cur_box)
                self.idx += 1
                continue
            else:
                self.render_current()
                return

        self.finish()

    def render_current(self):
        item, split_name = self.all_items[self.idx]
        lbl_color = "#00FF88" if item["label"] == "OK" else "#FF5252"
        self.info_lbl.config(
            fg=lbl_color,
            text=f"[{self.idx + 1}/{len(self.all_items)}] {item['label']} ({split_name}) | "
                 f"SPACE=Has Print | M=Blank Flap | X=Wrong Side | Drag=Redraw"
        )
        rgb = cv2.cvtColor(self.cur_img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        x, y, bw, bh = self.cur_box
        self.rect_id = self.canvas.create_rectangle(x, y, x + bw, y + bh, outline="#00FF00", width=3)

    def on_mouse_down(self, event):
        self.start_x, self.start_y = event.x, event.y

    def on_mouse_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline="#FFCC00", width=3
        )

    def on_mouse_up(self, event):
        x1, x2 = sorted([max(0, self.start_x), min(640, event.x)])
        y1, y2 = sorted([max(0, self.start_y), min(640, event.y)])
        if (x2 - x1) > 20 and (y2 - y1) > 15:
            self.cur_box = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            x, y, bw, bh = self.cur_box
            self.ref_template = cv2.cvtColor(self.cur_img_bgr[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
            self.default_flap_box = [x, y, bw, bh]
            self.render_current()

    def save_entry(self, status: str, box):
        item, _ = self.all_items[self.idx]
        img_path = item["path"]
        self.saved_boxes[img_path] = {"status": status, "box": box}
        if status == "has_print" and box is not None and self.ref_template is None:
            x, y, bw, bh = box
            self.ref_template = cv2.cvtColor(self.cur_img_bgr[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
            self.default_flap_box = [x, y, bw, bh]
        if (self.idx + 1) % 20 == 0 or (self.idx + 1) == len(self.all_items):
            with open(self.annotations_file, "w") as f:
                json.dump(self.saved_boxes, f, indent=2)

    def accept_has_print(self):
        self.save_entry("has_print", self.cur_box)
        self.idx += 1
        self.load_next_unannotated()

    def mark_missing_print(self):
        # Blank flap: saves default flap ROI so Stage 3b verifier can also test on the blank crop,
        # while YOLO receives an empty label file (no visible text).
        self.save_entry("missing_print", self.default_flap_box)
        self.idx += 1
        self.load_next_unannotated()

    def mark_wrong_view(self):
        # Sideways box / wrong face: no print region exists at all
        self.save_entry("wrong_view", None)
        self.idx += 1
        self.load_next_unannotated()

    def trigger_auto_ok(self):
        self.auto_ok_only = True
        self.accept_has_print()

    def finish(self):
        with open(self.annotations_file, "w") as f:
            json.dump(self.saved_boxes, f, indent=2)
        try:
            self.root.destroy()
        except Exception:
            pass


def annotate_and_export_yolo(manifest: dict):
    annotations_file = RAW_FILTERED_DIR / "print_region_boxes.json"
    saved_boxes = {}
    if annotations_file.exists():
        with open(annotations_file, "r") as f:
            raw_loaded = json.load(f)
            # Upgrade any old list entries to dict format
            for k, v in raw_loaded.items():
                saved_boxes[k] = {"status": "has_print", "box": v} if isinstance(v, list) else v

    all_items = (
        [(x, "train") for x in manifest["train_ok"]] +
        [(x, "val") for x in manifest["val_ok"]] +
        [(x, "test") for x in manifest["test_ok"]] +
        [(x, "test") for x in manifest["test_not_ok"]]
    )

    print(f"\n--- PRINT REGION ANNOTATOR ({len(all_items)} total images) ---")
    TkinterAnnotator(all_items, saved_boxes, annotations_file)

    for split in ["train", "val", "test"]:
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    exported = 0
    counts = {"has_print": 0, "missing_print": 0, "wrong_view": 0}
    for idx, (item, split_name) in enumerate(all_items):
        img_path = item["path"]
        if img_path not in saved_boxes:
            continue
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            continue
        img = stage0_preprocess(img_raw)
        h, w = img.shape[:2]

        entry = saved_boxes[img_path]
        status = entry.get("status", "has_print")
        box = entry.get("box")
        counts[status] = counts.get(status, 0) + 1

        stem = f"tapal_{item['label']}_{idx:04d}"
        out_img = YOLO_DATASET_DIR / "images" / split_name / f"{stem}.jpg"
        out_lbl = YOLO_DATASET_DIR / "labels" / split_name / f"{stem}.txt"

        cv2.imwrite(str(out_img), img)
        with open(out_lbl, "w") as f:
            if status == "has_print" and box is not None:
                x, y, bw, bh = box
                xc = np.clip((x + bw / 2.0) / w, 0.0, 1.0)
                yc = np.clip((y + bh / 2.0) / h, 0.0, 1.0)
                wn = np.clip(bw / w, 0.01, 1.0)
                hn = np.clip(bh / h, 0.01, 1.0)
                f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            else:
                # Empty label file tells YOLOv8 this image has NO visible print_region (background/negative)
                pass
        exported += 1

    yaml_content = (
        f"path: {YOLO_DATASET_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['print_region']\n"
    )
    with open(YOLO_DATASET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"[YOLOv8 Dataset Ready] Exported {exported} images "
          f"(Has Print: {counts['has_print']}, Blank Flap: {counts['missing_print']}, Wrong Side: {counts['wrong_view']})")


if __name__ == "__main__":
    manifest = filter_and_split_sessions()
    annotate_and_export_yolo(manifest)