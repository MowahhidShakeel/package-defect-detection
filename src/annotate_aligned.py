"""
Annotate the print block on ALIGNED carton images.

Every image is first levelled by the carton aligner, so a plain rectangle
fits the print block even when the carton was tilted on the conveyor.

    python annotate_aligned.py            annotate images that have no label yet
    python annotate_aligned.py --review   step through every image again

Keys
    SPACE / ENTER   accept the box shown (print is present)
    drag mouse      redraw the box
    F               carton is upside down: turn the view 180 degrees
    M               blank flap, no ink
    X               wrong side of the carton / out of frame
    B               go back one image
    Q / ESC         save and quit

The suggested box is a rough classical guess. Check every one: the detector
can only be as accurate as these boxes.
"""
from __future__ import annotations

import sys
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from label_locator import LocatorConfig, suggest_print_box
from locator_dataset import (
    SPLITS, HAS_PRINT, MISSING_PRINT, WRONG_VIEW, UprightView,
    load_manifest, load_annotations, save_annotations,
)

MAX_W, MAX_H = 1000, 720


class Annotator:
    def __init__(self, items, annotations, review: bool):
        self.items, self.ann, self.review = items, annotations, review
        self.cfg = LocatorConfig()
        self.idx = 0
        self.view = None
        self.box = None
        self.scale = 1.0
        self.upside_down = False
        self.drag_start = None

        self.root = tk.Tk()
        self.root.title("Tapal print-block annotator (aligned)")
        self.root.configure(bg="#222222")
        self.info = tk.Label(self.root, text="", fg="#00FF88", bg="#222222", font=("Arial", 10, "bold"))
        self.info.pack(pady=6)
        self.canvas = tk.Canvas(self.root, width=MAX_W, height=MAX_H, bg="black", cursor="cross", highlightthickness=0)
        self.canvas.pack()
        tk.Label(self.root, fg="#CCCCCC", bg="#222222", font=("Arial", 9),
                 text="SPACE accept | drag redraw | F upside down | M blank flap | X wrong side | B back | Q quit").pack(pady=6)

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        for key, fn in [("<space>", self.accept), ("<Return>", self.accept), ("f", self.flip), ("F", self.flip),
                        ("m", lambda: self.mark(MISSING_PRINT)), ("M", lambda: self.mark(MISSING_PRINT)),
                        ("x", lambda: self.mark(WRONG_VIEW)), ("X", lambda: self.mark(WRONG_VIEW)),
                        ("b", self.back), ("B", self.back), ("q", self.quit), ("Q", self.quit), ("<Escape>", self.quit)]:
            self.root.bind(key, lambda e, f=fn: f())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.advance(0)
        self.root.mainloop()

    # ------------------------------------------------------------- navigation
    def advance(self, step: int) -> None:
        self.idx += step
        while 0 <= self.idx < len(self.items):
            item, _ = self.items[self.idx]
            if step >= 0 and not self.review and item["path"] in self.ann:
                self.idx += 1
                continue
            if self.load(item):
                return
            self.idx += 1 if step >= 0 else -1
        self.quit()

    def back(self) -> None:
        if self.idx > 0:
            self.idx -= 1
            item, _ = self.items[self.idx]
            if not self.load(item):
                self.back()

    def load(self, item: dict) -> bool:
        self.img = cv2.imread(item["path"])
        if self.img is None:
            return False
        entry = self.ann.get(item["path"])
        self.upside_down = bool(entry.get("upside_down")) if entry else False
        self.build_view(entry)
        return True

    def build_view(self, entry=None) -> None:
        self.view = UprightView.build(self.img, self.cfg, self.upside_down)
        self.box = self.view.box_of(entry) if entry else None
        if self.box is None and self.view.found:
            self.box = suggest_print_box(self.view.image, self.view.face_rect, self.cfg)
        self.render()

    # ---------------------------------------------------------------- drawing
    def render(self) -> None:
        item, split = self.items[self.idx]
        h, w = self.view.image.shape[:2]
        self.scale = min(MAX_W / w, MAX_H / h)
        shown = cv2.resize(self.view.image, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_CUBIC)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
        self.canvas.config(width=shown.shape[1], height=shown.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        s = self.scale
        if self.view.face_rect:
            x1, y1, x2, y2 = self.view.face_rect
            self.canvas.create_rectangle(x1 * s, y1 * s, x2 * s, y2 * s, outline="#00BFFF", width=1, dash=(4, 3))
        if self.box:
            x1, y1, x2, y2 = self.box
            self.canvas.create_rectangle(x1 * s, y1 * s, x2 * s, y2 * s, outline="#00FF00", width=2, tags="box")
        existing = self.ann.get(item["path"], {}).get("status", "new")
        note = "" if self.view.found else " | CARTON FACE NOT FOUND"
        turned = " | turned 180" if self.upside_down else ""
        self.info.config(fg="#00FF88" if item["label"] == "OK" else "#FF5252",
                         text=f"[{self.idx + 1}/{len(self.items)}] {item['label']} ({split}) | saved: {existing}{turned}{note}")

    def on_down(self, e) -> None:
        self.drag_start = (e.x, e.y)

    def on_drag(self, e) -> None:
        if self.drag_start:
            self.canvas.delete("box")
            self.canvas.create_rectangle(*self.drag_start, e.x, e.y, outline="#FFCC00", width=2, tags="box")

    def on_up(self, e) -> None:
        if not self.drag_start:
            return
        (x0, y0), s = self.drag_start, self.scale
        self.drag_start = None
        x1, x2 = sorted([x0 / s, e.x / s])
        y1, y2 = sorted([y0 / s, e.y / s])
        if x2 - x1 > 12 and y2 - y1 > 8:
            self.box = (x1, y1, x2, y2)
        self.render()

    # ---------------------------------------------------------------- actions
    def flip(self) -> None:
        self.upside_down = not self.upside_down
        self.build_view(None)

    def accept(self) -> None:
        if self.box is None:
            self.info.config(text="No box yet: drag one around the print, or press M / X.")
            return
        self.store(HAS_PRINT, self.view.quad_of(self.box))

    def mark(self, status: str) -> None:
        self.store(status, None)

    def store(self, status: str, quad) -> None:
        item, _ = self.items[self.idx]
        self.ann[item["path"]] = {"status": status, "upside_down": self.upside_down, "quad": quad}
        save_annotations(self.ann)
        self.advance(1)

    def quit(self) -> None:
        save_annotations(self.ann)
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    manifest = load_manifest()
    items = [(item, key) for key, _ in SPLITS for item in manifest[key]]
    annotations = load_annotations()
    print(f"{len(items)} images, {sum(1 for i, _ in items if i['path'] in annotations)} already annotated.")
    Annotator(items, annotations, review="--review" in sys.argv)
    done = load_annotations()
    by_status = {}
    for i, _ in items:
        s = done.get(i["path"], {}).get("status", "not annotated")
        by_status[s] = by_status.get(s, 0) + 1
    print(f"Annotation state: {by_status}")


if __name__ == "__main__":
    main()
