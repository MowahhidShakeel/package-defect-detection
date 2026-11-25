#!/usr/bin/env python3
"""
Demo window for the carton label inspector.

    python3 jetson_gui.py --bundle .

Two modes:

    Folder   inspects the images already in a folder, one after another.
    Live     waits for images to arrive in a folder (the camera's FTP target)
             and inspects each one as soon as its upload has finished.

For every carton the window shows the camera image with the inspected
region outlined, the label crop, the defect map, the verdict with its
confidence, the defect score and the time each stage took. Every result is
written to log.csv in the results folder; for every NOT OK carton the
camera image and a picture with the defect map are saved as well.

Needs jetson_inspector.py next to it, and Tk:

    sudo apt-get install python3-tk

It also runs on a PC (Windows or Linux) with `pip install onnxruntime
opencv-python numpy`, which is the easy way to try it before going to the
Jetson. Written for Python 3.6.
"""
import argparse
import base64
import os
import queue
import threading
import time

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

import jetson_inspector as ji

GREEN, RED, GREY, DARK = "#1b8a3a", "#c62828", "#555555", "#202020"
HISTORY_ROWS = 500


# ==============================================================================
# The worker thread: the model lives here, so the window never freezes
# ==============================================================================
class Worker(threading.Thread):
    """Jobs go in through .jobs, messages come out through .out.

    Jobs:      ("folder", folder, out_dir, save_ok, pause_s)
               ("live",   folder, out_dir, save_ok, move)
               ("show",   path)                      inspect again, save nothing
               ("quit",)
    Messages:  ("ready", text) ("fatal", text) ("result", result, saved) ("progress", done, total)
               ("unreadable", name) ("error", text) ("idle",)
    """

    def __init__(self, bundle_dir, backend):
        threading.Thread.__init__(self)
        self.daemon = True
        self.bundle_dir, self.backend = bundle_dir, backend
        self.jobs, self.out = queue.Queue(), queue.Queue()
        self.stop = threading.Event()
        self.inspector = None

    def run(self):
        try:
            self.inspector = ji.Inspector(self.bundle_dir, self.backend)
            self.inspector.warm_up()
        except Exception as e:
            self.out.put(("fatal", "{}: {}".format(type(e).__name__, e)))
            return
        self.out.put(("ready", "Model {}  |  {}  |  threshold {:.4f}".format(
            self.inspector.bundle.get("name", "?"), self.inspector.backend.name, self.inspector.threshold)))
        while True:
            job = self.jobs.get()
            if job[0] == "quit":
                break
            self.stop.clear()
            try:
                if job[0] == "folder":
                    self._folder(*job[1:])
                elif job[0] == "live":
                    self._live(*job[1:])
                elif job[0] == "show":
                    r = self.inspector.inspect_file(job[1])
                    if r is None:
                        self.out.put(("error", "Could not read " + job[1]))
                    else:
                        self.out.put(("result", r, False))
            except Exception as e:
                self.out.put(("error", "{}: {}".format(type(e).__name__, e)))
            self.out.put(("idle",))
        self.inspector.close()

    def _one(self, path, store, move):
        r = self.inspector.inspect_file(path)
        if r is None:
            store.unreadable(path, move)
            self.out.put(("unreadable", os.path.basename(path)))
            return
        r["path"] = store.record(r, move)
        self.out.put(("result", r, True))

    def _folder(self, folder, out_dir, save_ok, pause_s):
        store = ji.ResultStore(out_dir, save_ok)
        try:
            names = ji.list_images(folder)
            self.out.put(("progress", 0, len(names)))
            for i, name in enumerate(names):
                if self.stop.is_set():
                    break
                self._one(os.path.join(folder, name), store, False)
                self.out.put(("progress", i + 1, len(names)))
                if pause_s > 0:
                    self.stop.wait(pause_s)
        finally:
            store.close()

    def _live(self, folder, out_dir, save_ok, move):
        store = ji.ResultStore(out_dir, save_ok)
        seen = set()
        try:
            while not self.stop.is_set():
                names = [n for n in ji.list_images(folder) if n not in seen]
                if not names:
                    self.stop.wait(0.02)
                    continue
                for name in names:
                    if self.stop.is_set():
                        break
                    path = os.path.join(folder, name)
                    if not ji.wait_until_written(path):
                        continue                    # still uploading, or already gone: look again next round
                    seen.add(name)
                    self._one(path, store, move)
                if move:
                    seen.clear()
        finally:
            store.close()


# ==============================================================================
# The window
# ==============================================================================
def to_photo(bgr, scale=1.0):
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class App(object):
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.running = False
        self.ready = False
        self.model_text = ""
        self.counts = {"OK": 0, "NOT_OK": 0}
        self.latencies = []
        self.paths = {}                     # history row -> where its image is now
        self.photos = {}                    # keeps the pictures alive (Tk forgets them otherwise)
        screen = root.winfo_screenheight()          # fit small screens, e.g. a 7 inch 1024 x 600 display
        self.scale = 1.0 if screen >= 960 else (0.75 if screen >= 720 else 0.55)
        self.history_height = 7 if screen >= 960 else (4 if screen >= 720 else 2)

        root.title("Carton label inspection")
        root.configure(bg="white")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        style = ttk.Style()
        style.configure("Treeview", rowheight=22)

        self._build_controls()
        self._build_display()
        self._build_history()
        self.status = tk.StringVar(value="Loading the model ...")
        tk.Label(root, textvariable=self.status, anchor="w", bg=DARK, fg="white", padx=8).pack(fill="x", side="bottom")

        self.worker = Worker(args.bundle, args.backend)
        self.worker.start()
        self.root.after(30, self.poll)

    # ---- layout ----------------------------------------------------------------
    def _build_controls(self):
        a = self.args
        bar = tk.Frame(self.root, bg="white", padx=8, pady=6)
        bar.pack(fill="x")
        self.mode = tk.StringVar(value="live" if a.watch else "folder")
        self.folder = tk.StringVar(value=a.watch or a.folder or "")
        self.out_dir = tk.StringVar(value=a.out)
        self.pause = tk.DoubleVar(value=a.pause)
        self.move = tk.BooleanVar(value=True)
        self.save_ok = tk.BooleanVar(value=False)

        top = tk.Frame(bar, bg="white")
        top.pack(fill="x")
        tk.Label(top, text="Mode", bg="white", font=("", 10, "bold"), width=14, anchor="w").pack(side="left")
        tk.Radiobutton(top, text="Folder of images", variable=self.mode, value="folder", bg="white",
                       command=self._mode_changed).pack(side="left")
        tk.Radiobutton(top, text="Live: wait for images from the camera", variable=self.mode, value="live",
                       bg="white", command=self._mode_changed).pack(side="left", padx=(12, 0))
        self.stop_button = tk.Button(top, text="Stop", width=8, font=("", 11, "bold"), state="disabled", command=self.stop)
        self.stop_button.pack(side="right", padx=(6, 0))
        self.start_button = tk.Button(top, text="Start", width=8, bg=GREEN, fg="white", font=("", 11, "bold"),
                                      state="disabled", command=self.start)
        self.start_button.pack(side="right")

        for label, var in (("Images folder", self.folder), ("Save results to", self.out_dir)):
            row = tk.Frame(bar, bg="white")
            row.pack(fill="x", pady=2)
            text = tk.Label(row, text=label, bg="white", width=14, anchor="w")
            text.pack(side="left")
            tk.Button(row, text="Browse", command=lambda v=var: self._browse(v)).pack(side="right", padx=(6, 0))
            tk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            if var is self.folder:
                self.folder_label = text

        options = tk.Frame(bar, bg="white")
        options.pack(fill="x")
        tk.Label(options, text="", bg="white", width=14).pack(side="left")
        self.pause_box = tk.Frame(options, bg="white")
        tk.Label(self.pause_box, text="Show each image for (s)", bg="white").pack(side="left")
        tk.Spinbox(self.pause_box, from_=0.0, to=5.0, increment=0.1, width=4, textvariable=self.pause).pack(side="left", padx=4)
        self.move_box = tk.Checkbutton(options, text="Move inspected images into the results folder",
                                       variable=self.move, bg="white")
        self.save_ok_box = tk.Checkbutton(options, text="Save a defect-map picture for OK cartons too",
                                          variable=self.save_ok, bg="white")
        self._mode_changed()

    def _build_display(self):
        s = self.scale
        body = tk.Frame(self.root, bg="white", padx=8)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="white")
        left.pack(side="left", anchor="n")
        right = tk.Frame(body, bg="white")
        right.pack(side="left", anchor="n", padx=(12, 0))

        tk.Label(left, text="Camera image (inspected region outlined)", bg="white", fg=GREY).pack(anchor="w")
        self.camera = tk.Label(left, bg="#dddddd", width=int(512 * s), height=int(480 * s), image=self._blank(512, 480))
        self.camera.pack()
        self.details = tk.StringVar(value="")
        tk.Label(left, textvariable=self.details, bg="white", justify="left", anchor="w",
                 font=("Courier", 10 if s == 1.0 else 8)).pack(anchor="w", pady=(6, 0))

        self.verdict = tk.Label(right, text="-", bg=GREY, fg="white", font=("", int(34 * s), "bold"), pady=int(8 * s))
        self.verdict.pack(fill="x")
        self.verdict_note = tk.Label(right, text="", bg="white", fg=GREY, justify="left", wraplength=int(448 * s))
        self.verdict_note.pack(anchor="w")
        tk.Label(right, text="Label region", bg="white", fg=GREY).pack(anchor="w")
        self.crop = tk.Label(right, bg="#dddddd", image=self._blank(448, 224))
        self.crop.pack()
        tk.Label(right, text="Defect map: colour marks where the print looks defective", bg="white", fg=GREY,
                 wraplength=int(448 * s), justify="left").pack(anchor="w", pady=(6, 0))
        self.heat = tk.Label(right, bg="#dddddd", image=self._blank(448, 224))
        self.heat.pack()

    def _build_history(self):
        box = tk.Frame(self.root, bg="white", padx=8, pady=6)
        box.pack(fill="x")
        self.summary = tk.StringVar(value="Inspected 0")
        tk.Label(box, textvariable=self.summary, bg="white", anchor="w", font=("", 10, "bold")).pack(fill="x")
        cols = (("time", "Time", 70), ("file", "Image", 260), ("verdict", "Verdict", 70),
                ("conf", "Confidence", 85), ("score", "Score", 70), ("ms", "Latency (ms)", 90))
        self.history = ttk.Treeview(box, columns=[c[0] for c in cols], show="headings", height=self.history_height)
        for key, title, width in cols:
            self.history.heading(key, text=title)
            self.history.column(key, width=width, anchor="w" if key == "file" else "center")
        self.history.tag_configure("bad", foreground=RED)
        bar = ttk.Scrollbar(box, orient="vertical", command=self.history.yview)
        self.history.configure(yscrollcommand=bar.set)
        self.history.pack(side="left", fill="x", expand=True)
        bar.pack(side="left", fill="y")
        self.history.bind("<<TreeviewSelect>>", self._history_clicked)

    def _blank(self, w, h):
        key = "blank{}x{}".format(w, h)
        self.photos[key] = tk.PhotoImage(width=int(w * self.scale), height=int(h * self.scale))
        return self.photos[key]

    def _mode_changed(self):
        live = self.mode.get() == "live"
        self.folder_label.config(text="Folder to watch" if live else "Images folder")
        for box in (self.pause_box, self.move_box, self.save_ok_box):
            box.pack_forget()
        (self.move_box if live else self.pause_box).pack(side="left")
        self.save_ok_box.pack(side="left", padx=(16, 0))

    def _browse(self, var):
        chosen = filedialog.askdirectory(initialdir=var.get() or os.path.expanduser("~"))
        if chosen:
            var.set(chosen)

    # ---- actions ---------------------------------------------------------------
    def start(self):
        folder, out_dir = self.folder.get().strip(), self.out_dir.get().strip()
        if not os.path.isdir(folder):
            messagebox.showerror("Folder not found", "This folder does not exist:\n" + (folder or "(empty)"))
            return
        if not out_dir:
            messagebox.showerror("Results folder", "Choose a folder to save the results to.")
            return
        if os.path.abspath(out_dir).startswith(os.path.abspath(folder) + os.sep) or os.path.abspath(out_dir) == os.path.abspath(folder):
            messagebox.showerror("Results folder", "The results folder must not be inside the images folder.")
            return
        try:
            pause = max(0.0, float(self.pause.get()))
        except (tk.TclError, ValueError):
            pause = 0.0
        self.counts = {"OK": 0, "NOT_OK": 0}
        self.latencies = []
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        if self.mode.get() == "live":
            self.status.set("Live: waiting for images in " + folder)
            self.worker.jobs.put(("live", folder, out_dir, self.save_ok.get(), self.move.get()))
        else:
            self.status.set("Inspecting the images in " + folder)
            self.worker.jobs.put(("folder", folder, out_dir, self.save_ok.get(), pause))

    def stop(self):
        self.worker.stop.set()
        self.stop_button.config(state="disabled")

    def on_close(self):
        self.worker.stop.set()
        self.worker.jobs.put(("quit",))
        self.worker.join(3.0)
        self.root.destroy()

    def _history_clicked(self, _event):
        chosen = self.history.selection()
        if self.running or not self.ready or not chosen:
            return
        path = self.paths.get(chosen[0])
        if path and os.path.exists(path):
            self.worker.jobs.put(("show", path))

    # ---- messages from the worker ------------------------------------------------
    def poll(self):
        latest = None
        try:
            while True:
                msg = self.worker.out.get_nowait()
                kind = msg[0]
                if kind == "result":
                    if msg[2]:
                        self._add_to_history(msg[1])
                    latest = msg[1]
                elif kind == "ready":
                    self.ready = True
                    self.model_text = msg[1]
                    self.status.set(msg[1] + "  |  Ready")
                    self.start_button.config(state="normal")
                    if self.args.autostart:
                        self.start()
                elif kind == "progress":
                    self.status.set("{}  |  {} of {} images".format(self.model_text, msg[1], msg[2]))
                elif kind == "unreadable":
                    self.status.set("Could not read " + msg[1])
                elif kind == "error":
                    self.status.set("Error: " + msg[1])
                elif kind == "fatal":
                    self.status.set("The model could not be loaded: " + msg[1])
                    messagebox.showerror("The model could not be loaded", msg[1])
                elif kind == "idle":
                    if self.running:
                        self.running = False
                        self.status.set(self.model_text + "  |  Stopped. Click a row to see that carton again.")
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
        except queue.Empty:
            pass
        if latest is not None:                  # only the newest picture is drawn; every result is in the list
            self.show(latest)
        self.root.after(30, self.poll)

    def _add_to_history(self, r):
        self.counts[r["verdict"]] += 1
        self.latencies.append(r["ms_total"])
        row = self.history.insert("", 0, values=(
            time.strftime("%H:%M:%S"), r["name"], "NOT OK" if r["verdict"] == "NOT_OK" else "OK",
            "{:.1f} %".format(r["confidence"]), "{:.4f}".format(r["score"]), "{:.0f}".format(r["ms_total"])),
            tags=("bad",) if r["verdict"] == "NOT_OK" else ())
        self.paths[row] = r["path"]
        rows = self.history.get_children()
        for old in rows[HISTORY_ROWS:]:
            self.history.delete(old)
            self.paths.pop(old, None)
        n = self.counts["OK"] + self.counts["NOT_OK"]
        recent = self.latencies[-200:]
        self.summary.set("Inspected {}     OK {}     NOT OK {}     median latency {:.0f} ms (last {})".format(
            n, self.counts["OK"], self.counts["NOT_OK"], float(np.median(recent)), len(recent)))

    def show(self, r):
        bad = r["verdict"] == "NOT_OK"
        self.verdict.config(text=ji.verdict_text(r), bg=RED if bad else GREEN)
        self.verdict_note.config(text="Confidence: how far the score is from the threshold (50% at the threshold)."
                                 + ("" if r["face_found"] else "  No carton found in the image."))
        self.photos["camera"] = to_photo(ji.camera_view(r), self.scale)
        self.photos["crop"] = to_photo(r["crop"], self.scale)
        self.photos["heat"] = to_photo(ji.heat_overlay(r["crop"], r["map"]), self.scale)
        self.camera.config(image=self.photos["camera"], width=self.photos["camera"].width(), height=self.photos["camera"].height())
        self.crop.config(image=self.photos["crop"])
        self.heat.config(image=self.photos["heat"])
        self.details.set(
            "Image         {}\n"
            "Defect score  {:.4f}   (NOT OK from {:.4f})\n"
            "Latency       {:.0f} ms total\n"
            "              read {:.0f} | align+crop {:.0f} | model {:.0f} | score {:.0f}".format(
                r["name"], r["score"], r["threshold"], r["ms_total"],
                r["ms_read"], r["ms_crop"], r["ms_model"], r["ms_score"]))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Demo window for the carton label inspector.")
    p.add_argument("--bundle", default=here, help="folder holding bundle.json and model.onnx / model.engine")
    p.add_argument("--backend", default="auto", choices=["auto", "trt", "ort"])
    p.add_argument("--folder", help="start in folder mode with this folder filled in")
    p.add_argument("--watch", metavar="FOLDER", help="start in live mode with this folder filled in")
    p.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "inspection_results"))
    p.add_argument("--pause", type=float, default=0.5, help="folder mode: seconds to show each image")
    p.add_argument("--autostart", action="store_true", help="press Start as soon as the model is loaded")
    args = p.parse_args()

    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
