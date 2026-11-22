#!/usr/bin/env python3
"""
Live inspection on the Jetson Nano.

This one file is all the Jetson needs, together with the bundle folder made
by `python -m deploy.export` on the workstation (model.onnx, bundle.json).
It does not import the rest of the project and needs no PyTorch: only
numpy and OpenCV (both come with JetPack), plus ONE of
    - tensorrt + pycuda   (fast; build model.engine first, see below)
    - onnxruntime         (simple; slower)

It is written for Python 3.6, the version on the original Jetson Nano.

Build the TensorRT engine once, on the Jetson itself:

    /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

Then:

    python3 jetson_inspector.py --bundle . --image some_carton.bmp        one image
    python3 jetson_inspector.py --bundle . --benchmark some_carton.bmp    speed of each stage
    python3 jetson_inspector.py --bundle . --watch /srv/ftp/incoming --out /home/nano/inspection

In --watch mode every image that appears in the folder (the camera's FTP
target) is inspected, logged to <out>/log.csv and moved to <out>/ok or
<out>/not_ok. With --save-maps a picture of the defect map is kept for
every NOT OK carton.

The alignment and cropping code below repeats src/preprocessing/align.py
on purpose. `deploy.export` checks that the two give the same crops.
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np

IMAGE_TYPES = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ==============================================================================
# Stage 1: align the carton and cut the label region (mirrors preprocessing/align.py)
# ==============================================================================
def _apply_affine(points, M):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts.dot(M[:, :2].T) + M[:, 2]


def _carton_mask(img, cfg):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg["hsv_ranges"]:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    k = max(5, int(round(0.03 * min(img.shape[:2]))) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return mask


def align_carton(img, cfg):
    """Returns (level image, face rectangle or None)."""
    h, w = img.shape[:2]
    found = cv2.findContours(_carton_mask(img, cfg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]          # OpenCV 3 and 4 differ here
    if not contours:
        return img, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg["min_face_area_frac"] * h * w:
        return img, None

    rect = cv2.minAreaRect(contour)
    pts = cv2.boxPoints(rect).astype(np.float64)
    e0, e1 = pts[1] - pts[0], pts[2] - pts[1]
    long_edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0

    if abs(angle) < cfg["min_align_angle_deg"]:
        aligned = img
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    else:
        M = cv2.getRotationMatrix2D((float(rect[0][0]), float(rect[0][1])), angle, 1.0)
        corners = _apply_affine(np.array([[0, 0], [w, 0], [w, h], [0, h]]), M)
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        M[0, 2] -= x_min
        M[1, 2] -= y_min
        size = (int(math.ceil(x_max - x_min)), int(math.ceil(y_max - y_min)))
        aligned = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    p = _apply_affine(pts, M)
    ah, aw = aligned.shape[:2]
    face = (max(0.0, p[:, 0].min()), max(0.0, p[:, 1].min()), min(float(aw), p[:, 0].max()), min(float(ah), p[:, 1].max()))
    return aligned, face


def label_crop(img, bundle):
    """Returns (crop BGR uint8, carton face found?)."""
    aligned, face = align_carton(img, bundle["align"])
    region = bundle["crop_region"]
    found = face is not None
    if not found:
        h, w = img.shape[:2]
        f = region["fallback_face"]
        face = (f[0] * w, f[1] * h, f[2] * w, f[3] * h)
    out_w, out_h = int(region["out_w"]), int(region["out_h"])
    fw = face[2] - face[0]
    cx, cy = face[0] + region["cu"] * fw, face[1] + region["cv"] * fw
    bw = region["wu"] * fw
    bh = bw * out_h / float(out_w)
    x1, y1 = cx - bw / 2.0, cy - bh / 2.0
    sx, sy = out_w / max(bw, 1e-6), out_h / max(bh, 1e-6)
    M = np.array([[sx, 0.0, -x1 * sx], [0.0, sy, -y1 * sy]])
    interp = cv2.INTER_AREA if sx < 1.0 else cv2.INTER_CUBIC
    crop = cv2.warpAffine(aligned, M, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_REPLICATE)
    return crop, found


# ==============================================================================
# Stage 2: the model. Input 1 x 3 x H x W float32, RGB, 0..255. Output: defect map 0..1.
# ==============================================================================
class OnnxRuntimeBackend(object):
    name = "onnxruntime"

    def __init__(self, onnx_path):
        import onnxruntime as ort
        wanted = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.name = "onnxruntime (" + self.session.get_providers()[0] + ")"

    def __call__(self, x):
        return self.session.run(None, {self.input_name: x})[0]


class TensorRTBackend(object):
    """TensorRT 7 / 8 (JetPack 4.x). NOT tested by the author: there was no Jetson to test on."""
    name = "tensorrt"

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.host, self.device, self.shapes = [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            self.host.append(host)
            self.device.append(cuda.mem_alloc(host.nbytes))
            self.shapes.append(shape)
        self.in_idx = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)][0]
        self.out_idx = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)][0]

    def __call__(self, x):
        np.copyto(self.host[self.in_idx], x.ravel())
        self.cuda.memcpy_htod(self.device[self.in_idx], self.host[self.in_idx])
        self.context.execute_v2([int(d) for d in self.device])
        self.cuda.memcpy_dtoh(self.host[self.out_idx], self.device[self.out_idx])
        return self.host[self.out_idx].reshape(self.shapes[self.out_idx]).astype(np.float32)


def load_backend(bundle_dir, choice):
    engine = os.path.join(bundle_dir, "model.engine")
    onnx_path = os.path.join(bundle_dir, "model.onnx")
    if choice in ("auto", "trt") and os.path.exists(engine):
        try:
            return TensorRTBackend(engine)
        except Exception as e:  # missing library, wrong TensorRT version, ...
            if choice == "trt":
                raise
            print("TensorRT engine found but could not be used ({}). Falling back to onnxruntime.".format(e))
    elif choice == "trt":
        raise IOError("No model.engine in {}. Build it with trtexec (see the top of this file).".format(bundle_dir))
    return OnnxRuntimeBackend(onnx_path)


# ==============================================================================
# Stage 3: score and verdict
# ==============================================================================
def carton_score(prob, top_k):
    """Mean of the top_k highest values of the lightly smoothed defect map
    (same as inspector.model.carton_score)."""
    smooth = cv2.blur(prob, (5, 5), borderType=cv2.BORDER_CONSTANT)
    flat = smooth.ravel()
    return float(np.partition(flat, flat.size - top_k)[flat.size - top_k:].mean())


class Inspector(object):
    def __init__(self, bundle_dir, backend="auto"):
        with open(os.path.join(bundle_dir, "bundle.json")) as f:
            self.bundle = json.load(f)
        self.backend = load_backend(bundle_dir, backend)
        self.threshold = float(self.bundle["threshold"])
        self.top_k = int(self.bundle["top_k"])

    def inspect(self, img_bgr):
        t0 = time.time()
        crop, face_found = label_crop(img_bgr, self.bundle)
        t1 = time.time()
        x = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        prob = np.asarray(self.backend(x), dtype=np.float32).reshape(crop.shape[0], crop.shape[1])
        t2 = time.time()
        score = carton_score(prob, self.top_k)
        t3 = time.time()
        return {
            "verdict": "NOT_OK" if score >= self.threshold else "OK",
            "score": score, "face_found": face_found, "crop": crop, "map": prob,
            "ms_crop": (t1 - t0) * 1000.0, "ms_model": (t2 - t1) * 1000.0,
            "ms_score": (t3 - t2) * 1000.0, "ms_total": (t3 - t0) * 1000.0,
        }


def map_picture(result):
    heat = cv2.applyColorMap((result["map"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(result["crop"], 0.55, heat, 0.45, 0)
    return np.hstack([result["crop"], overlay])


# ==============================================================================
# Command line
# ==============================================================================
def _wait_until_written(path, tries=40):
    """An FTP upload appears before it is complete: wait until the size stops changing."""
    last = -1
    for _ in range(tries):
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size > 0 and size == last:
            return True
        last = size
        time.sleep(0.05)
    return False


def watch(inspector, folder, out_dir, save_maps):
    for sub in ("ok", "not_ok", "unreadable", "maps"):
        if not os.path.isdir(os.path.join(out_dir, sub)):
            os.makedirs(os.path.join(out_dir, sub))
    log_path = os.path.join(out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    print("Watching {} | model: {} | threshold {:.4f} | backend: {}".format(
        folder, inspector.bundle["name"], inspector.threshold, inspector.backend.name))
    print("Press Ctrl+C to stop.")
    counts = {"OK": 0, "NOT_OK": 0}
    with open(log_path, "a") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["time", "file", "verdict", "score", "face_found", "ms_total", "ms_model"])
        try:
            while True:
                names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_TYPES))
                if not names:
                    time.sleep(0.02)
                    continue
                for name in names:
                    path = os.path.join(folder, name)
                    if not _wait_until_written(path):
                        continue
                    img = cv2.imread(path)
                    if img is None:
                        shutil.move(path, os.path.join(out_dir, "unreadable", name))
                        print("{}  could not be read".format(name))
                        continue
                    r = inspector.inspect(img)
                    counts[r["verdict"]] += 1
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([stamp, name, r["verdict"], "{:.4f}".format(r["score"]), int(r["face_found"]),
                                     "{:.1f}".format(r["ms_total"]), "{:.1f}".format(r["ms_model"])])
                    f.flush()
                    if r["verdict"] == "NOT_OK" and save_maps:
                        cv2.imwrite(os.path.join(out_dir, "maps", os.path.splitext(name)[0] + ".jpg"), map_picture(r))
                    shutil.move(path, os.path.join(out_dir, "ok" if r["verdict"] == "OK" else "not_ok", name))
                    print("{}  {:6s}  score {:.3f}  {:.0f} ms{}".format(
                        name, r["verdict"], r["score"], r["ms_total"], "" if r["face_found"] else "  (no carton face found)"))
        except KeyboardInterrupt:
            print("\nStopped. OK: {}  NOT OK: {}  Log: {}".format(counts["OK"], counts["NOT_OK"], log_path))


def benchmark(inspector, image_path, runs):
    img = cv2.imread(image_path)
    if img is None:
        sys.exit("Could not read " + image_path)
    for _ in range(10):                      # warm-up: the first runs are always slow
        inspector.inspect(img)
    rows = [inspector.inspect(img) for _ in range(runs)]
    print("Backend: {} | {} runs after warm-up | input {}x{}".format(
        inspector.backend.name, runs, inspector.bundle["crop_region"]["out_w"], inspector.bundle["crop_region"]["out_h"]))
    for key, label in (("ms_crop", "align + crop"), ("ms_model", "model"), ("ms_score", "score"), ("ms_total", "total")):
        v = np.array([r[key] for r in rows])
        print("  {:<13} median {:7.1f} ms   95th percentile {:7.1f} ms".format(label, np.median(v), np.percentile(v, 95)))
    total = np.median([r["ms_total"] for r in rows])
    print("  That is about {:.1f} cartons per second, not counting the FTP transfer.".format(1000.0 / total))


def main():
    p = argparse.ArgumentParser(description="Inspect Tapal carton labels on the Jetson Nano.")
    p.add_argument("--bundle", default=".", help="folder holding bundle.json and model.onnx / model.engine")
    p.add_argument("--backend", default="auto", choices=["auto", "trt", "ort"])
    p.add_argument("--image", help="inspect one image and print the verdict")
    p.add_argument("--benchmark", metavar="IMAGE", help="time each stage on one image")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--watch", metavar="FOLDER", help="inspect every image that arrives in this folder")
    p.add_argument("--out", default="inspection", help="where --watch puts its log and the sorted images")
    p.add_argument("--save-maps", action="store_true", help="keep a defect-map picture for each NOT OK carton")
    a = p.parse_args()

    inspector = Inspector(a.bundle, "ort" if a.backend == "ort" else a.backend)
    if a.benchmark:
        benchmark(inspector, a.benchmark, a.runs)
    elif a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit("Could not read " + a.image)
        r = inspector.inspect(img)
        print("{}  score {:.4f} (threshold {:.4f})  {:.0f} ms{}".format(
            r["verdict"], r["score"], inspector.threshold, r["ms_total"], "" if r["face_found"] else "  (no carton face found)"))
        if a.save_maps:
            out = os.path.splitext(os.path.basename(a.image))[0] + "_map.jpg"
            cv2.imwrite(out, map_picture(r))
            print("Defect map saved to " + out)
    elif a.watch:
        watch(inspector, a.watch, a.out, a.save_maps)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
