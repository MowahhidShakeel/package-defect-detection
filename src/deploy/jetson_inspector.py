#!/usr/bin/env python3
"""
Inspection on the Jetson Nano: the engine and the command line.

The Jetson needs only the bundle folder made by `python -m deploy.export`
on the workstation:

    model.onnx            the network
    bundle.json           threshold, crop region, alignment settings
    jetson_inspector.py   this file
    jetson_gui.py         the demo window (imports this file)

Nothing here imports the rest of the project and no PyTorch is needed: only
numpy and OpenCV (both come with JetPack), plus ONE of
    - tensorrt + pycuda   (fast; build model.engine first, see below)
    - onnxruntime         (simple; slower)

It is written for Python 3.6, the version on the original Jetson Nano.

Build the TensorRT engine once, on the Jetson itself:

    /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

Then, for the window:

    python3 jetson_gui.py --bundle .

or, without a screen:

    python3 jetson_inspector.py --bundle . --image some_carton.bmp        one image
    python3 jetson_inspector.py --bundle . --benchmark some_carton.bmp    speed of each stage
    python3 jetson_inspector.py --bundle . --folder some_folder --out results
    python3 jetson_inspector.py --bundle . --watch /srv/ftp/incoming --out results

--watch inspects every image that arrives in the folder (the camera's FTP
target) and moves it to <out>/ok or <out>/not_ok. --folder inspects the
images already in a folder and leaves them where they are. Both write
<out>/log.csv and, for every NOT OK carton, a picture with the defect map
to <out>/maps.

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
LOG_COLUMNS = ["time", "file", "verdict", "confidence_pct", "score", "threshold", "carton_found",
               "ms_total", "ms_read", "ms_crop", "ms_model", "picture"]


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
    """Returns (level image, face rectangle or None, the 2x3 transform that was applied)."""
    h, w = img.shape[:2]
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    found = cv2.findContours(_carton_mask(img, cfg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]          # OpenCV 3 and 4 differ here
    if not contours:
        return img, None, identity
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg["min_face_area_frac"] * h * w:
        return img, None, identity

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
        M = identity
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
    return aligned, face, M


def label_crop(img, bundle):
    """Returns (crop BGR uint8, carton face found?, the crop's four corners in the camera image)."""
    aligned, face, M_align = align_carton(img, bundle["align"])
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
    box = np.array([[x1, y1], [x1 + bw, y1], [x1 + bw, y1 + bh], [x1, y1 + bh]])
    quad = _apply_affine(box, cv2.invertAffineTransform(M_align))
    return crop, found, quad


# ==============================================================================
# Stage 2: the model. Input 1 x 3 x H x W float32, RGB, 0..255. Output: defect map 0..1.
# ==============================================================================
class OnnxRuntimeBackend(object):
    name = "ONNX Runtime"

    def __init__(self, onnx_path):
        import onnxruntime as ort
        wanted = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.name = "ONNX Runtime, " + self.session.get_providers()[0].replace("ExecutionProvider", "")

    def __call__(self, x):
        return self.session.run(None, {self.input_name: x})[0]

    def close(self):
        pass


class TensorRTBackend(object):
    """TensorRT 7 / 8 (JetPack 4.x). NOT tested by the author: there was no Jetson to test on.

    The CUDA context belongs to the thread that creates this object, so create
    it, use it and close it in the same thread (the demo window does this in
    its worker thread)."""
    name = "TensorRT"

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        cuda.init()
        self.cuda = cuda
        self.ctx = cuda.Device(0).make_context()
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

    def close(self):
        if self.ctx is not None:
            self.device, self.host = [], []
            self.context = None
            self.engine = None
            self.ctx.pop()
            self.ctx = None


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
# Stage 3: score, verdict and confidence
# ==============================================================================
def carton_score(prob, top_k):
    """Mean of the top_k highest values of the lightly smoothed defect map
    (same as inspector.model.carton_score)."""
    smooth = cv2.blur(prob, (5, 5), borderType=cv2.BORDER_CONSTANT)
    flat = smooth.ravel()
    return float(np.partition(flat, flat.size - top_k)[flat.size - top_k:].mean())


def confidence(score, threshold):
    """Confidence in the verdict, 50 to 100 %.

    It measures how far the score is from the threshold: 50 % for a score
    right at the threshold, 100 % for a score of 0 (verdict OK) or 1 (verdict
    NOT OK), and a straight line in between. It is a margin, not a measured
    probability: the scores have not been calibrated against real outcomes."""
    if score >= threshold:
        c = 0.5 + 0.5 * (score - threshold) / max(1.0 - threshold, 1e-6)
    else:
        c = 0.5 + 0.5 * (threshold - score) / max(threshold, 1e-6)
    return 100.0 * min(1.0, max(0.5, c))


class Inspector(object):
    def __init__(self, bundle_dir, backend="auto"):
        with open(os.path.join(bundle_dir, "bundle.json")) as f:
            self.bundle = json.load(f)
        self.backend = load_backend(bundle_dir, backend)
        self.threshold = float(self.bundle["threshold"])
        self.top_k = int(self.bundle["top_k"])

    def inspect(self, img_bgr):
        t0 = time.time()
        crop, face_found, quad = label_crop(img_bgr, self.bundle)
        t1 = time.time()
        x = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        prob = np.asarray(self.backend(x), dtype=np.float32).reshape(crop.shape[0], crop.shape[1])
        t2 = time.time()
        score = carton_score(prob, self.top_k)
        t3 = time.time()
        return {
            "verdict": "NOT_OK" if score >= self.threshold else "OK",
            "score": score, "threshold": self.threshold, "confidence": confidence(score, self.threshold),
            "face_found": face_found, "crop": crop, "map": prob, "quad": quad,
            "ms_read": 0.0, "ms_crop": (t1 - t0) * 1000.0, "ms_model": (t2 - t1) * 1000.0,
            "ms_score": (t3 - t2) * 1000.0, "ms_total": (t3 - t0) * 1000.0,
        }

    def inspect_file(self, path):
        """Read and inspect one image file. Returns None if it cannot be read.
        'ms_total' then includes reading the file."""
        t0 = time.time()
        img = cv2.imread(path)
        if img is None:
            return None
        ms_read = (time.time() - t0) * 1000.0
        r = self.inspect(img)
        r["ms_read"] = ms_read
        r["ms_total"] += ms_read
        r["image"] = img
        r["name"] = os.path.basename(path)
        r["path"] = path
        return r

    def warm_up(self, runs=3):
        """The first runs of a model are slow. Do them before the first real carton."""
        blank = np.zeros((480, 512, 3), dtype=np.uint8)
        for _ in range(runs):
            self.inspect(blank)

    def close(self):
        self.backend.close()


# ==============================================================================
# Pictures
# ==============================================================================
def verdict_text(result):
    return "{} {:.1f}%".format("NOT OK" if result["verdict"] == "NOT_OK" else "OK", result["confidence"])


def caption_bar(img, text, height=26):
    """Puts the text in a dark bar above the picture, white on dark, so it is
    readable whatever the picture shows."""
    scale = 0.5
    while scale > 0.3 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > img.shape[1] - 12:
        scale -= 0.02
    bar = np.full((height, img.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def heat_overlay(crop, prob):
    """The defect map laid over the crop. Where the map is near 0 the crop is
    left as it is, so a good carton looks like a plain photo."""
    prob = np.clip(prob.astype(np.float32), 0.0, 1.0)
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    alpha = (0.75 * np.sqrt(prob))[:, :, None]
    return (crop.astype(np.float32) * (1.0 - alpha) + heat * alpha).astype(np.uint8)


def camera_view(result):
    """The camera image with the inspected label region outlined."""
    out = result["image"].copy()
    colour = (0, 0, 255) if result["verdict"] == "NOT_OK" else (0, 200, 0)
    cv2.polylines(out, [np.round(result["quad"]).astype(np.int32).reshape(-1, 1, 2)], True, colour, 2, cv2.LINE_AA)
    return out


def map_picture(result):
    """Label crop and defect map side by side, with the verdict in a caption bar."""
    text = "{}   score {:.3f} (threshold {:.3f})   {:.0f} ms".format(
        verdict_text(result), result["score"], result["threshold"], result["ms_total"])
    if result.get("name"):
        text += "   " + result["name"]
    if not result["face_found"]:
        text += "   no carton found"
    return caption_bar(np.hstack([result["crop"], heat_overlay(result["crop"], result["map"])]), text)


# ==============================================================================
# Saving results
# ==============================================================================
class ResultStore(object):
    """Everything that is kept on disk about the inspections:

        <out>/log.csv        one row per carton
        <out>/maps/          a picture with the defect map for every NOT OK carton
        <out>/not_ok/        the camera image of every NOT OK carton
        <out>/ok/            the camera images of good cartons (only when images are moved)
        <out>/unreadable/    files that could not be opened (only when images are moved)
    """

    def __init__(self, out_dir, save_ok_pictures=False):
        self.out_dir = out_dir
        self.save_ok_pictures = save_ok_pictures
        for sub in ("ok", "not_ok", "unreadable", "maps"):
            if not os.path.isdir(os.path.join(out_dir, sub)):
                os.makedirs(os.path.join(out_dir, sub))
        self.log_path = os.path.join(out_dir, "log.csv")
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                header = f.readline().strip().split(",")
            if header != LOG_COLUMNS:           # a log from an older version: keep it under another name
                os.rename(self.log_path, os.path.join(out_dir, time.strftime("log_before_%Y%m%d_%H%M%S.csv")))
        new = not os.path.exists(self.log_path)
        self._file = open(self.log_path, "a")
        self._writer = csv.writer(self._file)
        if new:
            self._writer.writerow(LOG_COLUMNS)
            self._file.flush()

    def record(self, result, move=False):
        """Logs one result and saves its files. With move=True the camera image
        is moved out of the folder it arrived in. Returns where the image now is."""
        name, path = result["name"], result["path"]
        bad = result["verdict"] == "NOT_OK"
        picture = ""
        if bad or self.save_ok_pictures:
            picture = os.path.join(self.out_dir, "maps", os.path.splitext(name)[0] + ".jpg")
            cv2.imwrite(picture, map_picture(result), [cv2.IMWRITE_JPEG_QUALITY, 90])
        target = os.path.join(self.out_dir, "not_ok" if bad else "ok", name)
        if move:
            shutil.move(path, target)
            path = target
        elif bad:
            shutil.copy(path, target)
        self._writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), name, result["verdict"], "{:.1f}".format(result["confidence"]),
            "{:.4f}".format(result["score"]), "{:.4f}".format(result["threshold"]), int(result["face_found"]),
            "{:.1f}".format(result["ms_total"]), "{:.1f}".format(result["ms_read"]),
            "{:.1f}".format(result["ms_crop"]), "{:.1f}".format(result["ms_model"]),
            os.path.basename(picture)])
        self._file.flush()
        return path

    def unreadable(self, path, move=False):
        if move:
            shutil.move(path, os.path.join(self.out_dir, "unreadable", os.path.basename(path)))

    def close(self):
        self._file.close()


def list_images(folder):
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_TYPES))


def wait_until_written(path, tries=40):
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


# ==============================================================================
# Command line
# ==============================================================================
def _print_result(r):
    print("{}  {:<13}  score {:.3f}  {:.0f} ms{}".format(
        r["name"], verdict_text(r), r["score"], r["ms_total"], "" if r["face_found"] else "  (no carton found)"))


def run_folder(inspector, folder, out_dir, save_ok_pictures):
    store = ResultStore(out_dir, save_ok_pictures)
    counts = {"OK": 0, "NOT_OK": 0}
    for name in list_images(folder):
        r = inspector.inspect_file(os.path.join(folder, name))
        if r is None:
            print("{}  could not be read".format(name))
            continue
        store.record(r, move=False)
        counts[r["verdict"]] += 1
        _print_result(r)
    store.close()
    print("OK: {}  NOT OK: {}  Results: {}".format(counts["OK"], counts["NOT_OK"], out_dir))


def watch(inspector, folder, out_dir, save_ok_pictures):
    store = ResultStore(out_dir, save_ok_pictures)
    print("Watching {} | model: {} | threshold {:.4f} | backend: {}".format(
        folder, inspector.bundle["name"], inspector.threshold, inspector.backend.name))
    print("Press Ctrl+C to stop.")
    counts = {"OK": 0, "NOT_OK": 0}
    try:
        while True:
            names = list_images(folder)
            if not names:
                time.sleep(0.02)
                continue
            for name in names:
                path = os.path.join(folder, name)
                if not wait_until_written(path):
                    continue
                r = inspector.inspect_file(path)
                if r is None:
                    store.unreadable(path, move=True)
                    print("{}  could not be read".format(name))
                    continue
                store.record(r, move=True)
                counts[r["verdict"]] += 1
                _print_result(r)
    except KeyboardInterrupt:
        print("\nStopped. OK: {}  NOT OK: {}  Results: {}".format(counts["OK"], counts["NOT_OK"], out_dir))
    finally:
        store.close()


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
    p.add_argument("--folder", help="inspect every image already in this folder")
    p.add_argument("--watch", metavar="FOLDER", help="inspect every image that arrives in this folder")
    p.add_argument("--out", default="inspection_results", help="where --folder and --watch save their results")
    p.add_argument("--save-ok-pictures", action="store_true", help="keep a defect-map picture for good cartons too")
    a = p.parse_args()

    inspector = Inspector(a.bundle, a.backend)
    try:
        if a.benchmark:
            benchmark(inspector, a.benchmark, a.runs)
        elif a.image:
            r = inspector.inspect_file(a.image)
            if r is None:
                sys.exit("Could not read " + a.image)
            print("{}  score {:.4f} (threshold {:.4f})  {:.0f} ms{}".format(
                verdict_text(r), r["score"], r["threshold"], r["ms_total"], "" if r["face_found"] else "  (no carton found)"))
            out = os.path.splitext(os.path.basename(a.image))[0] + "_map.jpg"
            cv2.imwrite(out, map_picture(r))
            print("Picture with the defect map saved to " + out)
        elif a.folder:
            inspector.warm_up()
            run_folder(inspector, a.folder, a.out, a.save_ok_pictures)
        elif a.watch:
            inspector.warm_up()
            watch(inspector, a.watch, a.out, a.save_ok_pictures)
        else:
            p.print_help()
    finally:
        inspector.close()


if __name__ == "__main__":
    main()
