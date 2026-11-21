"""
Stage 1 of the inspection pipeline: find the printed label on a carton.

Order of operations for every image (training data preparation, evaluation
and live inference all call the same `LabelLocator.process`):

    1. Align    - find the carton face by colour, rotate the frame so the
                  carton is level (classical, no training).
    2. Locate   - a small detector finds the print block on the aligned
                  frame. Class 1 means the carton is upside down; the frame
                  is then turned 180 degrees.
    3. Check    - geometric rules decide the gross faults: no carton face,
                  no print, cut-off print, misplaced print, skewed print.
    4. Crop     - the print block is cut from the aligned frame at native
                  resolution, to a fixed shape, with real carton pixels as
                  margin (no grey padding).

This file has no project imports, so the same code runs on the workstation
and on the Jetson Nano. Requires: opencv-python, numpy. `ultralytics` is
imported only by `YoloDetector`.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

OK_TO_INSPECT = "OK_TO_INSPECT"
NOT_OK = "NOT_OK"

CLS_UPRIGHT = 0
CLS_UPSIDE_DOWN = 1
CLASS_NAMES = ["print_region", "print_region_upside_down"]

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class LocatorConfig:
    # --- Carton face segmentation (HSV). Tapal cartons are red; change these
    #     ranges to reuse the aligner on a product of another colour.
    hsv_ranges: Sequence = (((0, 70, 40), (15, 255, 255)), ((165, 70, 40), (180, 255, 255)))
    min_face_area_frac: float = 0.08      # face must cover at least this share of the frame
    min_align_angle_deg: float = 0.5      # do not resample the image for a smaller tilt

    # --- Detector
    det_conf: float = 0.35

    # --- Ink mask (dark ink on a brighter carton). Channel 2 is red in BGR:
    #     a red carton is bright there and black ink is dark.
    ink_channel: int = 2
    ink_blackhat_frac: float = 0.06       # blackhat kernel as a share of face height
    ink_min_strength: int = 22            # floor on the blackhat threshold (0-255)

    # --- Crop handed to the inspector
    crop_h: int = 128
    crop_w: int = 512
    crop_margin: float = 0.08             # margin around the detected box, as a share of its size

    # --- Gross-check tolerances. `None` means "take from calibration".
    skew_tol_deg: Optional[float] = None  # allowed print angle relative to the carton
    min_ink_frac: Optional[float] = None  # least share of ink pixels inside the box
    geometry_margin: float = 0.5          # widen each calibrated range by this share of its span
    geometry_min_margin: float = 0.02     # ... and by at least this much (face-relative units)


# ==============================================================================
# Small geometry helpers
# ==============================================================================
def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to an (N, 2) array of x, y points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def box_corners(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def bounding_box(points: np.ndarray) -> Box:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def rotate_box_180(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return width - x2, height - y2, width - x1, height - y1


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _long_edge_angle(points4: np.ndarray) -> Tuple[float, float, float]:
    """Angle in degrees (image coordinates, -90 < a <= 90) of the long edge of
    a rotated rectangle given by 4 corner points, plus long and short lengths."""
    e0, e1 = points4[1] - points4[0], points4[2] - points4[1]
    n0, n1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    long_edge, long_len, short_len = (e0, n0, n1) if n0 >= n1 else (e1, n1, n0)
    ang = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    if ang > 90.0:
        ang -= 180.0
    elif ang <= -90.0:
        ang += 180.0
    return ang, long_len, short_len


def rotate_expand(img: np.ndarray, angle_deg: float, center: Tuple[float, float]):
    """Rotate so that a direction at `angle_deg` becomes horizontal. The canvas
    grows so that nothing is cut off. Returns (rotated image, 2x3 matrix)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, 1.0)
    corners = apply_affine(np.array([[0, 0], [w, 0], [w, h], [0, h]]), M)
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    M[0, 2] -= x_min
    M[1, 2] -= y_min
    new_w, new_h = int(math.ceil(x_max - x_min)), int(math.ceil(y_max - y_min))
    out = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, M


# ==============================================================================
# Step 1: carton alignment
# ==============================================================================
@dataclass
class Alignment:
    found: bool
    aligned: np.ndarray                    # level frame (the input frame if no face was found)
    M: np.ndarray                          # 2x3 matrix: original -> aligned coordinates
    face_rect: Optional[Box] = None        # carton face in aligned coordinates
    carton_angle: float = 0.0              # tilt that was removed, in degrees


def carton_mask(img_bgr: np.ndarray, cfg: LocatorConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg.hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    k = max(5, int(round(0.03 * min(img_bgr.shape[:2]))) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return mask


def align_carton(img_bgr: np.ndarray, cfg: LocatorConfig) -> Alignment:
    """Find the carton face and rotate the frame so its long side is level.
    A 180-degree ambiguity remains; the detector resolves it (class 1)."""
    h, w = img_bgr.shape[:2]
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contours, _ = cv2.findContours(carton_mask(img_bgr, cfg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Alignment(False, img_bgr, identity)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg.min_face_area_frac * h * w:
        return Alignment(False, img_bgr, identity)

    rect = cv2.minAreaRect(contour)
    pts = cv2.boxPoints(rect).astype(np.float64)
    angle, _, _ = _long_edge_angle(pts)

    if abs(angle) < cfg.min_align_angle_deg:
        aligned, M = img_bgr, identity
    else:
        aligned, M = rotate_expand(img_bgr, angle, rect[0])
    face = bounding_box(apply_affine(pts, M))
    ah, aw = aligned.shape[:2]
    face = (max(0.0, face[0]), max(0.0, face[1]), min(float(aw), face[2]), min(float(ah), face[3]))
    return Alignment(True, aligned, M, face, angle)


# ==============================================================================
# Ink measurements (used by the checks and by the annotator's suggestions)
# ==============================================================================
def ink_mask(img_bgr: np.ndarray, region: Box, face_rect: Box, cfg: LocatorConfig) -> np.ndarray:
    """Binary mask (255 = ink) of dark print inside `region`."""
    x1, y1, x2, y2 = [int(round(v)) for v in region]
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.uint8)
    chan = img_bgr[y1:y2, x1:x2, cfg.ink_channel] if img_bgr.ndim == 3 else img_bgr[y1:y2, x1:x2]
    k = max(7, int(round(cfg.ink_blackhat_frac * (face_rect[3] - face_rect[1]))) | 1)
    blackhat = cv2.morphologyEx(chan, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    otsu, _ = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(blackhat, max(float(otsu), float(cfg.ink_min_strength)), 255, cv2.THRESH_BINARY)
    return mask


def text_angle(mask: np.ndarray, search_deg: float = 15.0) -> Optional[float]:
    """Angle in degrees of the text lines in an ink mask, by the projection
    profile method: the rotation that makes the row sums most sharply peaked
    is the one that lays the lines level. None if there is too little ink."""
    if np.count_nonzero(mask) < 60:
        return None
    h, w = mask.shape[:2]
    m = (mask > 0).astype(np.float32)
    center = (w / 2.0, h / 2.0)

    def sharpness(angle: float) -> float:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rows = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR).sum(axis=1)
        return float(np.var(rows))

    coarse = np.arange(-search_deg, search_deg + 0.5, 1.0)
    best = float(coarse[int(np.argmax([sharpness(a) for a in coarse]))])
    fine = np.arange(best - 1.0, best + 1.01, 0.2)
    best = float(fine[int(np.argmax([sharpness(a) for a in fine]))])
    return round(best, 2)


def suggest_print_box(img_bgr: np.ndarray, face_rect: Box, cfg: LocatorConfig) -> Optional[Box]:
    """Classical guess at the print block inside the carton face. Used only to
    pre-draw a box in the annotator; never used for a decision."""
    fx1, fy1, fx2, fy2 = face_rect
    fw, fh = fx2 - fx1, fy2 - fy1
    inner = (fx1 + 0.04 * fw, fy1 + 0.06 * fh, fx2 - 0.04 * fw, fy2 - 0.06 * fh)
    mask = ink_mask(img_bgr, inner, face_rect, cfg)
    # Merge dots into one blob per text block; the block with the most ink wins
    kx, ky = max(9, int(0.05 * fw) | 1), max(5, int(0.05 * fh) | 1)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    if n <= 1:
        return None
    ink_per_blob = [int(np.count_nonzero(mask[labels == i])) for i in range(1, n)]
    best = 1 + int(np.argmax(ink_per_blob))
    if ink_per_blob[best - 1] < 0.004 * fw * fh:
        return None
    block = np.where(labels == best, mask, 0)
    # Trim thin tails: keep only rows and columns that carry a fair share of ink
    rows, cols = block.sum(axis=1).astype(np.float64), block.sum(axis=0).astype(np.float64)
    ys = np.nonzero(rows >= 0.10 * rows.max())[0]
    xs = np.nonzero(cols >= 0.05 * cols.max())[0]
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    ox, oy = int(round(inner[0])), int(round(inner[1]))
    pad_x, pad_y = 0.015 * fw, 0.02 * fh
    return (max(fx1, ox + x1 - pad_x), max(fy1, oy + y1 - pad_y),
            min(fx2, ox + x2 + pad_x), min(fy2, oy + y2 + pad_y))


# ==============================================================================
# Step 4: crop
# ==============================================================================
def crop_region(img: np.ndarray, box: Box, cfg: LocatorConfig) -> np.ndarray:
    """Cut the print block to a fixed shape. The box is widened with real image
    pixels until it has the target aspect ratio, so characters keep their
    proportions and no flat padding is introduced."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = bw * (1 + 2 * cfg.crop_margin), bh * (1 + 2 * cfg.crop_margin)
    target = cfg.crop_w / cfg.crop_h
    if bw / max(bh, 1e-6) < target:
        bw = bh * target
    else:
        bh = bw / target
    h, w = img.shape[:2]
    # Slide the window back inside the frame where possible
    cx = min(max(cx, bw / 2.0), max(bw / 2.0, w - bw / 2.0))
    cy = min(max(cy, bh / 2.0), max(bh / 2.0, h - bh / 2.0))
    M = np.array([[cfg.crop_w / bw, 0.0, -(cx - bw / 2.0) * cfg.crop_w / bw],
                  [0.0, cfg.crop_h / bh, -(cy - bh / 2.0) * cfg.crop_h / bh]])
    interp = cv2.INTER_AREA if bw > cfg.crop_w else cv2.INTER_CUBIC
    return cv2.warpAffine(img, M, (cfg.crop_w, cfg.crop_h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


# ==============================================================================
# Detector wrapper
# ==============================================================================
@dataclass
class Detection:
    box: Box
    conf: float
    cls: int


class YoloDetector:
    """Ultralytics YOLO model as a callable: image -> list of Detection."""

    def __init__(self, weights: str | Path, device: Optional[str] = None, imgsz: int = 640):
        from ultralytics import YOLO  # imported here so the rest of the file needs only OpenCV
        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz

    def __call__(self, img_bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(img_bgr, conf=0.05, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [Detection(tuple(float(v) for v in b), float(c), int(k)) for b, c, k in zip(xyxy, conf, cls)]


# ==============================================================================
# The locator
# ==============================================================================
@dataclass
class LocateResult:
    status: str                              # OK_TO_INSPECT or NOT_OK
    reasons: List[str] = field(default_factory=list)
    crop: Optional[np.ndarray] = None        # cfg.crop_h x cfg.crop_w BGR, when a print block was found
    upright: Optional[np.ndarray] = None     # level, right-way-up frame
    box: Optional[Box] = None                # print block in `upright` coordinates
    face_rect: Optional[Box] = None          # carton face in `upright` coordinates
    carton_angle: float = 0.0
    upside_down: bool = False
    det_conf: float = 0.0
    print_skew: Optional[float] = None       # print angle relative to the carton, degrees
    ink_frac: Optional[float] = None
    features: Optional[dict] = None          # box position and size relative to the face
    calibrated: bool = False
    ms: float = 0.0


def face_relative_features(box: Box, face: Box) -> dict:
    fw, fh = max(face[2] - face[0], 1e-6), max(face[3] - face[1], 1e-6)
    return {
        "cx": ((box[0] + box[2]) / 2.0 - face[0]) / fw,
        "cy": ((box[1] + box[3]) / 2.0 - face[1]) / fh,
        "w": (box[2] - box[0]) / fw,
        "h": (box[3] - box[1]) / fh,
    }


class LabelLocator:
    """detector: any callable `image -> list[Detection]` (see YoloDetector).
    stats: output of `calibrate`, or None to run without the geometric checks
    (used while building the calibration itself)."""

    def __init__(self, detector: Callable[[np.ndarray], List[Detection]],
                 cfg: Optional[LocatorConfig] = None, stats: Optional[dict] = None):
        self.detector = detector
        self.cfg = cfg or LocatorConfig()
        self.stats = stats

    @classmethod
    def from_files(cls, weights: str | Path, stats_path: Optional[str | Path] = None,
                   cfg: Optional[LocatorConfig] = None, device: Optional[str] = None):
        stats = None
        if stats_path is not None and Path(stats_path).exists():
            stats = json.loads(Path(stats_path).read_text())
        return cls(YoloDetector(weights, device=device), cfg, stats)

    # ------------------------------------------------------------------ checks
    def _check(self, feats: dict, skew: Optional[float], ink_frac: float) -> List[str]:
        cfg, st, reasons = self.cfg, self.stats, []

        min_ink = cfg.min_ink_frac if cfg.min_ink_frac is not None else (st or {}).get("min_ink_frac")
        if min_ink is not None and ink_frac < min_ink:
            reasons.append("no_print")

        skew_tol = cfg.skew_tol_deg if cfg.skew_tol_deg is not None else (st or {}).get("skew_tol_deg")
        if "no_print" not in reasons and skew_tol is not None and skew is not None and abs(skew) > skew_tol:
            reasons.append("print_skewed")

        if st and "ranges" in st:
            def outside(key):
                lo, hi = st["ranges"][key]
                return feats[key] < lo or feats[key] > hi
            if feats["w"] < st["ranges"]["w"][0]:
                reasons.append("print_cut_off")
            elif outside("w") or outside("h"):
                reasons.append("print_size_abnormal")
            if outside("cx") or outside("cy"):
                reasons.append("print_misplaced")
        return reasons

    # ----------------------------------------------------------------- process
    def process(self, img_bgr: np.ndarray) -> LocateResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        calibrated = self.stats is not None

        def done(res: LocateResult) -> LocateResult:
            res.calibrated = calibrated
            res.ms = (time.perf_counter() - t0) * 1000.0
            return res

        # 1. Align
        al = align_carton(img_bgr, cfg)
        if not al.found:
            return done(LocateResult(NOT_OK, ["no_carton_face"], upright=img_bgr))

        # 2. Locate
        dets = [d for d in self.detector(al.aligned) if d.conf >= cfg.det_conf]
        if not dets:
            return done(LocateResult(NOT_OK, ["no_print"], upright=al.aligned, face_rect=al.face_rect,
                                     carton_angle=al.carton_angle))
        best = max(dets, key=lambda d: d.conf)
        upright, face, box = al.aligned, al.face_rect, best.box
        upside_down = best.cls == CLS_UPSIDE_DOWN
        if upside_down:
            ah, aw = upright.shape[:2]
            upright = cv2.rotate(upright, cv2.ROTATE_180)
            face, box = rotate_box_180(face, aw, ah), rotate_box_180(box, aw, ah)

        # 3. Check
        mask = ink_mask(upright, box, face, cfg)
        ink_frac = float(np.count_nonzero(mask)) / mask.size
        level = text_angle(mask)
        skew = None if level is None else -level   # the text's own angle on the level carton
        feats = face_relative_features(box, face)
        reasons = self._check(feats, skew, ink_frac)

        # 4. Crop
        crop = crop_region(upright, box, cfg)
        return done(LocateResult(
            NOT_OK if reasons else OK_TO_INSPECT, reasons, crop, upright, box, face,
            al.carton_angle, upside_down, best.conf, skew, ink_frac, feats))


# ==============================================================================
# Calibration of the gross-check tolerances from defect-free images
# ==============================================================================
def calibrate(locator: LabelLocator, images: Sequence[np.ndarray]) -> dict:
    """Run the locator on defect-free training images and record how much the
    print block's position, size, angle and ink coverage vary when all is
    well. The checks then reject anything clearly outside that spread."""
    cfg = locator.cfg
    saved, locator.stats = locator.stats, None
    feats, skews, inks = [], [], []
    try:
        for img in images:
            r = locator.process(img)
            if r.box is None:
                continue
            feats.append(r.features)
            inks.append(r.ink_frac)
            if r.print_skew is not None:
                skews.append(abs(r.print_skew))
    finally:
        locator.stats = saved
    if len(feats) < 10:
        raise RuntimeError(f"Calibration needs at least 10 located prints, got {len(feats)}.")

    ranges = {}
    for key in ("cx", "cy", "w", "h"):
        v = np.array([f[key] for f in feats])
        lo, hi = float(v.min()), float(v.max())
        pad = max(cfg.geometry_margin * (hi - lo), cfg.geometry_min_margin)
        ranges[key] = [lo - pad, hi + pad]
    return {
        "n_images": len(feats),
        "ranges": ranges,
        "min_ink_frac": 0.5 * float(np.min(inks)),
        "skew_tol_deg": max(3.0, float(np.max(skews)) + 2.0) if skews else 5.0,
        "observed_max_abs_skew_deg": float(np.max(skews)) if skews else None,
        "config": {k: v for k, v in asdict(cfg).items() if k != "hsv_ranges"},
    }


def draw_result(res: LocateResult) -> np.ndarray:
    """Debug picture: upright frame with the face, the print box and the verdict."""
    out = res.upright.copy()
    if res.face_rect:
        x1, y1, x2, y2 = [int(round(v)) for v in res.face_rect]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 1)
    if res.box:
        x1, y1, x2, y2 = [int(round(v)) for v in res.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ok = res.status == OK_TO_INSPECT
    label = "to inspector" if ok else "NOT OK: " + ", ".join(res.reasons)
    cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    return out
