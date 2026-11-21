"""
Carton alignment and label cropping (no trained model).

    1. Align - find the carton face by colour and rotate the frame level.
    2. Crop  - cut a fixed region of the carton face, the place where the
               print normally sits, to a fixed size.

The crop region is measured once from the annotated good cartons
(see build_crops.py) and stored in a small JSON file. The inspector model
then judges every crop, including blank flaps, wrong carton sides and
cut-off print: nothing is rejected here.

This file has no project imports, so the same code runs on the workstation
and on the Jetson Nano. Requires: opencv-python, numpy.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class AlignConfig:
    # --- Carton face segmentation (HSV). Tapal cartons are red; change these
    #     ranges to reuse the aligner on a product of another colour.
    hsv_ranges: Sequence = (((0, 70, 40), (15, 255, 255)), ((165, 70, 40), (180, 255, 255)))
    min_face_area_frac: float = 0.08      # face must cover at least this share of the frame
    min_align_angle_deg: float = 0.5      # do not resample the image for a smaller tilt

    # --- Ink mask (dark ink on a brighter carton). Channel 2 is red in BGR:
    #     a red carton is bright there and black ink is dark.
    ink_channel: int = 2
    ink_blackhat_frac: float = 0.06       # blackhat kernel as a share of face height
    ink_min_strength: int = 22            # floor on the blackhat threshold (0-255)

    # --- Crop handed to the inspector: least height in pixels. build_crops.py
    #     raises it if the camera resolves more, and picks the width to fit the
    #     measured print block. Both are multiples of 32.
    crop_h: int = 128


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


def carton_mask(img_bgr: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg.hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    k = max(5, int(round(0.03 * min(img_bgr.shape[:2]))) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return mask


def align_carton(img_bgr: np.ndarray, cfg: AlignConfig) -> Alignment:
    """Find the carton face and rotate the frame so its long side is level.
    Colour cannot tell 0 from 180 degrees: an upside-down carton stays upside
    down, and its crop will look abnormal to the inspector."""
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
def ink_mask(img_bgr: np.ndarray, region: Box, face_rect: Box, cfg: AlignConfig) -> np.ndarray:
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


def suggest_print_box(img_bgr: np.ndarray, face_rect: Box, cfg: AlignConfig) -> Optional[Box]:
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
# Step 2: fixed crop region on the carton face
# ==============================================================================
@dataclass
class CropRegion:
    """Where the print sits on the carton face. All lengths are in units of
    the face WIDTH, measured from the face's top-left corner, so characters
    are never stretched when the face height varies a little.

    `fallback_*` describe the usual face rectangle as a share of the frame;
    they are used only when no carton face is found, so that a crop can still
    be handed to the inspector."""
    cu: float                 # centre, across
    cv: float                 # centre, down
    wu: float                 # width
    out_w: int                # crop size in pixels handed to the inspector
    out_h: int
    fallback_face: Tuple[float, float, float, float] = (0.1, 0.25, 0.9, 0.8)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CropRegion":
        d = json.loads(Path(path).read_text())
        d["fallback_face"] = tuple(d["fallback_face"])
        return cls(**d)

    def box_on(self, face: Box) -> Box:
        """The region as a pixel box, given a face rectangle."""
        fw = face[2] - face[0]
        cx, cy = face[0] + self.cu * fw, face[1] + self.cv * fw
        w = self.wu * fw
        h = w * self.out_h / self.out_w
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def to_face_units(box: Box, face: Box) -> Box:
    """A pixel box expressed in face-width units from the face's top-left corner."""
    fw = max(face[2] - face[0], 1e-6)
    return ((box[0] - face[0]) / fw, (box[1] - face[1]) / fw, (box[2] - face[0]) / fw, (box[3] - face[1]) / fw)


def cut(img: np.ndarray, box: Box, out_w: int, out_h: int) -> np.ndarray:
    """Cut `box` from `img` and resample it to out_w x out_h. Parts of the box
    that fall outside the image are filled by repeating the edge pixels."""
    x1, y1, x2, y2 = box
    sx, sy = out_w / max(x2 - x1, 1e-6), out_h / max(y2 - y1, 1e-6)
    M = np.array([[sx, 0.0, -x1 * sx], [0.0, sy, -y1 * sy]])
    interp = cv2.INTER_AREA if sx < 1.0 else cv2.INTER_CUBIC
    return cv2.warpAffine(img, M, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


@dataclass
class LabelCrop:
    crop: np.ndarray            # region.out_h x region.out_w, BGR
    face_found: bool
    carton_angle: float         # tilt removed, degrees
    box: Box                    # crop region in the level frame
    aligned: np.ndarray         # level frame
    M: np.ndarray               # 2x3 matrix: original -> level frame


def extract_label_crop(img_bgr: np.ndarray, region: CropRegion, cfg: Optional[AlignConfig] = None) -> LabelCrop:
    """The whole of Stage 1, for training data, evaluation and live inference
    alike. Always returns a crop: if no carton face is found, the usual face
    position is assumed and the inspector decides."""
    cfg = cfg or AlignConfig()
    al = align_carton(img_bgr, cfg)
    if al.found:
        face = al.face_rect
    else:
        h, w = img_bgr.shape[:2]
        f = region.fallback_face
        face = (f[0] * w, f[1] * h, f[2] * w, f[3] * h)
    box = region.box_on(face)
    return LabelCrop(cut(al.aligned, box, region.out_w, region.out_h), al.found, al.carton_angle, box, al.aligned, al.M)

