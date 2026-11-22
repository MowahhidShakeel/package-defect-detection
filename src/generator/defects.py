"""
Synthetic print-defect generator.

Every crop is first separated into two layers:

    background  the carton as it would look with no print on it
    ink         how strongly each pixel is darkened by print (0 = none)

A defect is then an edit to the ink: remove some, weaken it, move it or add
more. The picture is rebuilt from the real background, so removed print
leaves real-looking carton behind and moved print leaves no hole.

    gen = DefectGenerator(seed=0)
    layout = gen.analyse(crop_bgr)            # once per image; reuse it
    sample = gen.make_defect(crop_bgr, layout)
    sample.image, sample.mask, sample.types

`sample.mask` marks the pixels that visibly changed (255 = defect).

No project imports: needs only opencv-python and numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEFECT_TYPES = [
    "fade",                # ink too weak, over the whole print or one area
    "partial_character",   # a continuous part of 1 to 3 characters missing
    "nozzle_dropout",      # one or two dot rows missing through a line
    "missing_characters",  # 1 to 4 neighbouring characters missing
    "missing_line",        # one whole line missing
    "cut_off",             # print ends early on one side, or runs off the carton edge
    "line_shift",          # one line displaced relative to the others
    "skew",                # print rotated relative to the carton
    "smudge",              # blot of extra ink with smeared print
    "no_print",            # blank flap
]


@dataclass
class GeneratorConfig:
    # How often each type is drawn. Types left out get weight 1.
    weights: Dict[str, float] = field(default_factory=lambda: {"no_print": 0.4, "missing_line": 0.7})
    p_two_defects: float = 0.2        # chance of applying two different defects to one image
    min_changed_pixels: int = 30      # a defect changing fewer pixels than this is discarded as invisible
    change_threshold: int = 10        # grey levels a pixel must change by to count as defective
    # HSV range of the carton (same values as preprocessing.align.AlignConfig)
    hsv_ranges: Sequence = (((0, 70, 40), (15, 255, 255)), ((165, 70, 40), (180, 255, 255)))
    ink_channel: int = 2              # red in BGR: bright carton, dark ink
    ink_min_strength: int = 18
    expected_lines: int = 3           # text lines in a good print


@dataclass
class Layout:
    """What `analyse` finds in one crop."""
    valid: bool
    background: Optional[np.ndarray] = None       # float32 BGR, print removed
    ink: Optional[np.ndarray] = None              # float32, 0..1 darkening by print
    edit_zone: Optional[np.ndarray] = None        # uint8, print pixels plus a small halo
    carton: Optional[np.ndarray] = None           # float32 0..1, soft mask of the carton face
    block: Optional[Tuple[int, int, int, int]] = None      # x1, y1, x2, y2 of the print
    lines: List[Tuple[int, int]] = field(default_factory=list)                 # (y1, y2) per text line
    chars: List[List[Tuple[int, int]]] = field(default_factory=list)           # (x1, x2) per character, per line
    ink_strength: float = 0.5                     # typical darkening at the middle of a stroke
    clean_texture: Optional[np.ndarray] = None    # float32 BGR fine grain of the plain carton


@dataclass
class DefectSample:
    image: np.ndarray          # uint8 BGR
    mask: np.ndarray           # uint8, 255 where the print visibly changed
    types: List[str]
    severity: List[float]      # 0..1 for each type applied


def _runs(flags: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Start/end of every stretch of True values at least `min_len` long."""
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def _soft(mask: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) if sigma > 0 else mask.astype(np.float32)


class DefectGenerator:
    def __init__(self, seed: Optional[int] = None, cfg: Optional[GeneratorConfig] = None):
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(seed)

    # ==========================================================================
    # Layer separation
    # ==========================================================================
    def analyse(self, crop_bgr: np.ndarray) -> Layout:
        cfg = self.cfg
        h, w = crop_bgr.shape[:2]
        img = crop_bgr.astype(np.float32)

        # --- Carton face (red)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        red = np.zeros((h, w), np.uint8)
        for lo, hi in cfg.hsv_ranges:
            red |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
        inner = cv2.erode(red, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

        # --- Dark print on the carton
        chan = crop_bgr[:, :, cfg.ink_channel]
        blackhat = cv2.morphologyEx(chan, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
        otsu, _ = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = ((blackhat > max(float(otsu), cfg.ink_min_strength)) & (inner > 0)).astype(np.uint8) * 255

        not_plain = cv2.dilate(dark, np.ones((11, 11), np.uint8))   # every dark mark and its surroundings, fold line included
        # Drop long thin lines (the carton fold) before looking for text
        joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_WIDTH] > 0.5 * w and stats[i, cv2.CC_STAT_HEIGHT] < 0.07 * h:
                dark[labels == i] = 0

        # The print is the cluster holding the most ink
        cluster = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 13)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cluster)
        if n <= 1:
            return Layout(False)
        amount = [int(np.count_nonzero(dark[labels == i])) for i in range(1, n)]
        best = 1 + int(np.argmax(amount))
        if amount[best - 1] < 150:
            return Layout(False)
        ink_px = np.where(labels == best, dark, 0).astype(np.uint8)

        rows = ink_px.sum(axis=1) / 255.0
        cols = ink_px.sum(axis=0) / 255.0
        ys, xs = np.flatnonzero(rows >= 0.08 * rows.max()), np.flatnonzero(cols >= 0.05 * cols.max())
        bx1, bx2, by1, by2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        ink_px[:max(0, by1 - 2)] = 0
        ink_px[by2 + 2:] = 0
        ink_px[:, :max(0, bx1 - 2)] = 0
        ink_px[:, bx2 + 2:] = 0

        # --- Text lines and characters
        row_sum = np.convolve(ink_px[:, bx1:bx2].sum(axis=1) / 255.0, np.ones(3) / 3, mode="same")
        lines = []
        for run in _runs(row_sum > 0.15 * row_sum.max(), min_len=8):
            lines += self._split_lines(row_sum, run, (by2 - by1) / float(cfg.expected_lines))
        if not lines:
            return Layout(False)
        chars = []
        for y1, y2 in lines:
            col_sum = ink_px[y1:y2].sum(axis=0) / 255.0
            runs = _runs(col_sum > 0.5, min_len=3)
            if runs:
                typical = float(np.median([b - a for a, b in runs]))
                split = []
                for a, b in runs:
                    parts = max(1, int(round((b - a) / max(typical, 1.0)))) if (b - a) > 1.7 * typical else 1
                    step = (b - a) / parts
                    split += [(int(round(a + k * step)), int(round(a + (k + 1) * step))) for k in range(parts)]
                runs = split
            chars.append(runs)

        # --- Background: what the carton looks like under the print
        zone = cv2.dilate(ink_px, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
        # Lighting under the print: a smooth surface fitted through the plain carton
        # around it. Fine grain is then copied in from plain carton nearby.
        clean = (cv2.erode(inner, np.ones((5, 5), np.uint8)) > 0) & (zone == 0) & (not_plain == 0)
        weight = clean.astype(np.float32)
        estimates = []
        for sigma in (9.0, 30.0):          # the wide pass covers spots far from any plain carton
            num = cv2.GaussianBlur(img * weight[..., None], (0, 0), sigma)
            den = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
            estimates.append((num / np.maximum(den, 1e-4), den))
        trust = np.clip(estimates[0][1] / 0.35, 0.0, 1.0)      # how much plain carton the narrow pass saw
        lighting = trust * estimates[0][0] + (1.0 - trust) * estimates[1][0]
        texture = img - cv2.GaussianBlur(img, (0, 0), 2.0)      # fine grain only
        clean_texture = self._donor_texture(texture, clean, (by2 - by1) + 12)
        zm = zone > 0
        # Blend the fill in gradually at the rim of the zone so no outline is left
        rim = cv2.GaussianBlur(cv2.dilate(zone, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0, (0, 0), 2.5)
        rim = np.where(inner > 0, rim, 0.0)[..., None]
        fill = np.clip(lighting + clean_texture, 0, 255)
        background = img + (fill - img) * rim

        b_chan = np.maximum(background[:, :, cfg.ink_channel], 1.0)
        ink = np.clip(1.0 - img[:, :, cfg.ink_channel] / b_chan, 0.0, 1.0) * zm
        strength = float(np.percentile(ink[ink_px > 0], 85)) if np.any(ink_px) else 0.5

        carton = _soft(cv2.erode(red, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) / 255.0, 1.0)
        return Layout(True, background, ink.astype(np.float32), zone, carton, (bx1, by1, bx2, by2),
                      lines, chars, max(strength, 0.15), clean_texture)

    @staticmethod
    def _split_lines(row_sum: np.ndarray, run: Tuple[int, int], line_h: float) -> List[Tuple[int, int]]:
        """Blur can join neighbouring text lines into one stretch of rows. Split a
        stretch that is too tall for one line at the emptiest row inside it."""
        a, b = run
        if b - a < 1.5 * line_h or b - a < 16:
            return [run]
        inner = row_sum[a + 5:b - 5]
        cut = a + 5 + int(np.argmin(inner))
        if row_sum[cut] > 0.8 * min(row_sum[a:cut].max(), row_sum[cut:b].max()):
            return [run]                      # no real dip: leave it as one line
        return (DefectGenerator._split_lines(row_sum, (a, cut), line_h)
                + DefectGenerator._split_lines(row_sum, (cut, b), line_h))

    def _donor_texture(self, texture: np.ndarray, clean: np.ndarray, shift: int) -> np.ndarray:
        """Fine carton grain for every pixel, taken from plain carton nearby,
        so that filled-in areas are not smoother than their surroundings."""
        h, w = clean.shape
        if clean.any():                       # keep the grain within its normal range: no stray edges
            limit = 3.0 * texture[clean].std(axis=0)
            texture = np.clip(texture, -limit, limit)
        out = np.zeros_like(texture)
        todo = np.ones((h, w), bool)
        for dy, dx in [(shift, 0), (-shift, 0), (shift, 23), (shift, -23), (2 * shift, 0), (shift // 2, 41)]:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            moved = cv2.warpAffine(texture, M, (w, h), flags=cv2.INTER_NEAREST)
            ok = cv2.warpAffine(clean.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
            take = todo & ok
            out[take] = moved[take]
            todo &= ~take
        if todo.any() and clean.any():        # anything still unfilled: draw at random from plain carton
            pool = texture[clean]
            out[todo] = pool[self.rng.integers(0, len(pool), int(todo.sum()))]
        return out

    # ==========================================================================
    # Rebuilding the picture
    # ==========================================================================
    @staticmethod
    def _render(img: np.ndarray, layout: Layout, remove: Optional[np.ndarray], add: Optional[np.ndarray]) -> np.ndarray:
        """remove: 0..1 share of the existing print to take away at each pixel.
        add: 0..1 extra darkening by new or moved ink."""
        out = img.astype(np.float32)
        if remove is not None:
            out += (layout.background - out) * np.clip(remove, 0, 1)[..., None]
        if add is not None:
            out *= (1.0 - np.clip(add, 0, 0.95))[..., None]
        return out

    # ==========================================================================
    # The defects. Each returns (remove map, add map); either may be None.
    # s is the severity, 0 (mild) to 1 (severe).
    # ==========================================================================
    def _fade(self, L: Layout, s: float):
        h, w = L.ink.shape
        strength = 0.35 + 0.6 * s
        if self.rng.random() < 0.6:
            return np.full((h, w), strength, np.float32), None
        x1, y1, x2, y2 = L.block
        cx, cy = self.rng.uniform(x1, x2), self.rng.uniform(y1, y2)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rx, ry = self.rng.uniform(0.15, 0.45) * (x2 - x1), self.rng.uniform(0.5, 1.2) * (y2 - y1)
        patch = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        return (strength * np.clip(patch * 1.6, 0, 1)).astype(np.float32), None

    def _char_boxes(self, L: Layout) -> List[Tuple[int, int, int, int]]:
        return [(x1, y1, x2, y2) for (y1, y2), row in zip(L.lines, L.chars) for x1, x2 in row]

    def _partial_character(self, L: Layout, s: float):
        boxes = self._char_boxes(L)
        if not boxes:
            return None, None
        h, w = L.ink.shape
        remove = np.zeros((h, w), np.float32)
        count = int(self.rng.integers(1, 4))
        for i in self.rng.choice(len(boxes), size=min(count, len(boxes)), replace=False):
            x1, y1, x2, y2 = boxes[i]
            share = 0.25 + 0.45 * s + self.rng.uniform(-0.05, 0.05)          # part of the character lost
            angle = self.rng.choice([0, 90, 180, 270]) + self.rng.uniform(-25, 25)
            yy, xx = np.mgrid[y1 - 1:y2 + 1, x1 - 1:x2 + 1].astype(np.float32)
            along = (xx - (x1 + x2) / 2) * math.cos(math.radians(angle)) + (yy - (y1 + y2) / 2) * math.sin(math.radians(angle))
            cut = np.quantile(along, 1.0 - np.clip(share, 0.15, 0.8))
            ys, xs = slice(max(0, y1 - 1), min(h, y2 + 1)), slice(max(0, x1 - 1), min(w, x2 + 1))
            piece = (along > cut).astype(np.float32)[: ys.stop - ys.start, : xs.stop - xs.start]
            remove[ys, xs] = np.maximum(remove[ys, xs], piece)
        return _soft(remove, 0.6), None

    def _nozzle_dropout(self, L: Layout, s: float):
        h, w = L.ink.shape
        y1, y2 = L.lines[int(self.rng.integers(len(L.lines)))]
        pitch = (y2 - y1) / 7.0                                              # the font is 7 dot rows tall
        rows = 1 if s < 0.5 else 2
        first = int(self.rng.integers(0, 7 - rows + 1))
        ya, yb = y1 + first * pitch, y1 + (first + rows) * pitch
        x1, _, x2, _ = L.block
        if self.rng.random() < 0.3:                                          # sometimes only part of the line
            span = self.rng.uniform(0.35, 0.8) * (x2 - x1)
            x1 = int(self.rng.uniform(x1, x2 - span))
            x2 = int(x1 + span)
        remove = np.zeros((h, w), np.float32)
        remove[int(round(ya)):max(int(round(yb)), int(round(ya)) + 2), max(0, x1 - 3):x2 + 3] = 0.75 + 0.25 * s
        return _soft(remove, 0.5), None

    def _missing_characters(self, L: Layout, s: float):
        candidates = [i for i, row in enumerate(L.chars) if row]
        if not candidates:
            return None, None
        li = int(self.rng.choice(candidates))
        row, (y1, y2) = L.chars[li], L.lines[li]
        count = min(len(row), 1 + int(round(3 * s)))
        start = int(self.rng.integers(0, len(row) - count + 1))
        remove = np.zeros(L.ink.shape, np.float32)
        remove[max(0, y1 - 3):y2 + 3, max(0, row[start][0] - 2):row[start + count - 1][1] + 2] = 1.0
        return _soft(remove, 0.6), None

    def _missing_line(self, L: Layout, s: float):
        y1, y2 = L.lines[int(self.rng.integers(len(L.lines)))]
        remove = np.zeros(L.ink.shape, np.float32)
        remove[max(0, y1 - 2):y2 + 2] = 1.0
        return _soft(remove, 1.2), None

    def _moved_ink(self, L: Layout, M: np.ndarray, part: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = L.ink.shape
        ink = L.ink if part is None else L.ink * part
        return cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR) * L.carton

    def _cut_off(self, L: Layout, s: float):
        h, w = L.ink.shape
        x1, y1, x2, y2 = L.block
        lost = (0.12 + 0.4 * s) * (x2 - x1)                                  # width of print that disappears
        right = self.rng.random() < 0.7
        if self.rng.random() < 0.5:
            # The whole print sits too far to one side and runs off the carton edge
            band = L.carton[y1:y2] > 0.5
            cols = np.flatnonzero(band.mean(axis=0) > 0.5)
            if len(cols):
                dx = (cols.max() - x2 + lost) if right else (cols.min() - x1 - lost)
                if abs(dx) >= 4:
                    M = np.float32([[1, 0, dx], [0, 1, 0]])
                    return np.ones(L.ink.shape, np.float32), self._moved_ink(L, M)
        # The print simply stops early
        remove = np.zeros((h, w), np.float32)
        if right:
            remove[:, int(x2 - lost):] = 1.0
        else:
            remove[:, :int(x1 + lost)] = 1.0
        return _soft(remove, 0.5), None

    def _line_shift(self, L: Layout, s: float):
        h, w = L.ink.shape
        y1, y2 = L.lines[int(self.rng.integers(len(L.lines)))]
        part = np.zeros((h, w), np.float32)
        part[max(0, y1 - 2):y2 + 2] = 1.0
        part = _soft(part, 1.2)
        dx = (5 + 22 * s) * self.rng.choice([-1, 1])
        dy = self.rng.uniform(0, 2 + 5 * s) * self.rng.choice([-1, 1])
        return part, self._moved_ink(L, np.float32([[1, 0, dx], [0, 1, dy]]), part)

    def _skew(self, L: Layout, s: float):
        x1, y1, x2, y2 = L.block
        angle = (3.0 + 8.0 * s) * self.rng.choice([-1, 1])
        M = cv2.getRotationMatrix2D(((x1 + x2) / 2.0, (y1 + y2) / 2.0), angle, 1.0).astype(np.float32)
        return np.ones(L.ink.shape, np.float32), self._moved_ink(L, M)

    def _smudge(self, L: Layout, s: float):
        h, w = L.ink.shape
        ys, xs = np.nonzero(L.ink > 0.5 * L.ink_strength)
        if len(xs) == 0:
            return None, None
        k = int(self.rng.integers(len(xs)))
        cx, cy = int(xs[k]), int(ys[k])
        rx, ry = 7 + 22 * s * self.rng.uniform(0.7, 1.2), 5 + 10 * s * self.rng.uniform(0.7, 1.2)
        # Irregular blot: an ellipse with a ragged edge
        blot = np.zeros((h, w), np.float32)
        cv2.ellipse(blot, (cx, cy), (int(rx), int(ry)), float(self.rng.uniform(-30, 30)), 0, 360, 1.0, -1)
        ragged = cv2.GaussianBlur(self.rng.random((h, w)).astype(np.float32), (0, 0), 2.5)
        ragged = (ragged - ragged.min()) / max(float(np.ptp(ragged)), 1e-6)
        blot = _soft(_soft(blot, 2.0) * (0.55 + 0.9 * ragged) > 0.5, 1.6)
        # Print inside and beside the blot is dragged sideways
        length = int(9 + 16 * s) | 1
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        R = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), float(self.rng.uniform(-20, 20)), 1.0)
        kernel = cv2.warpAffine(kernel, R, (length, length))
        kernel /= max(float(kernel.sum()), 1e-6)
        smeared = cv2.filter2D(L.ink, -1, kernel)
        zone = np.zeros((h, w), np.float32)
        cv2.ellipse(zone, (cx, cy), (int(1.8 * rx), int(1.8 * ry)), 0, 0, 360, 1.0, -1)
        zone = _soft(zone, 2.5)
        new_ink = np.maximum(L.ink, 1.5 * smeared)
        add = zone * new_ink + blot * L.ink_strength * self.rng.uniform(0.6, 1.05)
        return zone, np.clip(add, 0, min(0.95, 1.15 * L.ink_strength)) * L.carton

    def _no_print(self, L: Layout, s: float):
        return np.ones(L.ink.shape, np.float32), None

    # ==========================================================================
    # Public API
    # ==========================================================================
    def apply(self, crop_bgr: np.ndarray, layout: Layout, defect: str, severity: float) -> Optional[DefectSample]:
        """Apply one named defect at a chosen severity. None if it left no visible change."""
        return self._finish(crop_bgr, layout, [(defect, float(severity))])

    def make_defect(self, crop_bgr: np.ndarray, layout: Optional[Layout] = None,
                    types: Optional[Sequence[str]] = None) -> Optional[DefectSample]:
        """A random defect (sometimes two) at random severity. None if the crop has
        no recognisable print to edit, or nothing visible resulted after a few tries."""
        layout = layout or self.analyse(crop_bgr)
        if not layout.valid:
            return None
        types = list(types or DEFECT_TYPES)
        p = np.array([self.cfg.weights.get(t, 1.0) for t in types], dtype=np.float64)
        p /= p.sum()
        for _ in range(5):
            chosen = [str(self.rng.choice(types, p=p))]
            if len(types) > 1 and chosen[0] != "no_print" and self.rng.random() < self.cfg.p_two_defects:
                second = str(self.rng.choice(types, p=p))
                if second not in (chosen[0], "no_print"):
                    chosen.append(second)
            sample = self._finish(crop_bgr, layout, [(t, float(self.rng.random())) for t in chosen])
            if sample is not None:
                return sample
        return None

    def _finish(self, crop_bgr: np.ndarray, layout: Layout, plan: List[Tuple[str, float]]) -> Optional[DefectSample]:
        if not layout.valid:
            return None
        out = crop_bgr.astype(np.float32)
        for name, s in plan:
            remove, add = getattr(self, "_" + name)(layout, s)
            if remove is None and add is None:
                return None
            out = self._render(out, layout, remove, add)
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)

        ch = self.cfg.ink_channel
        changed = (np.abs(out[:, :, ch].astype(np.int16) - crop_bgr[:, :, ch].astype(np.int16)) > self.cfg.change_threshold)
        mask = cv2.morphologyEx(changed.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if int(np.count_nonzero(mask)) < self.cfg.min_changed_pixels:
            return None
        return DefectSample(out, mask, [n for n, _ in plan], [s for _, s in plan])

    def retouch_clean(self, crop_bgr: np.ndarray, layout: Optional[Layout] = None) -> np.ndarray:
        """For GOOD images during training: rebuild a patch of plain carton with
        the same fill used when print is removed, and leave the print alone.
        The result is still a good carton. Mixing these in stops a model from
        learning 'filled-in carton means defect' instead of the defect itself."""
        layout = layout or self.analyse(crop_bgr)
        if not layout.valid:
            return crop_bgr.copy()
        h, w = crop_bgr.shape[:2]
        free = ((layout.carton > 0.9) & (layout.edit_zone == 0)).astype(np.uint8)
        ys, xs = np.nonzero(cv2.erode(free, np.ones((9, 9), np.uint8)))
        if len(xs) == 0:
            return crop_bgr.copy()
        k = int(self.rng.integers(len(xs)))
        patch = np.zeros((h, w), np.uint8)
        cv2.ellipse(patch, (int(xs[k]), int(ys[k])), (int(self.rng.integers(6, 28)), int(self.rng.integers(4, 12))),
                    float(self.rng.uniform(0, 180)), 0, 360, 255, -1)
        patch &= free * 255
        fill = cv2.inpaint(crop_bgr, patch, 5, cv2.INPAINT_TELEA).astype(np.float32) + layout.clean_texture
        weight = _soft(patch / 255.0, 0.8)[..., None]
        out = crop_bgr.astype(np.float32) * (1 - weight) + np.clip(fill, 0, 255) * weight
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
