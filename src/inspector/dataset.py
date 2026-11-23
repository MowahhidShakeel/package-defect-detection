"""
Training and validation data for the inspector.

Training images are made fresh on every load from the good training crops:

    half  : a random synthetic defect, with its pixel mask
    half  : a good carton. A third are untouched, a third 'retouched' (a
            patch of plain carton rebuilt) and a third 'redrawn' (the print
            lifted and drawn back in place), so that neither rebuilt carton
            nor redrawn print is a clue to the answer

Then a small random shift, zoom and brightness change is applied.

The validation set is built once from the good VALIDATION crops with a fixed
seed, so every epoch and every model is scored on the same images.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA_DIR
from generator.defects import DefectGenerator
from inspector.model import to_tensor

CROPS_DIR = DATA_DIR / "crops_aligned"

P_DEFECT = 0.5
P_RETOUCH = 0.5
VAL_DEFECTS_PER_CROP = 12
VAL_GOOD_PER_CROP = 4
VAL_CROPS = 60               # good validation crops used for the per-epoch validation set
THRESHOLD_CROPS = 300        # good validation crops used to set the threshold


def crop_paths(split: str, limit: int = 0, seed: int = 0) -> list:
    """Paths of a split's crops. With `limit`, an evenly spread, repeatable
    sample of that many (crops are named in capture order)."""
    paths = sorted((CROPS_DIR / split).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No crops in {CROPS_DIR / split}. Run: python -m preprocessing.build_crops")
    if limit and len(paths) > limit:
        offset = np.random.default_rng(seed).integers(0, len(paths) // limit)
        paths = [paths[int(i)] for i in (np.linspace(0, len(paths) - 1 - offset, limit) + offset)]
    return paths


def load_crops(split: str, limit: int = 0, seed: int = 0) -> Tuple[List[np.ndarray], List[str]]:
    paths = crop_paths(split, limit, seed)
    return [cv2.imread(str(p)) for p in paths], [p.name for p in paths]


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Small changes a good carton could show anyway. Nothing here may create
    or hide a defect: no blur, no contrast loss, no rotation, no flips."""
    h, w = img.shape[:2]
    scale = rng.uniform(0.97, 1.03)
    tx, ty = rng.uniform(-8, 8), rng.uniform(-5, 5)
    M = np.float32([[scale, 0, tx + (1 - scale) * w / 2], [0, scale, ty + (1 - scale) * h / 2]])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    out = img.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-6, 6)
    out += rng.normal(0, rng.uniform(0, 1.5), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), mask


def _pack(img: np.ndarray, mask: np.ndarray, label: int):
    return to_tensor(img), torch.from_numpy(mask > 0).float().unsqueeze(0), torch.tensor(float(label))


class TrainSet(Dataset):
    """Each item is a random good training crop turned into a training sample.
    An epoch is `samples_per_epoch` such draws, so its length does not depend
    on how many crops there are; over a whole run every crop is still drawn
    many times. Crops are read and separated into layers when drawn, which
    keeps memory flat however large the dataset is."""

    def __init__(self, seed: int, samples_per_epoch: int = 1500, redraw_ok: bool = True):
        self.redraw_ok = redraw_ok
        self.paths = crop_paths("train_ok")
        self.gen = DefectGenerator(seed=seed)
        self.rng = np.random.default_rng(seed + 1000)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        crop = cv2.imread(str(self.paths[int(self.rng.integers(len(self.paths)))]))
        layout = self.gen.analyse(crop)          # a crop with no findable print is served as a good carton
        blank = np.zeros(crop.shape[:2], np.uint8)
        if self.rng.random() < P_DEFECT:
            sample = self.gen.make_defect(crop, layout)
            if sample is not None:
                return _pack(*augment(sample.image, sample.mask, self.rng), 1)
        roll = self.rng.random()
        if self.redraw_ok and roll < 1 / 3:
            img = self.gen.redraw_clean(crop, layout)
        elif roll < (2 / 3 if self.redraw_ok else P_RETOUCH):
            img = self.gen.retouch_clean(crop, layout)
        else:
            img = crop
        return _pack(*augment(img, blank, self.rng), 0)


def build_validation_set(seed: int = 12345):
    """Fixed list of (image uint8 BGR, mask uint8, label, kind) from the good
    validation crops. `kind` is 'good', 'retouched' or the defect type(s)."""
    crops, _ = load_crops("val_ok", limit=VAL_CROPS, seed=seed)
    gen = DefectGenerator(seed=seed)
    items = []
    for crop in crops:
        layout = gen.analyse(crop)
        blank = np.zeros(crop.shape[:2], np.uint8)
        for k in range(VAL_GOOD_PER_CROP):
            if k % 2 == 0:
                items.append((crop, blank, 0, "good"))
            else:
                items.append((gen.retouch_clean(crop, layout), blank, 0, "retouched"))
        for _ in range(VAL_DEFECTS_PER_CROP):
            sample = gen.make_defect(crop, layout)
            if sample is not None:
                items.append((sample.image, sample.mask, 1, "+".join(sample.types)))
    return items


def good_validation_variants(seed: int = 54321, per_crop: int = 4) -> List[np.ndarray]:
    """Good validation crops under the small shifts, zooms and brightness
    changes a good carton may show, half of them retouched. Used to find how
    high a good carton can score, which sets the OK / NOT OK threshold."""
    crops, _ = load_crops("val_ok", limit=THRESHOLD_CROPS, seed=seed)
    gen = DefectGenerator(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for crop in crops:
        layout = gen.analyse(crop)
        blank = np.zeros(crop.shape[:2], np.uint8)
        out.append(crop)
        for k in range(per_crop - 1):
            img = gen.retouch_clean(crop, layout) if k % 2 else crop
            out.append(augment(img, blank, rng)[0])
    return out


class ListSet(Dataset):
    """Wraps a fixed list of (image, mask, label, ...) for a DataLoader."""

    def __init__(self, items):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img, mask, label = self.items[idx][:3]
        return _pack(img, mask, label)
