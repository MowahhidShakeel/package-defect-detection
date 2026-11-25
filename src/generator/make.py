"""
Preview the synthetic defects and export a fixed synthetic test set.

Run from the src folder:

    python -m generator.make preview     review sheets, to judge realism by eye
    python -m generator.make export      fixed, seeded synthetic test set from the good TEST crops
    python -m generator.make             both

Outputs
    results/generator/preview_<type>.jpg     each defect at five severities on four cartons
    results/generator/preview_mixed.jpg      random defects as training will see them, with the mask outlined
    results/generator/preview_retouch.jpg    good cartons after `retouch_clean` (must still look good)
    data/synthetic_test/images, masks        the exported test set
    data/synthetic_test/metadata.csv         one row per image: source crop, defect types, severities

Training does NOT read the exported set. During training, defects are made
fresh each time an image is loaded (see DefectGenerator.make_defect).
"""
from __future__ import annotations

import csv
import sys

import cv2
import numpy as np

from config import DATA_DIR, RESULTS_DIR, SEED
from generator.defects import DEFECT_TYPES, DefectGenerator
from viz import caption_bar

CROPS_DIR = DATA_DIR / "crops_aligned"
SYNTHETIC_TEST_DIR = DATA_DIR / "synthetic_test"
GENERATOR_RESULTS_DIR = RESULTS_DIR / "generator"

PREVIEW_SEVERITIES = [0.1, 0.3, 0.5, 0.7, 0.95]
PREVIEW_CARTONS = 4
EXPORT_PER_TYPE = 2          # per good test crop: this many of every single defect type ...
EXPORT_MIXED = 4             # ... plus this many random draws (which include double defects)
EXPORT_SOURCE_CROPS = 80     # good test crops used, spread evenly over the test set


def _load(split: str):
    paths = sorted((CROPS_DIR / split).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No crops in {CROPS_DIR / split}. Run: python -m preprocessing.build_crops")
    return paths


def _label(img: np.ndarray, text: str) -> np.ndarray:
    return caption_bar(img, text, height=22)


def _outline(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    return out


def preview() -> None:
    GENERATOR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gen = DefectGenerator(seed=SEED)
    paths = _load("train_ok")
    picks = [paths[i] for i in np.linspace(0, len(paths) - 1, PREVIEW_CARTONS).astype(int)]
    crops = [cv2.imread(str(p)) for p in picks]
    layouts = [gen.analyse(c) for c in crops]
    unreadable = sum(not l.valid for l in layouts)
    if unreadable:
        print(f"WARNING: no print found in {unreadable} of {len(crops)} preview crops.")

    for name in DEFECT_TYPES:
        rows = []
        for crop, layout in zip(crops, layouts):
            row = [_label(crop, "original")]
            for s in PREVIEW_SEVERITIES:
                sample = gen.apply(crop, layout, name, s)
                row.append(_label(sample.image, f"{name} {s:.2f}") if sample else _label(crop, "no visible change"))
            rows.append(np.hstack(row))
        cv2.imwrite(str(GENERATOR_RESULTS_DIR / f"preview_{name}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    rows = []
    for _ in range(6):
        row = []
        for _ in range(4):
            i = int(gen.rng.integers(len(crops)))
            sample = gen.make_defect(crops[i], layouts[i])
            row.append(_label(_outline(sample.image, sample.mask), " + ".join(sample.types)) if sample
                       else _label(crops[i], "none"))
        rows.append(np.hstack(row))
    cv2.imwrite(str(GENERATOR_RESULTS_DIR / "preview_mixed.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    row = [np.vstack([_label(c, "original"), _label(gen.retouch_clean(c, l), "retouched, still OK")])
           for c, l in zip(crops, layouts)]
    cv2.imwrite(str(GENERATOR_RESULTS_DIR / "preview_retouch.jpg"), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Preview sheets: {GENERATOR_RESULTS_DIR}")


def export_test_set() -> None:
    for sub in ("images", "masks"):
        d = SYNTHETIC_TEST_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
    gen = DefectGenerator(seed=SEED + 1)
    rows, skipped = [], 0
    sources = _load("test_ok")
    if len(sources) > EXPORT_SOURCE_CROPS:
        sources = [sources[i] for i in np.linspace(0, len(sources) - 1, EXPORT_SOURCE_CROPS).astype(int)]
    for path in sources:
        crop = cv2.imread(str(path))
        layout = gen.analyse(crop)
        if not layout.valid:
            skipped += 1
            continue
        plan = [(t, None) for t in DEFECT_TYPES for _ in range(EXPORT_PER_TYPE)] + [(None, None)] * EXPORT_MIXED
        for k, (name, _) in enumerate(plan):
            sample = (gen.apply(crop, layout, name, float(gen.rng.random())) if name
                      else gen.make_defect(crop, layout))
            if sample is None:
                continue
            stem = f"{path.stem}__{k:02d}_{'-'.join(sample.types)}"
            cv2.imwrite(str(SYNTHETIC_TEST_DIR / "images" / f"{stem}.png"), sample.image)
            cv2.imwrite(str(SYNTHETIC_TEST_DIR / "masks" / f"{stem}.png"), sample.mask)
            rows.append({"image": f"images/{stem}.png", "mask": f"masks/{stem}.png", "source_crop": path.name,
                         "types": "|".join(sample.types),
                         "severity": "|".join(f"{s:.3f}" for s in sample.severity),
                         "defect_pixels": int(np.count_nonzero(sample.mask))})
    with open(SYNTHETIC_TEST_DIR / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for r in rows:
        for t in r["types"].split("|"):
            counts[t] = counts.get(t, 0) + 1
    print(f"Synthetic test set: {len(rows)} images in {SYNTHETIC_TEST_DIR} (good test crops skipped: {skipped})")
    print("Per type: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step not in ("preview", "export", "all"):
        print(__doc__)
        sys.exit(1)
    if step in ("preview", "all"):
        preview()
    if step in ("export", "all"):
        export_test_set()
