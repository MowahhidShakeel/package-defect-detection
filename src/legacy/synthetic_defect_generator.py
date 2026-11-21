import os
import glob
import random
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from config import CROPS_DIR, SYNTHETIC_DIR, DEFECT_TYPES, SEVERITIES, SEED

random.seed(SEED)
np.random.seed(SEED)

def get_ink_mask(img_gray: np.ndarray) -> np.ndarray:
    """Returns binary mask (255 = ink, 0 = background) using Otsu thresholding."""
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove border padding artifacts
    mask[:, -8:] = 0
    return mask

def get_text_rows(ink_mask: np.ndarray):
    """Finds horizontal text row intervals [(y1, y2), ...] via horizontal projection."""
    proj = np.sum(ink_mask > 0, axis=1)
    active = proj > max(2, int(0.05 * ink_mask.shape[1]))
    rows = []
    in_row = False
    start = 0
    for y, val in enumerate(active):
        if val and not in_row:
            in_row = True
            start = y
        elif not val and in_row:
            in_row = False
            if (y - start) >= 4:
                rows.append((start, y))
    if in_row and (len(active) - start) >= 4:
        rows.append((start, len(active)))
    if not rows:
        h = ink_mask.shape[0]
        rows = [(int(h * 0.15), int(h * 0.45)), (int(h * 0.55), int(h * 0.85))]
    return rows

def apply_defect(img: np.ndarray, defect_type: str, severity: int, donor_glyphs: list):
    """
    Applies one of the 6 defect types at severity {1, 2, 3}.
    Returns (defective_img, defect_pixel_mask) where mask is 255 at defective pixels.
    """
    out = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    ink_mask = get_ink_mask(img)
    bg_val = float(np.median(img[ink_mask == 0])) if np.any(ink_mask == 0) else 200.0
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    if defect_type == "fade":
        # Reduce ink contrast toward background value by severity factor
        alpha = {1: 0.45, 2: 0.70, 3: 0.88}[severity]
        out[ink_mask > 0] = out[ink_mask > 0] * (1.0 - alpha) + bg_val * alpha
        gt_mask[ink_mask > 0] = 255

    elif defect_type == "smudge":
        # Local blur plus dark ink blob over part of a word
        ys, xs = np.where(ink_mask > 0)
        if len(xs) > 10:
            idx = random.randint(0, len(xs) - 1)
            cx, cy = int(xs[idx]), int(ys[idx])
        else:
            cx, cy = w // 2, h // 2
        rx = {1: 8, 2: 14, 3: 22}[severity]
        ry = {1: 6, 2: 10, 3: 15}[severity]

        blob_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(blob_mask, (cx, cy), (rx, ry), random.uniform(-25, 25), 0, 360, 255, -1)
        ink_dark = float(np.percentile(img[ink_mask > 0], 15)) if np.any(ink_mask > 0) else 30.0
        soft_mask = cv2.GaussianBlur(blob_mask.astype(np.float32) / 255.0, (9, 9), 3.0)
        blurred_img = cv2.GaussianBlur(out, (9, 9), 2.5)
        out = out * (1.0 - soft_mask) + (0.6 * ink_dark + 0.4 * blurred_img) * soft_mask
        gt_mask = (soft_mask > 0.2).astype(np.uint8) * 255

    elif defect_type == "broken_strokes":
        # Random morphological erosion / stroke dropout on the ink mask
        drop_prob = {1: 0.25, 2: 0.50, 3: 0.75}[severity]
        k_size = {1: 2, 2: 3, 3: 4}[severity]
        noise_grid = cv2.resize(
            (np.random.rand(h // k_size + 1, w // k_size + 1) < drop_prob).astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST
        )
        erode_locs = (ink_mask > 0) & (noise_grid > 0)
        out[erode_locs] = bg_val + np.random.normal(0, 3, size=np.sum(erode_locs))
        gt_mask[erode_locs] = 255

    elif defect_type == "missing_line":
        # Blank one row of the block (partially at sev 1, mostly at sev 2, completely at sev 3)
        rows = get_text_rows(ink_mask)
        y1, y2 = random.choice(rows)
        ys, xs = np.where(ink_mask[y1:y2, :] > 0)
        x_min = int(np.min(xs)) if len(xs) > 0 else 10
        x_max = int(np.max(xs)) if len(xs) > 0 else w - 20
        span = max(10, x_max - x_min)
        erase_frac = {1: 0.45, 2: 0.75, 3: 1.0}[severity]
        ex2 = min(w, x_min + int(span * erase_frac))

        region_ink = ink_mask[y1:y2, x_min:ex2] > 0
        patch = np.full((y2 - y1, ex2 - x_min), bg_val, dtype=np.float32)
        patch += np.random.normal(0, 2.5, patch.shape)
        out[y1:y2, x_min:ex2][region_ink] = patch[region_ink]
        gt_mask[y1:y2, x_min:ex2][region_ink] = 255

    elif defect_type == "character_swap":
        # Paste 1, 2, or 3 digit/character crops from another location/image
        num_swaps = {1: 1, 2: 2, 3: 3}[severity]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask)
        valid_Boxes = [
            stats[i] for i in range(1, num_labels)
            if 4 <= stats[i, cv2.CC_STAT_WIDTH] <= 28 and 8 <= stats[i, cv2.CC_STAT_HEIGHT] <= 40
        ]
        random.shuffle(valid_Boxes)
        for box in valid_Boxes[:num_swaps]:
            bx, by, bw, bh, _ = box
            if donor_glyphs:
                donor = random.choice(donor_glyphs)
                donor_resized = cv2.resize(donor, (bw, bh), interpolation=cv2.INTER_LINEAR)
            else:
                donor_resized = cv2.flip(img[by:by+bh, bx:bx+bw], 0)
            out[by:by+bh, bx:bx+bw] = donor_resized.astype(np.float32)
            gt_mask[by:by+bh, bx:bx+bw] = 255

        if np.sum(gt_mask) == 0:
            # Fallback if connected components didn't isolate single glyph
            bx, by, bw, bh = w // 3, h // 3, 14 * severity, 22
            out[by:by+bh, bx:bx+bw] = cv2.flip(out[by:by+bh, bx:bx+bw], -1)
            gt_mask[by:by+bh, bx:bx+bw] = 255

    elif defect_type == "spacing_shift":
        # Translate one row or a group of characters horizontally/vertically
        rows = get_text_rows(ink_mask)
        y1, y2 = random.choice(rows)
        dx = {1: 6, 2: 12, 3: 22}[severity] * random.choice([-1, 1])
        dy = {1: 2, 2: 4, 3: 7}[severity] * random.choice([-1, 1])

        row_slice = out[y1:y2, :].copy()
        row_ink = ink_mask[y1:y2, :].copy()
        out[y1:y2, :][row_ink > 0] = bg_val

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_slice = cv2.warpAffine(row_slice, M, (w, y2 - y1), borderValue=bg_val)
        shifted_ink = cv2.warpAffine(row_ink, M, (w, y2 - y1), borderValue=0)

        out[y1:y2, :][shifted_ink > 0] = shifted_slice[shifted_ink > 0]
        gt_mask[y1:y2, :][(row_ink > 0) | (shifted_ink > 0)] = 255

    return np.clip(out, 0, 255).astype(np.uint8), gt_mask

def extract_donor_glyphs(crop_paths: list) -> list:
    glyphs = []
    for p in crop_paths[:50]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = get_ink_mask(img)
        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n_lbl):
            bx, by, bw, bh, area = stats[i]
            if 5 <= bw <= 25 and 10 <= bh <= 36 and area > 20:
                glyphs.append(img[by:by+bh, bx:bx+bw].copy())
    return glyphs

def build_synthetic_datasets():
    for split in ["train_ok", "val_ok", "test_ok"]:
        src_dir = CROPS_DIR / split
        crop_paths = sorted(src_dir.glob("*.png"))
        if not crop_paths:
            print(f"[Warning] No crops found in {src_dir}. Run 03_stage1_yolo_locator_and_crop.py first.")
            continue

        donor_glyphs = extract_donor_glyphs(crop_paths)
        out_img_dir = SYNTHETIC_DIR / split / "images"
        out_msk_dir = SYNTHETIC_DIR / split / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_msk_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for cpath in tqdm(crop_paths, desc=f"Synthesizing defects for {split}"):
            clean_img = cv2.imread(str(cpath), cv2.IMREAD_GRAYSCALE)
            if clean_img is None:
                continue

            # For test_ok, generate full stratified benchmark (all 6 types x 3 severities per crop)
            # For train_ok/val_ok, generate 6 random (type, severity) negatives per crop
            pairs = (
                [(dt, sev) for dt in DEFECT_TYPES for sev in SEVERITIES]
                if split == "test_ok"
                else [(dt, random.choice(SEVERITIES)) for dt in DEFECT_TYPES]
            )

            for dt, sev in pairs:
                def_img, mask = apply_defect(clean_img, dt, sev, donor_glyphs)
                fname = f"{cpath.stem}__{dt}__sev{sev}.png"
                cv2.imwrite(str(out_img_dir / fname), def_img)
                cv2.imwrite(str(out_msk_dir / fname), mask)
                records.append({
                    "clean_source": str(cpath),
                    "defect_image": str(out_img_dir / fname),
                    "mask_image": str(out_msk_dir / fname),
                    "defect_type": dt,
                    "severity": sev,
                    "split": split
                })

        pd.DataFrame(records).to_csv(SYNTHETIC_DIR / f"{split}_metadata.csv", index=False)
        print(f"[Synthetic {split}] Generated {len(records)} defective crops + pixel masks.")

if __name__ == "__main__":
    build_synthetic_datasets()