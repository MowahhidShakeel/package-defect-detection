"""
Measure where the print sits on good cartons, fix the crop region, and write
the label crops for every image.

Run from the src folder:

    python -m preprocessing.build_crops                  write all crops with the saved region
    python -m preprocessing.build_crops --remeasure      measure the region again from annotations first
    python -m preprocessing.build_crops --measure-only   print the measurements and stop

The crop region is measured once from annotated GOOD TRAINING cartons and
saved. As long as the camera has not moved it is reused, so new images need
no annotation: every image goes through `extract_label_crop`, the same
function the live system will call.

Each crop is then checked automatically: is there a print, and does it lie
clear of the crop's edges? Good cartons that fail are listed in the log and
shown on a contact sheet, for a look by eye.

Outputs
    weights/crop_region.json                     the fixed region (needed at inference)
    data/crops_aligned/<split>/*.png             colour crops for the generator and inspector
    results/preprocessing/crops_log.csv          one row per image
    results/preprocessing/sheet_<group>.jpg      contact sheets to check by eye
    results/preprocessing/sheet_flagged_good.jpg good cartons whose print was not found clear of the edges
"""
from __future__ import annotations

import csv
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

from generator.defects import DefectGenerator
from preprocessing.align import AlignConfig, CropRegion, extract_label_crop, to_face_units, bounding_box, apply_affine
from preprocessing.annotations import (
    CROP_REGION_PATH, CROPS_DIR, PREPROCESSING_RESULTS_DIR, SPLIT_KEYS, HAS_PRINT,
    UprightView, load_annotations, load_manifest,
)

MARGIN = 0.06          # extra room around the print on every side, as a share of the region width
SHEET_MAX = 24         # crops per contact sheet
EDGE_CLEARANCE = 4     # pixels the print must keep from the crop's edges


def _annotated_prints(split_key: str, manifest: dict, annotations: dict, cfg: AlignConfig):
    """Yield (print box in face units, face rect, frame size, print box in pixels)
    for every annotated print in a split whose carton face was found."""
    for item in manifest[split_key]:
        entry = annotations.get(item["path"])
        if not entry or entry["status"] != HAS_PRINT or entry.get("upside_down"):
            continue
        img = cv2.imread(item["path"])
        if img is None:
            continue
        view = UprightView.build(img, cfg, False)
        if not view.found:
            continue
        box = view.box_of(entry)
        yield to_face_units(box, view.face_rect), view.face_rect, view.image.shape[:2], box


def measure_region(cfg: AlignConfig, manifest: dict, annotations: dict) -> CropRegion:
    rows = list(_annotated_prints("train_ok", manifest, annotations, cfg))
    if len(rows) < 10:
        raise RuntimeError(f"Only {len(rows)} annotated good training cartons found; need at least 10.")
    units = np.array([r[0] for r in rows])                      # x1, y1, x2, y2 in face-width units
    faces = np.array([r[1] for r in rows])
    frames = np.array([r[2] for r in rows], dtype=np.float64)   # h, w
    pixels = np.array([r[3] for r in rows])
    face_w = float(np.median(faces[:, 2] - faces[:, 0]))

    # Smallest box that holds every good print, plus a margin, widened to the crop's shape
    x1, y1, x2, y2 = units[:, 0].min(), units[:, 1].min(), units[:, 2].max(), units[:, 3].max()
    pad = MARGIN * (x2 - x1)
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    # Crop width: the multiple of 32 that best fits the print block's shape
    w, h = x2 - x1, y2 - y1
    # Height: at least cfg.crop_h, and never below the camera's own resolution
    out_h = max(cfg.crop_h, int(np.ceil(h * face_w / 32.0)) * 32)
    out_w = max(128, int(round(out_h * w / h / 32.0)) * 32)
    if w / h < out_w / out_h:
        w = h * out_w / out_h
    else:
        h = w * out_h / out_w
    fallback = tuple(float(np.median(faces[:, i] / frames[:, 1 if i % 2 == 0 else 0])) for i in range(4))
    region = CropRegion(float((x1 + x2) / 2), float((y1 + y2) / 2), float(w), out_w, out_h, fallback)

    # ---------------------------------------------------------------- report
    cx, cy = (units[:, 0] + units[:, 2]) / 2, (units[:, 1] + units[:, 3]) / 2
    pw, ph = pixels[:, 2] - pixels[:, 0], pixels[:, 3] - pixels[:, 1]
    scale = out_w / (region.wu * face_w)
    region_h = region.wu * out_h / out_w
    fill = float(np.median((units[:, 2] - units[:, 0]) * (units[:, 3] - units[:, 1])) / (region.wu * region_h))

    print(f"\n=== Print position on {len(rows)} good training cartons ===")
    print(f"Carton face width: median {face_w:.0f} px (range {np.min(faces[:, 2] - faces[:, 0]):.0f} to "
          f"{np.max(faces[:, 2] - faces[:, 0]):.0f})")
    print(f"Print block size : median {np.median(pw):.0f} x {np.median(ph):.0f} px "
          f"(width {pw.min():.0f} to {pw.max():.0f}, height {ph.min():.0f} to {ph.max():.0f})")
    print(f"Print centre moves across the face by {(cx.max() - cx.min()) * face_w:.0f} px sideways "
          f"and {(cy.max() - cy.min()) * face_w:.0f} px up-down (spread: "
          f"{np.std(cx) * face_w:.1f} / {np.std(cy) * face_w:.1f} px std)")
    print(f"\n=== Fixed crop region ===")
    print(f"Covers {region.wu * face_w:.0f} x {region_h * face_w:.0f} px of the level frame, "
          f"written out at {out_w} x {out_h}")
    print(f"Magnification {scale:.2f}x: each text line is about {np.median(ph) / 3 * scale:.0f} px tall in the crop "
          f"({np.median(ph) / 3:.0f} px in the camera image)")
    print(f"The print fills about {100 * fill:.0f}% of the crop")
    if fill < 0.25:
        print("NOTE: the print fills under a quarter of the crop because its position varies a lot. "
              "A crop that follows the print (a locator) would give larger characters.")
    return region


def _inside_share(box, region_box) -> float:
    ix = max(0.0, min(box[2], region_box[2]) - max(box[0], region_box[0]))
    iy = max(0.0, min(box[3], region_box[3]) - max(box[1], region_box[1]))
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return ix * iy / area


def _contact_sheet(crops: list, path) -> None:
    if not crops:
        return
    crops = crops[:SHEET_MAX]
    h, w = crops[0].shape[:2]
    cols = 3
    rows = int(np.ceil(len(crops) / cols))
    sheet = np.full((rows * (h + 6), cols * (w + 6), 3), 255, dtype=np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        sheet[r * (h + 6):r * (h + 6) + h, k * (w + 6):k * (w + 6) + w] = c
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def write_crops(region: CropRegion, cfg: AlignConfig, manifest: dict, annotations: dict) -> None:
    PREPROCESSING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, sheets, times, flagged = [], {}, [], []
    checker = DefectGenerator(seed=0)
    for split_key in SPLIT_KEYS:
        out_dir = CROPS_DIR / split_key
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()

        for idx, item in enumerate(tqdm(manifest[split_key], desc=f"Cropping {split_key}")):
            img = cv2.imread(item["path"])
            if img is None:
                continue
            t0 = time.perf_counter()
            res = extract_label_crop(img, region, cfg)        # model-free, annotation-free
            times.append((time.perf_counter() - t0) * 1000.0)

            name = f"{split_key}_{idx:04d}_{item['session']}.png"
            cv2.imwrite(str(out_dir / name), res.crop)

            # Scoring only: how much of the annotated print landed inside the crop
            entry = annotations.get(item["path"], {})
            inside = ""
            if entry.get("status") == HAS_PRINT and entry.get("quad") and res.face_found:
                inside = round(_inside_share(bounding_box(apply_affine(np.array(entry["quad"]), res.M)), res.box), 4)
            status = entry.get("status", "not_annotated")

            # Automatic check of the crop itself (no annotation needed)
            layout = checker.analyse(res.crop)
            if not layout.valid:
                print_check = "no_print_found"
            else:
                bx1, by1, bx2, by2 = layout.block
                ch, cw = res.crop.shape[:2]
                clear = min(bx1, by1, cw - bx2, ch - by2)
                print_check = "ok" if clear >= EDGE_CLEARANCE else "near_edge"
                if len(layout.lines) != checker.cfg.expected_lines:
                    print_check += f"_{len(layout.lines)}_lines"
            if item["label"] == "OK" and print_check != "ok":
                flagged.append(res.crop)

            rows.append({"split": split_key, "crop": name, "path": item["path"], "label": item["label"],
                         "annotated_status": status, "face_found": res.face_found,
                         "carton_angle": round(res.carton_angle, 2), "print_inside_crop": inside,
                         "print_check": print_check, "ms": round(times[-1], 2)})
            group = split_key if split_key != "test_not_ok" else f"test_not_ok_{status}"
            sheets.setdefault(group, []).append(res.crop)

    with open(PREPROCESSING_RESULTS_DIR / "crops_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for old in PREPROCESSING_RESULTS_DIR.glob("sheet_*.jpg"):
        old.unlink()
    for group, crops in sheets.items():
        # spread the sheet over the whole split instead of showing its first minutes only
        picks = [crops[i] for i in np.linspace(0, len(crops) - 1, min(SHEET_MAX, len(crops))).astype(int)]
        _contact_sheet(picks, PREPROCESSING_RESULTS_DIR / f"sheet_{group}.jpg")
    _contact_sheet(flagged, PREPROCESSING_RESULTS_DIR / "sheet_flagged_good.jpg")

    # ---------------------------------------------------------------- report
    print(f"\n=== Crops written: {len(rows)} ===")
    for split_key in SPLIT_KEYS:
        sub = [r for r in rows if r["split"] == split_key]
        no_face = sum(not r["face_found"] for r in sub)
        ins = [r["print_inside_crop"] for r in sub if r["print_inside_crop"] != ""]
        odd = sum(r["print_check"] != "ok" for r in sub)
        line = f"{split_key:>12}: {len(sub)} crops | carton face not found: {no_face} | print check not ok: {odd}"
        if ins:
            whole = sum(v >= 0.98 for v in ins)
            line += f" | print fully inside the crop: {whole}/{len(ins)} (worst case {100 * min(ins):.0f}% inside)"
        print(line)
    print("\n'Print check not ok' means no print was found, it touches the crop's edge, or it does not have three")
    print("lines. For NOT OK cartons that is expected. For good cartons, look at sheet_flagged_good.jpg: each one is")
    print("either a carton that should not be in the OK folder, or a sign the crop region no longer fits.")
    if flagged:
        print(f"Good cartons flagged: {len(flagged)}. They are marked in the print_check column of crops_log.csv.")
    print(f"\nTime per image on this machine: median {np.median(times):.1f} ms | "
          f"95th percentile {np.percentile(times, 95):.1f} ms. Not a Jetson Nano figure.")
    print(f"Crops: {CROPS_DIR}\nLog and contact sheets: {PREPROCESSING_RESULTS_DIR}")


def main() -> None:
    cfg = AlignConfig()
    manifest, annotations = load_manifest(), load_annotations()
    remeasure = "--remeasure" in sys.argv or "--measure-only" in sys.argv or not CROP_REGION_PATH.exists()
    if remeasure:
        if not annotations:
            raise FileNotFoundError("No annotations found. Run: python -m preprocessing.annotate_aligned")
        region = measure_region(cfg, manifest, annotations)
        if "--measure-only" in sys.argv:
            return
        CROP_REGION_PATH.parent.mkdir(parents=True, exist_ok=True)
        region.save(CROP_REGION_PATH)
        print(f"Saved {CROP_REGION_PATH}")
    else:
        region = CropRegion.load(CROP_REGION_PATH)
        print(f"Using the saved crop region {CROP_REGION_PATH} ({region.out_w} x {region.out_h}). "
              f"Pass --remeasure to measure it again from annotations.")
    write_crops(region, cfg, manifest, annotations)


if __name__ == "__main__":
    main()
