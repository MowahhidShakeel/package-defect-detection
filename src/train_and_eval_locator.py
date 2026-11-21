"""
Train, calibrate and evaluate the label locator.

    python train_and_eval_locator.py export      build the aligned YOLO dataset from the annotations
    python train_and_eval_locator.py train       train YOLOv8n on it
    python train_and_eval_locator.py calibrate   learn the gross-check tolerances from good training cartons
    python train_and_eval_locator.py evaluate    model-only evaluation; also writes the crops for the inspector
    python train_and_eval_locator.py all         the four steps in order

Evaluation never reads an annotation to make a decision. Annotations are
used only afterwards, to score what the model decided.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys

import cv2
import numpy as np

from config import DEVICE, SEED, WEIGHTS_DIR
from label_locator import LabelLocator, LocatorConfig, OK_TO_INSPECT, box_iou, calibrate, draw_result
from locator_dataset import (
    CROPS_ALIGNED_DIR, LOCATOR_RESULTS_DIR, LOCATOR_STATS, LOCATOR_WEIGHTS, YOLO_ALIGNED_DIR,
    HAS_PRINT, UprightView, export_yolo_dataset, load_annotations, load_manifest,
)

LOCATOR_EPOCHS = 80
LOCATOR_BATCH = 16
WARMUP_IMAGES = 5          # excluded from the latency figures
IOU_GOOD = 0.70


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ==============================================================================
# Train
# ==============================================================================
def train() -> None:
    from ultralytics import YOLO
    data_yaml = YOLO_ALIGNED_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}. Run the 'export' step first.")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml), epochs=LOCATOR_EPOCHS, imgsz=640, batch=LOCATOR_BATCH, seed=SEED,
        device=0 if DEVICE == "cuda" else "cpu",
        project=str(WEIGHTS_DIR / "yolo_runs"), name="locator_aligned", exist_ok=True,
        # Frames are already level, and "upside down" is a class: flips must stay off,
        # and rotation stays small so the two classes are not blurred together.
        fliplr=0.0, flipud=0.0, degrees=3.0, mosaic=0.0,
        translate=0.05, scale=0.2, hsv_h=0.0, hsv_s=0.3, hsv_v=0.4,
    )
    best = WEIGHTS_DIR / "yolo_runs" / "locator_aligned" / "weights" / "best.pt"
    shutil.copy(best, LOCATOR_WEIGHTS)
    metrics = YOLO(str(LOCATOR_WEIGHTS)).val(data=str(data_yaml), split="val", verbose=False)
    print(f"[Locator] val mAP50 {metrics.box.map50:.4f} | mAP50-95 {metrics.box.map:.4f} | saved {LOCATOR_WEIGHTS}")


# ==============================================================================
# Calibrate
# ==============================================================================
def run_calibration() -> dict:
    locator = LabelLocator.from_files(LOCATOR_WEIGHTS, None, LocatorConfig(), device=DEVICE)
    images = [img for item in load_manifest()["train_ok"] if (img := cv2.imread(item["path"])) is not None]
    stats = calibrate(locator, images)
    LOCATOR_STATS.write_text(json.dumps(stats, indent=2))
    print(f"[Locator] calibrated on {stats['n_images']} of {len(images)} good training cartons")
    for key, (lo, hi) in stats["ranges"].items():
        print(f"    {key:>2}: allowed {lo:.3f} to {hi:.3f} (face-relative)")
    print(f"    skew tolerance {stats['skew_tol_deg']:.1f} deg (largest seen on good cartons: "
          f"{stats['observed_max_abs_skew_deg']}) | minimum ink share {stats['min_ink_frac']:.4f}")
    return stats


# ==============================================================================
# Evaluate (model only)
# ==============================================================================
def evaluate() -> None:
    cfg = LocatorConfig()
    locator = LabelLocator.from_files(LOCATOR_WEIGHTS, LOCATOR_STATS, cfg, device=DEVICE)
    if locator.stats is None:
        raise FileNotFoundError(f"Missing {LOCATOR_STATS}. Run the 'calibrate' step first.")
    manifest, annotations = load_manifest(), load_annotations()
    debug_dir = LOCATOR_RESULTS_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = DEVICE == "cuda"
    if use_cuda:
        import torch

    rows, seen = [], 0
    for split_key in ("train_ok", "val_ok", "test_ok", "test_not_ok"):
        out_dir = CROPS_ALIGNED_DIR / split_key
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()

        for idx, item in enumerate(manifest[split_key]):
            img = cv2.imread(item["path"])
            if img is None:
                continue
            if use_cuda:
                torch.cuda.synchronize()
            res = locator.process(img)
            if use_cuda:
                torch.cuda.synchronize()
            seen += 1

            entry = annotations.get(item["path"], {})
            iou, flip_ok = None, None
            if entry.get("status") == HAS_PRINT and res.box is not None:
                truth_flip = bool(entry.get("upside_down"))
                flip_ok = res.upside_down == truth_flip
                if flip_ok:
                    truth = UprightView.build(img, cfg, truth_flip).box_of(entry)
                    iou = box_iou(res.box, truth) if truth else None

            name = f"{split_key}_{idx:04d}_{item['session']}"
            if res.status == OK_TO_INSPECT:
                cv2.imwrite(str(out_dir / f"{name}.png"), res.crop)
            rejected_good = item["label"] == "OK" and res.status != OK_TO_INSPECT
            if rejected_good or (iou is not None and iou < IOU_GOOD) or flip_ok is False:
                cv2.imwrite(str(debug_dir / f"{name}.jpg"), draw_result(res))

            rows.append({
                "split": split_key, "path": item["path"], "label": item["label"],
                "annotated_status": entry.get("status", ""), "status": res.status,
                "reasons": "|".join(res.reasons), "det_conf": round(res.det_conf, 4),
                "upside_down": res.upside_down, "orientation_correct": "" if flip_ok is None else flip_ok,
                "iou": "" if iou is None else round(iou, 4),
                "carton_angle": round(res.carton_angle, 2),
                "print_skew": "" if res.print_skew is None else res.print_skew,
                "ink_frac": "" if res.ink_frac is None else round(res.ink_frac, 4),
                "ms": round(res.ms, 2), "warmup": seen <= WARMUP_IMAGES,
            })

    LOCATOR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCATOR_RESULTS_DIR / "locator_evaluation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------ report
    print("\n================ LABEL LOCATOR: MODEL-ONLY EVALUATION ================")
    print("(train_ok figures are shown for reference only; the tolerances were calibrated on those images.)")

    held_out = [r for r in rows if r["split"] in ("val_ok", "test_ok", "test_not_ok")]
    ious = [r["iou"] for r in held_out if r["iou"] != ""]
    flips = [r["orientation_correct"] for r in held_out if r["orientation_correct"] != ""]
    if ious:
        print(f"\nBox accuracy on {len(ious)} held-out annotated prints: mean IoU {np.mean(ious):.3f} | "
              f"median {np.median(ious):.3f} | IoU >= {IOU_GOOD}: {100 * np.mean(np.array(ious) >= IOU_GOOD):.1f}%")
    if flips:
        print(f"Orientation (upright vs upside down) correct: {sum(flips)}/{len(flips)}")

    print("\nGood cartons wrongly rejected at Stage 1 (false rejects):")
    for split_key in ("train_ok", "val_ok", "test_ok"):
        sub = [r for r in rows if r["split"] == split_key]
        k = sum(r["status"] != OK_TO_INSPECT for r in sub)
        lo, hi = clopper_pearson(k, len(sub))
        reasons = {}
        for r in sub:
            for reason in filter(None, r["reasons"].split("|")):
                reasons[reason] = reasons.get(reason, 0) + 1
        print(f"    {split_key:>9}: {k}/{len(sub)} = {100 * k / max(1, len(sub)):.2f}% "
              f"(95% CI {100 * lo:.1f}% to {100 * hi:.1f}%) {reasons if reasons else ''}")

    print("\nReal NOT OK cartons at Stage 1:")
    bad = [r for r in rows if r["split"] == "test_not_ok"]
    for status in sorted({r["annotated_status"] for r in bad}):
        sub = [r for r in bad if r["annotated_status"] == status]
        caught = [r for r in sub if r["status"] != OK_TO_INSPECT]
        reasons = {}
        for r in caught:
            for reason in filter(None, r["reasons"].split("|")):
                reasons[reason] = reasons.get(reason, 0) + 1
        print(f"    annotated '{status or 'not annotated'}': {len(caught)}/{len(sub)} rejected here {reasons}; "
              f"{len(sub) - len(caught)} passed on to the inspector")

    ms = np.array([r["ms"] for r in rows if not r["warmup"]])
    if len(ms):
        print(f"\nStage 1 time per image on this machine ({DEVICE}): median {np.median(ms):.1f} ms | "
              f"95th percentile {np.percentile(ms, 95):.1f} ms. Not a Jetson Nano figure.")
    print(f"\nPer-image results: {LOCATOR_RESULTS_DIR / 'locator_evaluation.csv'}")
    print(f"Pictures of every false reject and poor box: {debug_dir}")
    print(f"Crops for the inspector: {CROPS_ALIGNED_DIR}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("export", "all"):
        export_yolo_dataset()
    if step in ("train", "all"):
        train()
    if step in ("calibrate", "all"):
        run_calibration()
    if step in ("evaluate", "all"):
        evaluate()
    if step not in ("export", "train", "calibrate", "evaluate", "all"):
        print(__doc__)
