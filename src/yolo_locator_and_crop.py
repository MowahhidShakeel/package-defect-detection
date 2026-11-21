import os
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from config import (
    RAW_FILTERED_DIR, YOLO_DATASET_DIR, CROPS_DIR, WEIGHTS_DIR,
    YOLO_EPOCHS, YOLO_BATCH_SIZE, NORM_HEIGHT, NORM_MAX_WIDTH, DEVICE
)
from annotate import stage0_preprocess  # Defined inline below for standalone execution

def stage0_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (640, 640))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p2, p98 = np.percentile(l, (2, 98))
    if p98 > p2:
        l = np.clip((l.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def deskew_and_normalize_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Stage 2:
    1. Convert to grayscale.
    2. Estimate skew angle from ink pixels using minAreaRect and rotate horizontal.
    3. Resize to fixed height of 64 px keeping aspect ratio (Li & Zhang 2025), right-padded to NORM_MAX_WIDTH.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr

    # Otsu threshold to isolate dark ink on bright background (or vice versa)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connect horizontal text strokes to estimate line orientation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    coords = np.column_stack(np.where(connected > 0))
    if len(coords) > 50:
        rect = cv2.minAreaRect(coords[:, ::-1])  # (x, y)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        # Only deskew plausible conveyor rotation angles (-20 to +20 deg)
        if abs(angle) <= 20.0 and abs(angle) > 0.3:
            h, w = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Resize to fixed height 64 px keeping aspect ratio
    h, w = gray.shape[:2]
    scale = NORM_HEIGHT / max(1, h)
    new_w = min(NORM_MAX_WIDTH, max(16, int(round(w * scale))))
    resized = cv2.resize(gray, (new_w, NORM_HEIGHT), interpolation=cv2.INTER_AREA)

    bg_median = int(np.median(resized))
    canvas = np.full((NORM_HEIGHT, NORM_MAX_WIDTH), bg_median, dtype=np.uint8)
    canvas[:, :new_w] = resized
    return canvas

def train_stage1_locator() -> Path:
    data_yaml = YOLO_DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}. Run 01_prepare_tapal_and_annotate.py first.")

    best_weights = WEIGHTS_DIR / "yolov8n_print_region.pt"
    model = YOLO("yolov8n.pt")
    print("[Stage 1] Training YOLOv8n single-class print_region locator...")
    model.train(
        data=str(data_yaml),
        epochs=YOLO_EPOCHS,
        imgsz=640,
        batch=YOLO_BATCH_SIZE,
        device=0 if DEVICE == "cuda" else "cpu",
        project=str(WEIGHTS_DIR / "yolo_runs"),
        name="locator",
        exist_ok=True,
    )
    trained_pt = WEIGHTS_DIR / "yolo_runs" / "locator" / "weights" / "best.pt"
    if trained_pt.exists():
        import shutil
        shutil.copy(trained_pt, best_weights)

    metrics = YOLO(str(best_weights)).val(data=str(data_yaml), split="val", verbose=False)
    print(f"[Stage 1 Validation] mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")
    return best_weights

def crop_all_splits(locator_weights: Path):
    manifest_path = RAW_FILTERED_DIR / "tapal_session_split.json"
    fallback_boxes_path = RAW_FILTERED_DIR / "print_region_boxes.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    fallback_boxes = {}
    if fallback_boxes_path.exists():
        with open(fallback_boxes_path, "r") as f:
            fallback_boxes = json.load(f)

    locator = YOLO(str(locator_weights))
    stage1_rejects = []

    for split_key in ["train_ok", "val_ok", "test_ok", "test_not_ok"]:
        out_sub = CROPS_DIR / split_key
        out_sub.mkdir(parents=True, exist_ok=True)
        # Clear old crops if re-running
        for old_f in out_sub.glob("*.png"):
            old_f.unlink()

        items = manifest[split_key]
        for idx, item in enumerate(tqdm(items, desc=f"Cropping & Normalising {split_key}")):
            img_path = item["path"]
            entry = fallback_boxes.get(img_path, {"status": "has_print", "box": [110, 260, 395, 125]})
            if isinstance(entry, list):
                entry = {"status": "has_print", "box": entry}

            status = entry.get("status", "has_print")
            if status == "wrong_view":
                # Stage 1 rejects sideways / out-of-frame boxes before Stage 2/3
                stage1_rejects.append({"path": img_path, "split": split_key, "reason": "wrong_view"})
                continue

            img_raw = cv2.imread(img_path)
            if img_raw is None:
                continue
            img = stage0_preprocess(img_raw)

            preds = locator.predict(img, conf=0.25, verbose=False)[0]
            if len(preds.boxes) > 0:
                best_box = preds.boxes[int(np.argmax(preds.boxes.conf.cpu().numpy()))]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
            elif entry.get("box") is not None:
                x, y, bw, bh = entry["box"]
                x1, y1, x2, y2 = x, y, x + bw, y + bh
            else:
                x1, y1, x2, y2 = 110, 260, 505, 385

            crop_bgr = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            norm_gray = deskew_and_normalize_crop(crop_bgr)

            out_name = f"{split_key}_{idx:04d}_{status}_{item['session']}.png"
            cv2.imwrite(str(out_sub / out_name), norm_gray)

    with open(CROPS_DIR / "stage1_rejected_views.json", "w") as f:
        json.dump(stage1_rejects, f, indent=2)
    print(f"[Stage 2 Complete] Crops saved to {CROPS_DIR} | Stage 1 View Rejects: {len(stage1_rejects)}")

if __name__ == "__main__":
    weights = train_stage1_locator()
    crop_all_splits(weights)