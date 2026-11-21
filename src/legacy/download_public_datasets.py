import os
import glob
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from config import (
    EXPDATE_DIR, IMPACT_DIR, YOLO_DATASET_DIR,
    NORM_HEIGHT, NORM_MAX_WIDTH, SEED
)

random.seed(SEED)
np.random.seed(SEED)

def normalize_crop_aspect_ratio(img_gray: np.ndarray) -> np.ndarray:
    """Resizes to height 64px keeping aspect ratio, right-pads to NORM_MAX_WIDTH."""
    h, w = img_gray.shape[:2]
    scale = NORM_HEIGHT / max(1, h)
    new_w = min(NORM_MAX_WIDTH, max(16, int(round(w * scale))))
    resized = cv2.resize(img_gray, (new_w, NORM_HEIGHT), interpolation=cv2.INTER_AREA)
    canvas = np.full((NORM_HEIGHT, NORM_MAX_WIDTH), int(np.median(resized)), dtype=np.uint8)
    canvas[:, :new_w] = resized
    return canvas

def setup_expdate():
    """Downloads ExpDate from Hugging Face (dimun/dates) and formats for YOLO + Verifier."""
    print("[ExpDate] Attempting download from Hugging Face ('dimun/dates')...")
    exp_crops_dir = EXPDATE_DIR / "date_real_crops"
    exp_crops_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        ds = load_dataset("dimun/dates", split="train")
        for idx, row in enumerate(tqdm(ds, desc="Processing ExpDate")):
            pil_img = row.get("image")
            if pil_img is None:
                continue
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # If dataset item is already a crop or has bbox annotations
            if "bboxes" in row and row["bboxes"]:
                img_resized = cv2.resize(img_bgr, (640, 640))
                h, w = img_bgr.shape[:2]
                out_img = YOLO_DATASET_DIR / "images" / "train" / f"expdate_{idx:04d}.jpg"
                out_lbl = YOLO_DATASET_DIR / "labels" / "train" / f"expdate_{idx:04d}.txt"
                cv2.imwrite(str(out_img), img_resized)
                with open(out_lbl, "w") as f:
                    for box in row["bboxes"]:
                        x1, y1, x2, y2 = box[:4]
                        xc, yc = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                        bw, bh = (x2 - x1) / w, (y2 - y1) / h
                        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                        crop = gray[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                        if crop.size > 0:
                            norm = normalize_crop_aspect_ratio(crop)
                            cv2.imwrite(str(exp_crops_dir / f"exp_crop_{idx:04d}.png"), norm)
            else:
                norm = normalize_crop_aspect_ratio(gray)
                cv2.imwrite(str(exp_crops_dir / f"exp_crop_{idx:04d}.png"), norm)
        print(f"[ExpDate] Saved normalized crops to {exp_crops_dir}")
    except Exception as e:
        print(f"[ExpDate Notice] Could not fetch 'dimun/dates' automatically ({e}). "
              f"If you have local ExpDate images, place them in {EXPDATE_DIR / 'raw'} and re-run.")

def setup_impact_word_crops(num_fallback_words: int = 1500):
    """
    Prepares IMPACT historical printed word crops for Stage 3b ResNet-34 pre-training.
    1. If raw IMPACT PAGE-XML + TIFF/PNG pages exist in data/impact_words/raw, parses word polygons.
    2. Otherwise, generates a synthetic historical printed-word corpus with realistic ink ageing,
       bleed, and speckled paper degradation so the pre-training script runs out-of-the-box.
    """
    raw_impact = IMPACT_DIR / "raw"
    crops_out = IMPACT_DIR / "word_crops"
    crops_out.mkdir(parents=True, exist_ok=True)

    xml_files = list(raw_impact.glob("*.xml")) if raw_impact.exists() else []
    extracted = 0

    if xml_files:
        print(f"[IMPACT] Found {len(xml_files)} PAGE-XML files. Extracting printed word crops...")
        for xml_path in tqdm(xml_files):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_candidates = list(raw_impact.glob(f"{xml_path.stem}.*"))
            img_Candidates = [p for p in img_candidates if p.suffix.lower() in [".png", ".jpg", ".tif", ".tiff"]]
            if not img_Candidates:
                continue
            page_img = cv2.imread(str(img_Candidates[0]), cv2.IMREAD_GRAYSCALE)
            if page_img is None:
                continue

            for elem in root.iter():
                if elem.tag.endswith("Word"):
                    word_id = elem.attrib.get("id", f"w_{extracted}")
                    coords = [c for c in elem if c.tag.endswith("Coords")]
                    if not coords:
                        continue
                    pts_str = coords[0].attrib.get("points", "")
                    pts = [tuple(map(int, p.split(","))) for p in pts_str.split() if "," in p]
                    if len(pts) < 3:
                        continue
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(page_img.shape[1], max(xs)), min(page_img.shape[0], max(ys))
                    if (x2 - x1) > 12 and (y2 - y1) > 8:
                        crop = normalize_crop_aspect_ratio(page_img[y1:y2, x1:x2])
                        cv2.imwrite(str(crops_out / f"impact_{extracted:05d}_{word_id}.png"), crop)
                        extracted += 1

    if extracted == 0:
        print("[IMPACT] Generating degraded printed-word pre-training corpus (warm-start substrate)...")
        vocab = [
            "BATCH", "EXPIRY", "MFG", "PRICE", "RS", "PKR", "WEIGHT", "NET", "2025", "2026",
            "OCT", "NOV", "DEC", "JAN", "LOT", "PACKED", "TAPAL", "DANEDAR", "TEA", "GRAMS",
            "01", "02", "15", "28", "450G", "900G", "190", "550", "CODE", "LINE", "SHIFT"
        ]
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX
        ]
        for i in tqdm(range(num_fallback_words), desc="Synthesizing IMPACT-style word crops"):
            word = f"{random.choice(vocab)}-{random.randint(10, 99)}"
            font = random.choice(fonts)
            # Generate 3 samples per word identity (anchor, positive, degraded)
            for sample_idx in range(3):
                bg_val = random.randint(185, 235)
                canvas = np.full((64, 180), bg_val, dtype=np.uint8)
                ink_val = random.randint(15, 70)
                scale = random.uniform(0.75, 1.05)
                thick = random.choice([1, 2])
                ox, oy = random.randint(6, 16), random.randint(40, 50)
                cv2.putText(canvas, word, (ox, oy), font, scale, ink_val, thick, cv2.LINE_AA)

                # Historical paper grain + ageing noise
                noise = np.random.normal(0, random.uniform(4, 12), canvas.shape).astype(np.float32)
                canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                if sample_idx == 2:
                    # Apply severe historical ink degradation (erosion or smudge) for negative pair
                    if random.random() < 0.5:
                        k = np.ones((3, 3), np.uint8)
                        canvas = cv2.dilate(canvas, k, iterations=1)  # erodes dark ink on light bg
                    else:
                        bx = random.randint(20, 120)
                        cv2.circle(canvas, (bx, 32), random.randint(8, 14), ink_val, -1)
                        canvas = cv2.GaussianBlur(canvas, (5, 5), 1.5)

                norm = normalize_crop_aspect_ratio(canvas)
                tag = "neg" if sample_idx == 2 else f"pos{sample_idx}"
                cv2.imwrite(str(crops_out / f"word_{i:04d}_{tag}.png"), norm)

    print(f"[IMPACT Ready] Word crops saved in {crops_out}")

if __name__ == "__main__":
    setup_expdate()
    setup_impact_word_crops()