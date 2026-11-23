"""
Split the Tapal images into train / validation / test.

    python -m preprocessing.split_dataset        (run from the src folder)

File names look like 111540_230315_1_0000008890.bmp:
capture time (hhmmss), date (yymmdd), line number, camera frame number.

Good cartons are split by the MINUTE they were captured in. All cartons of
one minute go to the same split, so the validation and test sets contain
print times the model never saw in training. That is the situation on the
line every day, and it is what the false reject rate should be measured on.

Every real NOT OK carton goes to the test set only.
Writes data/tapal_filtered/tapal_session_split.json.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from config import RAW_FILTERED_DIR, SEED, TAPAL_NOT_OK_RAW_DIR, TAPAL_OK_RAW_DIR

NAME_RE = re.compile(r"^(?:.*_)?(\d{6})_(\d{6})_(\d+)_(\d+)$")
IMAGE_TYPES = (".bmp", ".png", ".jpg", ".jpeg")
TRAIN_SHARE, VAL_SHARE = 0.70, 0.15


def describe(path: Path) -> dict:
    m = NAME_RE.match(path.stem)
    if not m:
        return {"minute": "unknown", "frame": path.stem}
    return {"minute": f"{m.group(2)}_{m.group(1)[:4]}", "frame": m.group(4)}


def list_images(folder: Path) -> list:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}. Check the dataset paths in config.py.")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)


def split_dataset() -> dict:
    ok, not_ok = list_images(TAPAL_OK_RAW_DIR), list_images(TAPAL_NOT_OK_RAW_DIR)
    by_minute = {}
    for p in ok:
        by_minute.setdefault(describe(p)["minute"], []).append(p)

    # Hand out whole minutes at random until each split has its share of the images
    minutes = sorted(by_minute)
    random.Random(SEED).shuffle(minutes)
    targets = {"train_ok": TRAIN_SHARE * len(ok), "val_ok": VAL_SHARE * len(ok)}
    manifest = {"train_ok": [], "val_ok": [], "test_ok": [], "test_not_ok": []}
    minutes_in = {k: [] for k in manifest}
    for minute in minutes:
        split = next((k for k in ("train_ok", "val_ok") if len(manifest[k]) < targets[k]), "test_ok")
        minutes_in[split].append(minute)
        for p in by_minute[minute]:
            manifest[split].append({"path": str(p), "session": p.stem, "label": "OK", "minute": minute})
    for p in not_ok:
        manifest["test_not_ok"].append({"path": str(p), "session": p.stem, "label": "Not_OK",
                                        "minute": describe(p)["minute"]})

    RAW_FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_FILTERED_DIR / "tapal_session_split.json"
    out.write_text(json.dumps(manifest, indent=2))

    print(f"{len(ok)} good and {len(not_ok)} NOT OK images | {len(minutes)} capture minutes of good cartons")
    for k in ("train_ok", "val_ok", "test_ok"):
        print(f"  {k:>9}: {len(manifest[k]):5d} images from {len(minutes_in[k]):2d} minutes")
    print(f"  test_not_ok: {len(manifest['test_not_ok'])} images")
    overlap = {describe(Path(i['path']))['frame'] for i in manifest['test_not_ok']} & \
              {describe(p)['frame'] for p in ok}
    if overlap:
        print(f"WARNING: {len(overlap)} frame numbers appear in both the OK and NG folders: {sorted(overlap)[:5]}")
    print(f"Saved {out}")
    return manifest


if __name__ == "__main__":
    split_dataset()
