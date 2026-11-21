"""
Split the Tapal images into train / validation / test.

    python -m preprocessing.split_dataset        (run from the src folder)

Good cartons are grouped by camera frame, so copies of one frame never land
in two splits. Every real NOT OK carton goes to the test set only.
Writes data/tapal_filtered/tapal_session_split.json.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from config import TAPAL_OK_RAW_DIR, TAPAL_NOT_OK_RAW_DIR, RAW_FILTERED_DIR


def parse_tapal_filename(filepath: Path):
    """
    Parses filenames like: augmented_0_111605_230315_1_0000008974.jpg
    Returns (copy_index, session_id, unique_frame_id)
    """
    stem = filepath.stem
    m = re.match(r"^augmented_(\d+)_(\d+_\d+_\d+)_(\d+)$", stem, re.IGNORECASE)
    if m:
        copy_idx = int(m.group(1))
        session_id = m.group(2)
        frame_num = m.group(3)
        unique_frame_id = f"{session_id}_{frame_num}"
        return copy_idx, session_id, unique_frame_id

    digits = re.findall(r"\d+", stem)
    session_id = digits[1] if len(digits) >= 2 else "session_default"
    return 0, session_id, stem


def filter_and_split_sessions():
    ok_files = sorted([
        Path(p) for ext in ("*.bmp", "*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(str(TAPAL_OK_RAW_DIR / ext))
    ])
    not_ok_files = sorted([
        Path(p) for ext in ("*.bmp", "*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(str(TAPAL_NOT_OK_RAW_DIR / ext))
    ])

    # Sort OK files by physical camera frame ID so copies of the same frame stay in the same split
    ok_real = sorted(ok_files, key=lambda p: parse_tapal_filename(p)[2])
    not_ok_real = sorted(not_ok_files, key=lambda p: parse_tapal_filename(p)[2])

    print(f"[Tapal Dataset] OK: {len(ok_real)} images | Not_OK: {len(not_ok_real)} images "
          f"(Total: {len(ok_real) + len(not_ok_real)})")

    # Group by unique frame ID first, then partition into 10 contiguous session blocks
    # so identical frames never leak across train/val/test
    unique_frames = []
    frame_to_paths = {}
    for p in ok_real:
        _, _, fid = parse_tapal_filename(p)
        if fid not in frame_to_paths:
            unique_frames.append(fid)
            frame_to_paths[fid] = []
        frame_to_paths[fid].append(p)

    n_unique = len(unique_frames)
    n_train = max(1, int(0.70 * n_unique))
    n_val = max(1, int(0.15 * n_unique))

    train_fids = set(unique_frames[:n_train])
    val_fids = set(unique_frames[n_train:n_train + n_val])

    manifest = {"train_ok": [], "val_ok": [], "test_ok": [], "test_not_ok": []}
    for fid in unique_frames:
        target = "train_ok" if fid in train_fids else ("val_ok" if fid in val_fids else "test_ok")
        for p in frame_to_paths[fid]:
            _, sid, _ = parse_tapal_filename(p)
            manifest[target].append({"path": str(p), "session": fid, "label": "OK"})

    for p in not_ok_real:
        _, sid, fid = parse_tapal_filename(p)
        manifest["test_not_ok"].append({"path": str(p), "session": fid, "label": "Not_OK"})

    manifest_path = RAW_FILTERED_DIR / "tapal_session_split.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Session Split Saved] Train OK: {len(manifest['train_ok'])}, "
          f"Val OK: {len(manifest['val_ok'])}, Test OK: {len(manifest['test_ok'])}, "
          f"Test Real Not_OK: {len(manifest['test_not_ok'])}")
    return manifest


if __name__ == "__main__":
    filter_and_split_sessions()
