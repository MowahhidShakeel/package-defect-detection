"""
Look for defective cartons hiding in the good training set.

    python -m evaluation.audit_train                    (run from the src folder)
    python -m evaluation.audit_train --encoder resnet18 --seed 1

A folder of thousands of cartons labelled OK usually holds a few that are
not. Training on them teaches the model that those defects are acceptable.
This scores every good TRAINING crop with a trained model and shows the
highest-scoring ones, for a look by eye. Any that are truly defective should
be moved out of the OK folder before the dataset is split and trained again.

A high score here is not proof of a defect: the model has seen these
cartons, so most scores are low, and what stands out is whatever it found
hardest to accept.

Outputs in results/evaluation/<model>/
    audit_train.csv         every training crop with its score, highest first
    audit_train_top.jpg     the 48 highest-scoring crops, each beside its defect map
"""
from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np

from config import DEVICE, RESULTS_DIR
from evaluation.evaluate import frame_of, run
from inspector.dataset import crop_paths
from inspector.model import load_checkpoint
from inspector.train import checkpoint_path, run_name

TOP = 48


def main() -> None:
    p = argparse.ArgumentParser(description="Find suspicious cartons in the good training set.")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    model, _ = load_checkpoint(checkpoint_path(a.encoder, a.seed, a.tag), DEVICE)
    paths = crop_paths("train_ok")
    scores = []
    for i in range(0, len(paths), 64):                      # in chunks, to keep memory flat
        s, _ = run(model, [cv2.imread(str(q)) for q in paths[i:i + 64]], DEVICE)
        scores += s.tolist()
    order = np.argsort(scores)[::-1]

    out_dir = RESULTS_DIR / "evaluation" / run_name(a.encoder, a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit_train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "crop", "raw_image"])
        for rank, i in enumerate(order, 1):
            writer.writerow([rank, round(scores[i], 5), paths[i].name, frame_of(paths[i].name)])

    top = [cv2.imread(str(paths[i])) for i in order[:TOP]]
    _, maps = run(model, top, DEVICE)
    tiles = []
    for rank, (crop, prob, i) in enumerate(zip(top, maps, order[:TOP]), 1):
        overlay = cv2.addWeighted(crop, 0.55, cv2.applyColorMap(prob, cv2.COLORMAP_JET), 0.45, 0)
        tile = np.hstack([crop, overlay])
        text = f"#{rank}  score {scores[i]:.3f}  {frame_of(paths[i].name)}"
        cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
    cv2.imwrite(str(out_dir / "audit_train_top.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    s = np.array(scores)
    print(f"{len(paths)} good training crops scored with {run_name(a.encoder, a.tag)} seed {a.seed}.")
    print(f"Score: median {np.median(s):.4f} | 99th percentile {np.percentile(s, 99):.4f} | highest {s.max():.4f}")
    print(f"Crops scoring above 0.5: {int((s > 0.5).sum())}")
    print(f"Look at {out_dir / 'audit_train_top.jpg'} (highest 48) and {out_dir / 'audit_train.csv'}.")


if __name__ == "__main__":
    main()
