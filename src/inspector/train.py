"""
Train the inspector.

Run from the src folder:

    python -m inspector.train                               ResNet-34, seeds 0 1 2
    python -m inspector.train --encoder mobilevit_xxs       another encoder, same everything else
    python -m inspector.train --seeds 0 --epochs 5          a quick trial run
    python -m inspector.train --encoder resnet18 --tag trial   a variant, saved beside the others

Each seed is a full, independent training run. Comparing models on several
seeds shows whether a difference between them is real or luck.

Outputs
    weights/inspector_<encoder>[_<tag>]_seed<n>.pt          best epoch by validation loss
    results/inspector/train_<encoder>[_<tag>]_seed<n>.csv   loss and validation scores per epoch
"""
from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import DEVICE, RESULTS_DIR, WEIGHTS_DIR
from inspector.dataset import ListSet, TrainSet, build_validation_set
from inspector.model import DefectMapNet, carton_score, count_parameters, save_checkpoint

INSPECTOR_RESULTS_DIR = RESULTS_DIR / "inspector"


def run_name(encoder: str, tag: str = "") -> str:
    return f"{encoder}_{tag}" if tag else encoder


def checkpoint_path(encoder: str, seed: int, tag: str = ""):
    return WEIGHTS_DIR / f"inspector_{run_name(encoder, tag)}_seed{seed}.pt"


def defect_map_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-pixel cross-entropy plus Dice. Dice keeps small defects from being
    drowned out by the many clean pixels."""
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    prob = torch.sigmoid(logits)
    inter = (prob * mask).sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1.0)
    return bce + dice.mean()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, n, scores, labels = 0.0, 0, [], []
    for x, m, y in loader:
        x, m = x.to(device), m.to(device)
        logits = model(x)
        total += defect_map_loss(logits, m).item() * len(x)
        n += len(x)
        scores += carton_score(logits).cpu().tolist()
        labels += y.tolist()
    return total / max(n, 1), roc_auc_score(labels, scores)


def train_one(encoder: str, seed: int, epochs: int, batch: int, repeats: int, pretrained: bool,
              redraw_ok: bool = True, tag: str = "") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = DEVICE
    use_amp = device == "cuda"

    train_set = TrainSet(seed=seed, repeats=repeats, redraw_ok=redraw_ok)
    # num_workers=0: the generator keeps its own random state, and it is fast enough in-process
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(ListSet(build_validation_set()), batch_size=batch, shuffle=False, num_workers=0)

    model = DefectMapNet(encoder, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-4},                                   # pretrained: small steps
        {"params": list(model.decoder.parameters()) + list(model.head.parameters()), "lr": 1e-3},
    ], weight_decay=1e-4)
    steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    name = run_name(encoder, tag)
    print(f"\n=== {name} | seed {seed} | {count_parameters(model) / 1e6:.1f}M parameters | {device} | "
          f"{len(train_set.crops)} training crops, {len(train_set)} samples per epoch, {len(val_loader.dataset)} validation images ===")
    INSPECTOR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = INSPECTOR_RESULTS_DIR / f"train_{name}_seed{seed}.csv"
    best, rows, start = float("inf"), [], time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, m, _ in train_loader:
            x, m = x.to(device), m.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = defect_map_loss(model(x).float(), m)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(x)
            seen += len(x)

        val_loss, val_auroc = validate(model, val_loader, device)
        saved = val_loss < best
        if saved:
            best = val_loss
            save_checkpoint(checkpoint_path(encoder, seed, tag), model, seed=seed, epoch=epoch,
                            val_loss=val_loss, val_auroc=val_auroc, pretrained=pretrained, redraw_ok=redraw_ok)
        rows.append({"epoch": epoch, "train_loss": round(running / seen, 5), "val_loss": round(val_loss, 5),
                     "val_image_auroc": round(val_auroc, 5), "saved": saved})
        print(f"  epoch {epoch:3d}/{epochs} | train loss {running / seen:.4f} | val loss {val_loss:.4f} | "
              f"val AUROC {val_auroc:.4f}{' | saved' if saved else ''}")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  done in {(time.time() - start) / 60:.1f} min | best val loss {best:.4f} | {checkpoint_path(encoder, seed, tag)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the defect-map inspector.")
    p.add_argument("--encoder", default="resnet34", help="any timm model that supports features_only")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--repeats", type=int, default=10, help="times each training crop is drawn per epoch")
    p.add_argument("--no-pretrained", action="store_true", help="start from random weights (for testing only)")
    p.add_argument("--no-redraw-ok", action="store_true",
                   help="do NOT train on good cartons whose print was redrawn in place. Without them the model "
                        "learns 'redrawn print means defect' (see DefectGenerator.redraw_clean); for comparison only")
    p.add_argument("--tag", default="", help="suffix for the saved files, so a variant does not overwrite the baseline")
    a = p.parse_args()
    for seed in a.seeds:
        train_one(a.encoder, seed, a.epochs, a.batch, a.repeats, not a.no_pretrained, not a.no_redraw_ok, a.tag)


if __name__ == "__main__":
    main()
