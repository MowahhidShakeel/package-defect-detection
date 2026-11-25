"""
Package a trained inspector for the Jetson Nano.

Run from the src folder, on the workstation:

    python -m deploy.export                          ResNet-34, the seed with the best validation loss
    python -m deploy.export --encoder resnet18
    python -m deploy.export --seed 1 --tag redraw

Writes deploy_bundle/<name>/ in the repository root:

    model.onnx              the network; input 1 x 3 x H x W, RGB, 0..255; output the defect map
    bundle.json             threshold, crop region, alignment settings
    jetson_inspector.py     the inspection engine and command line for the Jetson
    jetson_gui.py           the demo window

Copy that folder to the Jetson and run `python3 jetson_gui.py` there. The
window also runs on this workstation, which is the quick way to try it:

    python deploy_bundle/resnet34/jetson_gui.py

The threshold is the same 'good' threshold the evaluation uses: just above
the highest score of any good validation carton. The seed is chosen by
validation loss. No test image influences either choice.

After writing the bundle this script checks it against the PyTorch
pipeline on test images: same crops, same scores, same verdicts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, REPO_ROOT
from evaluation.evaluate import GOOD_MARGIN, run
from inspector.dataset import good_validation_variants
from inspector.model import IMAGENET_MEAN, IMAGENET_STD, TOP_K, carton_score, load_checkpoint, to_tensor
from inspector.train import checkpoint_path, run_name
from preprocessing.align import AlignConfig, CropRegion, extract_label_crop
from preprocessing.annotations import CROP_REGION_PATH, load_manifest

BUNDLE_ROOT = REPO_ROOT / "deploy_bundle"
OPSET = 13          # old enough for the TensorRT that ships with JetPack 4.x


class Deployable(nn.Module):
    """The inspector with its input scaling and the final sigmoid built in, so
    the Jetson only has to hand over raw RGB pixels."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((rgb - self.mean) / self.std))


def pick_seed(encoder: str, tag: str) -> int:
    best, best_loss = None, float("inf")
    for path in sorted(checkpoint_path(encoder, 0, tag).parent.glob(f"inspector_{run_name(encoder, tag)}_seed*.pt")):
        info = torch.load(str(path), map_location="cpu", weights_only=False)
        if info.get("val_loss", float("inf")) < best_loss:
            best, best_loss = int(info["seed"]), float(info["val_loss"])
    if best is None:
        raise FileNotFoundError(f"No trained weights found for {run_name(encoder, tag)}. Run inspector.train first.")
    print(f"Seed {best} has the best validation loss ({best_loss:.4f}).")
    return best


def export(encoder: str, tag: str, seed: int | None) -> Path:
    name = run_name(encoder, tag)
    if seed is None:
        seed = pick_seed(encoder, tag)
    model, info = load_checkpoint(checkpoint_path(encoder, seed, tag), DEVICE)
    region = CropRegion.load(CROP_REGION_PATH)
    cfg = AlignConfig()

    # Threshold from good validation cartons only
    good_scores, _ = run(model, good_validation_variants(), DEVICE)
    threshold = float(good_scores.max()) + GOOD_MARGIN

    out_dir = BUNDLE_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = Deployable(model).to("cpu").eval()
    dummy = torch.zeros(1, 3, region.out_h, region.out_w)
    kwargs = dict(input_names=["image"], output_names=["defect_map"], opset_version=OPSET, do_constant_folding=True)
    try:
        torch.onnx.export(wrapper, dummy, str(out_dir / "model.onnx"), dynamo=False, **kwargs)
    except TypeError:                       # older PyTorch: no 'dynamo' argument
        torch.onnx.export(wrapper, dummy, str(out_dir / "model.onnx"), **kwargs)

    bundle = {
        "name": name, "encoder": encoder, "seed": seed, "epoch": info.get("epoch"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": round(threshold, 6), "top_k": TOP_K,
        "highest_good_validation_score": round(float(good_scores.max()), 6),
        "crop_region": {**region.__dict__, "fallback_face": list(region.fallback_face)},
        "align": {"hsv_ranges": [[list(lo), list(hi)] for lo, hi in cfg.hsv_ranges],
                  "min_face_area_frac": cfg.min_face_area_frac, "min_align_angle_deg": cfg.min_align_angle_deg},
    }
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    for script in ("jetson_inspector.py", "jetson_gui.py"):
        shutil.copy(Path(__file__).with_name(script), out_dir / script)
    size_mb = (out_dir / "model.onnx").stat().st_size / 1e6
    print(f"Bundle written to {out_dir} (model.onnx {size_mb:.1f} MB, input {region.out_w} x {region.out_h}, "
          f"threshold {threshold:.4f})")
    check(out_dir, model, region, cfg, threshold)
    return out_dir


def check(out_dir: Path, model, region: CropRegion, cfg: AlignConfig, threshold: float, per_split: int = 12) -> None:
    """Run the Jetson script (with onnxruntime) and the PyTorch pipeline on the
    same raw test images and compare."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("onnxruntime is not installed, so the bundle was not checked. Install it and run again:  pip install onnxruntime")
        return
    from deploy.jetson_inspector import Inspector

    jetson = Inspector(str(out_dir), backend="ort")
    manifest = load_manifest()
    items = manifest["test_ok"][:per_split] + manifest["test_not_ok"][::max(1, len(manifest["test_not_ok"]) // per_split)]
    crop_diff, score_diff, disagree, n = 0.0, 0.0, 0, 0
    for item in items:
        img = cv2.imread(item["path"])
        if img is None:
            continue
        ref_crop = extract_label_crop(img, region, cfg).crop
        with torch.no_grad():
            ref_score = float(carton_score(model(to_tensor(ref_crop).unsqueeze(0).to(DEVICE))).item())
        r = jetson.inspect(img)
        crop_diff = max(crop_diff, float(np.abs(r["crop"].astype(np.int16) - ref_crop.astype(np.int16)).max()))
        score_diff = max(score_diff, abs(r["score"] - ref_score))
        disagree += int((r["score"] >= threshold) != (ref_score >= threshold))
        n += 1
    if n == 0:
        print("No raw test images could be read, so the bundle was not checked.")
        return
    ok = crop_diff == 0 and score_diff < 0.01 and disagree == 0
    print(f"Bundle check on {n} raw test images: largest crop difference {crop_diff:.0f} grey levels, "
          f"largest score difference {score_diff:.5f}, verdicts that differ: {disagree} -> {'PASSED' if ok else 'FAILED'}")
    if not ok:
        print("The Jetson script does not reproduce the PyTorch pipeline. Do not deploy this bundle.")


def main() -> None:
    p = argparse.ArgumentParser(description="Package a trained inspector for the Jetson Nano.")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=None, help="default: the seed with the best validation loss")
    a = p.parse_args()
    export(a.encoder, a.tag, a.seed)


if __name__ == "__main__":
    main()
