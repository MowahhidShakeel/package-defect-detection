"""
Evaluate a trained inspector.

Run from the src folder:

    python -m evaluation.evaluate                           ResNet-34, seeds 0 1 2
    python -m evaluation.evaluate --encoder mobilevit_xxs --seeds 0 1 2

For each seed:
    1. The OK / NOT OK threshold is set on the VALIDATION images only.
    2. Good test cartons            -> false reject rate
    3. Real NOT OK test cartons     -> recall, split by fault kind
    4. Synthetic test set           -> recall per defect type, and how well the map finds the defect
    5. Retouched good test cartons  -> shortcut check: must score like untouched ones
    6. Time per carton on this machine

Real test images are never used to choose anything.

Outputs in results/evaluation/<encoder>/
    summary.csv                    one row per seed, plus mean and spread
    per_image_seed<n>.csv          every test image with its score and verdict
    maps_seed<n>/*.jpg             crop next to its defect map: every mistake, plus examples
"""
from __future__ import annotations

import argparse
import csv
import re
import time

import cv2
import numpy as np
import torch
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, DEVICE, RESULTS_DIR
from generator.defects import DefectGenerator
from inspector.dataset import build_validation_set, load_crops
from inspector.model import carton_score, count_parameters, load_checkpoint, to_tensor
from inspector.train import checkpoint_path

SYNTHETIC_TEST_DIR = DATA_DIR / "synthetic_test"
CROPS_LOG = RESULTS_DIR / "preprocessing" / "crops_log.csv"
NAME_RE = re.compile(r"^(train_ok|val_ok|test_ok|test_not_ok)_\d{4}_(.+)\.png$")
WARMUP = 10
EXAMPLES_PER_KIND = 6


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def frame_of(name: str) -> str:
    """Camera frame a crop came from. Copies of one frame are not independent evidence."""
    m = NAME_RE.match(name)
    return m.group(2) if m else name


@torch.no_grad()
def run(model, images, device, batch: int = 16):
    """Scores (N,) and probability maps (N x H x W, uint8 0-255) for a list of BGR crops."""
    scores, maps = [], []
    for i in range(0, len(images), batch):
        x = torch.stack([to_tensor(im) for im in images[i:i + batch]]).to(device)
        logits = model(x)
        scores += carton_score(logits).cpu().tolist()
        maps += list((torch.sigmoid(logits)[:, 0] * 255).round().byte().cpu().numpy())
    return np.array(scores), maps


@torch.no_grad()
def time_per_carton(model, image, device, runs: int = 60):
    x = to_tensor(image).unsqueeze(0).to(device)
    times = []
    for i in range(WARMUP + runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        carton_score(model(x))
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


def pick_threshold(good: np.ndarray, bad: np.ndarray) -> float:
    """The cut that best separates good from defective validation images
    (largest recall minus false reject rate), placed midway in the gap."""
    values = np.unique(np.concatenate([good, bad]))
    cuts = np.concatenate([[values[0] - 1e-6], (values[:-1] + values[1:]) / 2.0, [values[-1] + 1e-6]])
    quality = [np.mean(bad >= c) - np.mean(good >= c) for c in cuts]
    best = float(np.max(quality))
    ties = cuts[np.isclose(quality, best)]
    return float(ties[len(ties) // 2])


def save_map(path, crop, prob, text) -> None:
    heat = cv2.applyColorMap(prob, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop, 0.55, heat, 0.45, 0)
    out = np.hstack([crop, overlay])
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])


def load_synthetic_test():
    meta = SYNTHETIC_TEST_DIR / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"Missing {meta}. Run: python -m generator.make export")
    rows = list(csv.DictReader(open(meta)))
    images = [cv2.imread(str(SYNTHETIC_TEST_DIR / r["image"])) for r in rows]
    masks = [cv2.imread(str(SYNTHETIC_TEST_DIR / r["mask"]), cv2.IMREAD_GRAYSCALE) for r in rows]
    return rows, images, masks


def evaluate_seed(encoder: str, seed: int, out_dir, data) -> dict:
    device = DEVICE
    model, info = load_checkpoint(checkpoint_path(encoder, seed), device)
    maps_dir = out_dir / f"maps_seed{seed}"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for old in maps_dir.glob("*.jpg"):
        old.unlink()

    # 1. Threshold from validation images only
    val = data["val"]
    val_scores, _ = run(model, [v[0] for v in val], device)
    val_labels = np.array([v[2] for v in val])
    thr = pick_threshold(val_scores[val_labels == 0], val_scores[val_labels == 1])

    # 2-3. Real test cartons
    ok_scores, ok_maps = run(model, data["ok_imgs"], device)
    bad_scores, bad_maps = run(model, data["bad_imgs"], device)
    ok_rej, bad_hit = ok_scores >= thr, bad_scores >= thr

    # 4. Synthetic test set
    syn_scores, syn_maps = run(model, data["syn_imgs"], device)
    syn_hit = syn_scores >= thr
    rng = np.random.default_rng(0)
    px_true, px_score = [], []
    for mask, prob in zip(data["syn_masks"], syn_maps):          # a sample of pixels keeps this fast
        idx = rng.choice(mask.size, size=4000, replace=False)
        px_true.append(mask.ravel()[idx] > 0)
        px_score.append(prob.ravel()[idx])
    pixel_auroc = roc_auc_score(np.concatenate(px_true), np.concatenate(px_score))

    # 5. Shortcut check
    ret_scores, _ = run(model, data["retouched"], device)
    ret_rej = ret_scores >= thr

    # 6. Speed
    ms_median, ms_p95 = time_per_carton(model, data["ok_imgs"][0], device)

    # ------------------------------------------------------------- per-image log
    rows = []
    for name, s, r in zip(data["ok_names"], ok_scores, ok_rej):
        rows.append({"set": "test_ok", "image": name, "kind": "good", "frame": frame_of(name),
                     "score": round(float(s), 5), "verdict": "NOT_OK" if r else "OK", "correct": not r})
    for name, kind, s, hit in zip(data["bad_names"], data["bad_kinds"], bad_scores, bad_hit):
        rows.append({"set": "test_not_ok", "image": name, "kind": kind, "frame": frame_of(name),
                     "score": round(float(s), 5), "verdict": "NOT_OK" if hit else "OK", "correct": bool(hit)})
    for r, s, hit in zip(data["syn_rows"], syn_scores, syn_hit):
        rows.append({"set": "synthetic", "image": r["image"], "kind": r["types"], "frame": frame_of(r["source_crop"]),
                     "score": round(float(s), 5), "verdict": "NOT_OK" if hit else "OK", "correct": bool(hit)})
    with open(out_dir / f"per_image_seed{seed}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------- pictures
    for i in np.flatnonzero(ok_rej):
        save_map(maps_dir / f"FALSE_REJECT_{data['ok_names'][i][:-4]}.jpg", data["ok_imgs"][i], ok_maps[i],
                 f"good carton rejected, score {ok_scores[i]:.3f} (threshold {thr:.3f})")
    for i in np.flatnonzero(~bad_hit):
        save_map(maps_dir / f"MISS_{data['bad_kinds'][i]}_{data['bad_names'][i][:-4]}.jpg", data["bad_imgs"][i],
                 bad_maps[i], f"real {data['bad_kinds'][i]} missed, score {bad_scores[i]:.3f} (threshold {thr:.3f})")
    shown = {}
    for i, kind in enumerate(data["bad_kinds"]):
        if bad_hit[i] and shown.get(kind, 0) < EXAMPLES_PER_KIND:
            shown[kind] = shown.get(kind, 0) + 1
            save_map(maps_dir / f"caught_{kind}_{shown[kind]}.jpg", data["bad_imgs"][i], bad_maps[i],
                     f"real {kind} caught, score {bad_scores[i]:.3f}")
    for i in np.flatnonzero(~syn_hit)[:30]:
        save_map(maps_dir / f"MISS_synthetic_{i:04d}_{data['syn_rows'][i]['types'].replace('|', '+')}.jpg",
                 data["syn_imgs"][i], syn_maps[i], f"synthetic {data['syn_rows'][i]['types']} missed, score {syn_scores[i]:.3f}")

    # ------------------------------------------------------------- report
    n_ok, n_bad = len(ok_scores), len(bad_scores)
    fr_lo, fr_hi = clopper_pearson(int(ok_rej.sum()), n_ok)
    miss = int((~bad_hit).sum())
    ms_lo, ms_hi = clopper_pearson(miss, n_bad)
    ok_frames = sorted({frame_of(n) for n in data["ok_names"]})
    bad_frames = sorted({frame_of(n) for n in data["bad_names"]})
    bad_frame_caught = sum(all(h for n, h in zip(data["bad_names"], bad_hit) if frame_of(n) == fr) for fr in bad_frames)
    ok_frame_clean = sum(not any(r for n, r in zip(data["ok_names"], ok_rej) if frame_of(n) == fr) for fr in ok_frames)

    print(f"\n--- {encoder} seed {seed} (epoch {info.get('epoch')}, {count_parameters(model) / 1e6:.1f}M parameters) ---")
    print(f"Threshold {thr:.4f}, set on {int((val_labels == 0).sum())} good and {int((val_labels == 1).sum())} "
          f"synthetic-defect validation images")
    print(f"Good test cartons rejected : {int(ok_rej.sum())}/{n_ok} = {100 * ok_rej.mean():.2f}% "
          f"(95% CI {100 * fr_lo:.1f} to {100 * fr_hi:.1f}%) | frames with no reject: {ok_frame_clean}/{len(ok_frames)}")
    print(f"Real NOT OK cartons caught : {n_bad - miss}/{n_bad} = {100 * bad_hit.mean():.2f}% "
          f"(miss rate 95% CI {100 * ms_lo:.1f} to {100 * ms_hi:.1f}%) | frames fully caught: {bad_frame_caught}/{len(bad_frames)}")
    for kind in sorted(set(data["bad_kinds"])):
        sel = np.array([k == kind for k in data["bad_kinds"]])
        print(f"      {kind:<16} {int(bad_hit[sel].sum())}/{int(sel.sum())}")
    print(f"Real AUROC (good vs NOT OK): {roc_auc_score([0] * n_ok + [1] * n_bad, np.concatenate([ok_scores, bad_scores])):.4f}")
    print(f"Synthetic defects caught   : {int(syn_hit.sum())}/{len(syn_hit)} = {100 * syn_hit.mean():.2f}% | "
          f"map quality (pixel AUROC) {pixel_auroc:.4f}")
    type_recall = {}
    for t in sorted({t for r in data["syn_rows"] for t in r["types"].split("|")}):
        sel = np.array([r["types"] == t for r in data["syn_rows"]])            # single-defect images only
        if sel.any():
            sev = np.array([float(r["severity"]) for r, s in zip(data["syn_rows"], sel) if s])
            hits = syn_hit[sel]
            mild = hits[sev < 0.34].mean() if (sev < 0.34).any() else float("nan")
            type_recall[t] = float(hits.mean())
            print(f"      {t:<20} {int(hits.sum()):3d}/{int(sel.sum()):3d} | mild (severity under 0.34): {100 * mild:.0f}%")
    print(f"Shortcut check             : retouched good cartons rejected {int(ret_rej.sum())}/{len(ret_rej)} "
          f"(untouched: {int(ok_rej.sum())}/{n_ok}); mean score {ret_scores.mean():.4f} vs {ok_scores.mean():.4f}")
    print(f"Time per carton ({device})   : median {ms_median:.1f} ms, 95th percentile {ms_p95:.1f} ms. Not a Jetson Nano figure.")

    summary = {
        "seed": seed, "epoch": info.get("epoch"), "threshold": round(thr, 5),
        "false_reject_pct": round(100 * float(ok_rej.mean()), 3),
        "real_recall_pct": round(100 * float(bad_hit.mean()), 3),
        "real_auroc": round(float(roc_auc_score([0] * n_ok + [1] * n_bad, np.concatenate([ok_scores, bad_scores]))), 5),
        "synthetic_recall_pct": round(100 * float(syn_hit.mean()), 3),
        "synthetic_auroc": round(float(roc_auc_score([0] * n_ok + [1] * len(syn_scores), np.concatenate([ok_scores, syn_scores]))), 5),
        "pixel_auroc": round(float(pixel_auroc), 5),
        "retouched_reject_pct": round(100 * float(ret_rej.mean()), 3),
        "ms_median": round(ms_median, 2), "parameters_m": round(count_parameters(model) / 1e6, 2),
    }
    for kind in sorted(set(data["bad_kinds"])):
        sel = np.array([k == kind for k in data["bad_kinds"]])
        summary[f"real_{kind}_recall_pct"] = round(100 * float(bad_hit[sel].mean()), 2)
    for t, v in type_recall.items():
        summary[f"syn_{t}_recall_pct"] = round(100 * v, 2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the defect-map inspector.")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()

    out_dir = RESULTS_DIR / "evaluation" / a.encoder
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_imgs, ok_names = load_crops("test_ok")
    bad_imgs, bad_names = load_crops("test_not_ok")
    kinds = {}
    if CROPS_LOG.exists():
        kinds = {r["crop"]: r["annotated_status"] for r in csv.DictReader(open(CROPS_LOG))}
    syn_rows, syn_imgs, syn_masks = load_synthetic_test()
    gen = DefectGenerator(seed=777)
    data = {
        "val": build_validation_set(),
        "ok_imgs": ok_imgs, "ok_names": ok_names,
        "bad_imgs": bad_imgs, "bad_names": bad_names,
        "bad_kinds": [kinds.get(n, "unlabelled") for n in bad_names],
        "syn_rows": syn_rows, "syn_imgs": syn_imgs, "syn_masks": syn_masks,
        "retouched": [gen.retouch_clean(im) for im in ok_imgs],
    }
    n_ok_frames = len({frame_of(n) for n in ok_names})
    n_bad_frames = len({frame_of(n) for n in bad_names})
    print(f"Test data: {len(ok_imgs)} good crops from {n_ok_frames} camera frames | {len(bad_imgs)} real NOT OK crops "
          f"from {n_bad_frames} frames | {len(syn_imgs)} synthetic defects")
    print("Copies of one frame are near-identical, so the confidence intervals below are narrower than the "
          "evidence really supports. Read the per-frame counts alongside them.")

    summaries = []
    for seed in a.seeds:
        if not checkpoint_path(a.encoder, seed).exists():
            print(f"\nNo checkpoint for seed {seed}: {checkpoint_path(a.encoder, seed)}. Skipping.")
            continue
        summaries.append(evaluate_seed(a.encoder, seed, out_dir, data))
    if not summaries:
        return

    keys = list(summaries[0].keys())
    numeric = [k for k in keys if k not in ("seed", "epoch")]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
        if len(summaries) > 1:
            writer.writerow({"seed": "mean", **{k: round(float(np.mean([s[k] for s in summaries])), 4) for k in numeric}})
            writer.writerow({"seed": "std", **{k: round(float(np.std([s[k] for s in summaries])), 4) for k in numeric}})

    if len(summaries) > 1:
        print(f"\n=== {a.encoder}: mean and spread over {len(summaries)} seeds ===")
        for k, label in [("false_reject_pct", "Good cartons rejected (%)"), ("real_recall_pct", "Real NOT OK caught (%)"),
                         ("real_auroc", "Real AUROC"), ("synthetic_recall_pct", "Synthetic defects caught (%)"),
                         ("pixel_auroc", "Pixel AUROC"), ("retouched_reject_pct", "Retouched good rejected (%)"),
                         ("ms_median", "Time per carton (ms)")]:
            v = [s[k] for s in summaries]
            print(f"  {label:<30} {np.mean(v):8.3f} +/- {np.std(v):.3f}")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
