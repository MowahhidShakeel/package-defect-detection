import os
import time
import pickle
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from config import (
    CROPS_DIR, SYNTHETIC_DIR, WEIGHTS_DIR, RESULTS_DIR, HEATMAPS_DIR,
    DEFECT_TYPES, SEVERITIES, TARGET_FAR_VAL, DEVICE
)
from train_baselines import extract_hog_lbp_features, ResNet18BinaryClassifier
from train_resnet34_embedding import ResNet34Embedder, img_to_tensor

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """Exact 95% Clopper-Pearson confidence interval for k successes out of n trials."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi

def compute_embedding_score_and_heatmap(
    model: ResNet34Embedder, ref_bank: torch.Tensor, img_gray: np.ndarray, need_heatmap: bool = False
):
    """
    Computes distance to nearest reference anchor in feature space.
    If need_heatmap=True, computes spatial gradient sensitivity map of the embedding distance
    with respect to input pixels (Stage 4 gradient map of the embedding distance).
    """
    x = img_to_tensor(img_gray).unsqueeze(0).to(DEVICE)
    ref_bank = ref_bank.to(DEVICE)

    if not need_heatmap:
        with torch.no_grad():
            emb = model(x)
            dists = torch.cdist(emb, ref_bank, p=2).squeeze(0)
            score = float(torch.min(dists).item())
        return score, None

    x.requires_grad_(True)
    emb = model(x)
    dists = torch.cdist(emb, ref_bank, p=2).squeeze(0)
    min_idx = torch.argmin(dists)
    min_dist = dists[min_idx]

    model.zero_grad()
    min_dist.backward()

    # Aggregate gradient magnitude across channels and smooth into a spatial defect heatmap
    grad_map = x.grad.detach().abs().max(dim=1)[0].squeeze(0).cpu().numpy()
    grad_map = cv2.GaussianBlur(grad_map, (11, 11), 3.0)
    if grad_map.max() > grad_map.min():
        grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min())
    return float(min_dist.item()), grad_map

def save_heatmap_overlay(img_gray: np.ndarray, heatmap: np.ndarray, score: float, thr: float, out_path: Path):
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    hm_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, hm_color, 0.45, 0)
    status = "Not OK" if score >= thr else "OK"
    color = (0, 0, 255) if status == "Not OK" else (0, 200, 0)
    cv2.putText(overlay, f"{status} ({score:.3f})", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    cv2.imwrite(str(out_path), np.vstack([img_bgr, overlay]))

def evaluate_all_models():
    val_ok_paths = sorted((CROPS_DIR / "val_ok").glob("*.png"))
    test_ok_paths = sorted((CROPS_DIR / "test_ok").glob("*.png"))
    test_real_not_ok_paths = sorted((CROPS_DIR / "test_not_ok").glob("*.png"))
    synth_test_df = pd.read_csv(SYNTHETIC_DIR / "test_ok_metadata.csv")

    # Load models
    models_dict = {}

    # 1. ResNet-34 (with IMPACT pre-training)
    for tag, label in [("with_impact", "ResNet-34 Triplet (IMPACT Pre-trained)"),
                       ("no_impact", "ResNet-34 Triplet (No IMPACT Pre-train)")]:
        w_path = WEIGHTS_DIR / f"resnet34_tapal_{tag}.pt"
        b_path = WEIGHTS_DIR / f"reference_bank_{tag}.pt"
        if w_path.exists() and b_path.exists():
            m = ResNet34Embedder(imagenet_init=False).to(DEVICE)
            m.load_state_dict(torch.load(w_path, map_location=DEVICE))
            m.eval()
            bank = torch.load(b_path, map_location=DEVICE)
            models_dict[label] = ("embedding", m, bank)

    # 2. ResNet-18 Binary Classifier Baseline
    r18_path = WEIGHTS_DIR / "baseline_resnet18_classifier.pt"
    if r18_path.exists():
        r18 = ResNet18BinaryClassifier(pretrained=False).to(DEVICE)
        r18.load_state_dict(torch.load(r18_path, map_location=DEVICE))
        r18.eval()
        models_dict["Baseline: ResNet-18 Classifier"] = ("resnet18", r18, None)

    # 3. HOG + LBP + SVM Baseline
    svm_path = WEIGHTS_DIR / "baseline_hog_lbp_svm.pkl"
    if svm_path.exists():
        with open(svm_path, "rb") as f:
            svm_clf = pickle.load(f)
        models_dict["Baseline: HOG + LBP + SVM"] = ("svm", svm_clf, None)

    summary_rows = []
    severity_tables = {}

    for model_name, (mtype, model_obj, ref_bank) in models_dict.items():
        print(f"\n=== Evaluating: {model_name} ===")

        def score_image(img_gray: np.ndarray, want_hm: bool = False):
            t0 = time.perf_counter()
            if mtype == "embedding":
                s, hm = compute_embedding_score_and_heatmap(model_obj, ref_bank, img_gray, need_heatmap=want_hm)
            elif mtype == "resnet18":
                with torch.no_grad():
                    x = img_to_tensor(img_gray).unsqueeze(0).to(DEVICE)
                    s = float(torch.sigmoid(model_obj(x)).item())
                hm = None
            else:
                feat = extract_hog_lbp_features(img_gray).reshape(1, -1)
                s = float(model_obj.predict_proba(feat)[0, 1])
                hm = None
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return s, hm, dt_ms

        # 1. Calibrate operating threshold on val_ok (95th percentile of OK scores)
        val_ok_scores = [score_image(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))[0] for p in val_ok_paths]
        threshold = float(np.percentile(val_ok_scores, (1.0 - TARGET_FAR_VAL) * 100.0))

        # 2. Score Test OK & Test Real Not_OK
        latencies = []
        test_ok_scores = []
        for p in test_ok_paths:
            s, _, dt = score_image(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
            test_ok_scores.append(s)
            latencies.append(dt)

        real_def_scores = []
        for idx, p in enumerate(test_real_not_ok_paths):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            want_hm = (mtype == "embedding" and "IMPACT Pre-trained" in model_name)
            s, hm, dt = score_image(img, want_hm=want_hm)
            real_def_scores.append(s)
            latencies.append(dt)
            if want_hm and hm is not None:
                save_heatmap_overlay(img, hm, s, threshold, HEATMAPS_DIR / f"real_defect_{idx:03d}.png")

        # Real Not_OK Recall, False Accept Rate (Miss Rate), and 95% Clopper-Pearson Interval
        n_real_def = len(real_def_scores)
        n_caught = int(np.sum(np.array(real_def_scores) >= threshold))
        n_missed = n_real_def - n_caught
        real_recall = n_caught / max(1, n_real_def)
        far_real = n_missed / max(1, n_real_def)
        far_ci_lo, far_ci_hi = clopper_pearson_ci(n_missed, n_real_def)

        # Real AUROC
        y_true_real = [0] * len(test_ok_scores) + [1] * len(real_def_scores)
        y_score_real = test_ok_scores + real_def_scores
        auroc_real = roc_auc_score(y_true_real, y_score_real) if len(set(y_true_real)) > 1 else float("nan")

        # 3. Synthetic Benchmark Evaluation (6 Types x 3 Severities + Pixel AUROC)
        synth_scores = []
        pixel_aurocs = []
        sev_matrix = {dt: {sev: [] for sev in SEVERITIES} for dt in DEFECT_TYPES}

        for idx, row in tqdm(synth_test_df.iterrows(), total=len(synth_test_df), desc="Synthetic Benchmark"):
            img = cv2.imread(row["defect_image"], cv2.IMREAD_GRAYSCALE)
            want_hm = (mtype == "embedding" and "IMPACT Pre-trained" in model_name and idx < 150)
            s, hm, _ = score_image(img, want_hm=want_hm)
            synth_scores.append(s)
            sev_matrix[row["defect_type"]][int(row["severity"])].append(int(s >= threshold))

            if want_hm and hm is not None:
                mask = cv2.imread(row["mask_image"], cv2.IMREAD_GRAYSCALE)
                mask_bin = (mask > 127).astype(np.uint8).ravel()
                if 0 < mask_bin.sum() < len(mask_bin):
                    pixel_aurocs.append(roc_auc_score(mask_bin, hm.ravel()))
                if idx < 20:
                    save_heatmap_overlay(img, hm, s, threshold, HEATMAPS_DIR / f"synth_{idx:03d}_{row['defect_type']}.png")

        y_true_synth = [0] * len(test_ok_scores) + [1] * len(synth_scores)
        auroc_synth = roc_auc_score(y_true_synth, test_ok_scores + synth_scores)
        mean_pixel_auroc = float(np.mean(pixel_aurocs)) if pixel_aurocs else float("nan")

        # Format 6x3 table
        df_sev = pd.DataFrame({
            f"Severity {sev}": {dt: np.mean(sev_matrix[dt][sev]) * 100.0 for dt in DEFECT_TYPES}
            for sev in SEVERITIES
        })
        df_sev["Overall Type Recall (%)"] = df_sev.mean(axis=1)
        severity_tables[model_name] = df_sev
        df_sev.to_csv(RESULTS_DIR / f"severity_recall_{mtype}_{len(summary_rows)}.csv")

        summary_rows.append({
            "Model": model_name,
            "Synth AUROC": f"{auroc_synth:.4f}",
            "Real AUROC": f"{auroc_real:.4f}",
            "Real Defect Recall": f"{real_recall*100:.2f}% ({n_caught}/{n_real_def})",
            "False-Accept Rate (Miss)": f"{far_real*100:.2f}% (95% CI: [{far_ci_lo*100:.1f}%, {far_ci_hi*100:.1f}%])",
            "Pixel AUROC (Loc)": f"{mean_pixel_auroc:.4f}" if not np.isnan(mean_pixel_auroc) else "N/A",
            "Latency (ms/crop)": f"{np.mean(latencies):.2f} ms"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "benchmark_summary_table.csv", index=False)
    print("\n================ FINAL BENCHMARK SUMMARY ================")
    print(summary_df.to_string(index=False))

    for name, table in severity_tables.items():
        print(f"\n--- Per-Type & Per-Severity Recall (%): {name} ---")
        print(table.round(2).to_string())

if __name__ == "__main__":
    evaluate_all_models()