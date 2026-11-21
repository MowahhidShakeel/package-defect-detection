import os
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm import tqdm
from config import (
    CROPS_DIR, SYNTHETIC_DIR, IMPACT_DIR, WEIGHTS_DIR,
    EMBEDDING_DIM, TRIPLET_MARGIN, PRETRAIN_EPOCHS, FINETUNE_EPOCHS,
    BATCH_SIZE, LEARNING_RATE, DEVICE, SEED
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================================================================
# 1. RESNET-34 EMBEDDING ARCHITECTURE
# ==============================================================================
class ResNet34Embedder(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, imagenet_init: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if imagenet_init else None
        backbone = models.resnet34(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        emb = self.head(feats)
        return F.normalize(emb, p=2, dim=-1)

def img_to_tensor(img_gray: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_gray).float().unsqueeze(0) / 255.0
    return t.repeat(3, 1, 1)

# ==============================================================================
# 2. IMPACT WORD TRIPLET DATASET (PRE-TRAINING)
# ==============================================================================
class ImpactTripletDataset(Dataset):
    def __init__(self, word_crops_dir: Path):
        self.groups = {}
        for p in sorted(word_crops_dir.glob("*.png")):
            prefix = "_".join(p.stem.split("_")[:2])
            self.groups.setdefault(prefix, []).append(p)
        self.keys = [k for k, v in self.groups.items() if len(v) >= 2]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        group = self.groups[k]
        pos_candidates = [p for p in group if "neg" not in p.stem]
        neg_candidates = [p for p in group if "neg" in p.stem]

        anchor_p = pos_candidates[0]
        pos_p = pos_candidates[1] if len(pos_candidates) > 1 else anchor_p
        if neg_candidates:
            neg_p = random.choice(neg_candidates)
        else:
            other_k = random.choice([ok for ok in self.keys if ok != k])
            neg_p = random.choice(self.groups[other_k])

        a = img_to_tensor(cv2.imread(str(anchor_p), cv2.IMREAD_GRAYSCALE))
        p = img_to_tensor(cv2.imread(str(pos_p), cv2.IMREAD_GRAYSCALE))
        n = img_to_tensor(cv2.imread(str(neg_p), cv2.IMREAD_GRAYSCALE))
        return a, p, n

# ==============================================================================
# 3. TAPAL TRIPLET DATASET (REFERENCE, GOOD SAME-PRINT, SYNTHETIC DEFECTIVE)
# ==============================================================================
class TapalTripletDataset(Dataset):
    def __init__(self, split_name: str = "train_ok"):
        self.ok_paths = sorted((CROPS_DIR / split_name).glob("*.png"))
        meta_csv = SYNTHETIC_DIR / f"{split_name}_metadata.csv"
        df = pd.read_csv(meta_csv)

        self.defects_by_source = {}
        for _, row in df.iterrows():
            self.defects_by_source.setdefault(Path(row["clean_source"]).name, []).append(row["defect_image"])

        # Group OK crops by session so positive pairs share identical timestamp print
        self.session_groups = {}
        for p in self.ok_paths:
            sess = "_".join(p.stem.split("_")[3:]) or "default"
            self.session_groups.setdefault(sess, []).append(p)

    def __len__(self):
        return len(self.ok_paths) * 4  # 4 triplets per anchor per epoch

    def __getitem__(self, idx):
        anchor_path = self.ok_paths[idx % len(self.ok_paths)]
        sess = "_".join(anchor_path.stem.split("_")[3:]) or "default"
        same_sess_pool = [p for p in self.session_groups.get(sess, []) if p != anchor_path]
        pos_path = random.choice(same_sess_pool) if same_sess_pool else random.choice(self.ok_paths)

        neg_list = self.defects_by_source.get(anchor_path.name, [])
        if neg_list:
            neg_path = random.choice(neg_list)
        else:
            random_ok = random.choice(self.ok_paths)
            neg_path = random.choice(self.defects_by_source[random_ok.name])

        a_img = cv2.imread(str(anchor_path), cv2.IMREAD_GRAYSCALE)
        p_img = cv2.imread(str(pos_path), cv2.IMREAD_GRAYSCALE)
        n_img = cv2.imread(str(neg_path), cv2.IMREAD_GRAYSCALE)

        return img_to_tensor(a_img), img_to_tensor(p_img), img_to_tensor(n_img)

# ==============================================================================
# 4. TRAINING & REFERENCE BANK EXPORT
# ==============================================================================
def pretrain_on_impact() -> Path:
    word_dir = IMPACT_DIR / "word_crops"
    ds = ImpactTripletDataset(word_dir)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = ResNet34Embedder(imagenet_init=True).to(DEVICE)
    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"[Stage 3b] Pre-training ResNet-34 on IMPACT word crops ({len(ds)} word groups)...")
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for a, p, n in loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == PRETRAIN_EPOCHS:
            print(f"  [IMPACT Pre-train] Epoch {epoch:02d}/{PRETRAIN_EPOCHS} | Triplet Loss: {total_loss/len(loader):.4f}")

    out_path = WEIGHTS_DIR / "resnet34_impact_pretrained.pt"
    torch.save(model.state_dict(), out_path)
    return out_path

def finetune_on_tapal(use_impact_pretrain: bool = True):
    tag = "with_impact" if use_impact_pretrain else "no_impact"
    model = ResNet34Embedder(imagenet_init=True).to(DEVICE)

    if use_impact_pretrain:
        impact_ckpt = WEIGHTS_DIR / "resnet34_impact_pretrained.pt"
        if not impact_ckpt.exists():
            impact_ckpt = pretrain_on_impact()
        model.load_state_dict(torch.load(impact_ckpt, map_location=DEVICE))
        print("[Stage 3b] Loaded IMPACT pre-trained weights for Tapal fine-tuning.")
    else:
        print("[Stage 3b Ablation] Training ResNet-34 on Tapal WITHOUT IMPACT pre-training.")

    train_ds = TapalTripletDataset("train_ok")
    val_ds = TapalTripletDataset("val_ok")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    best_val_loss = float("inf")
    best_model_path = WEIGHTS_DIR / f"resnet34_tapal_{tag}.pt"

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            optimizer.zero_grad()
            emb_a, emb_p, emb_n = model(a), model(p), model(n)
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                val_loss += criterion(model(a), model(p), model(n)).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == FINETUNE_EPOCHS:
            print(f"  [Tapal Fine-Tune ({tag})] Epoch {epoch:02d}/{FINETUNE_EPOCHS} | "
                  f"Train Loss: {tr_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    # Build and save the Reference Embedding Bank from all clean training OK crops
    # (Covers slight timestamp variations across sessions)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    ref_embeddings = []
    with torch.no_grad():
        for p in sorted((CROPS_DIR / "train_ok").glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            t = img_to_tensor(img).unsqueeze(0).to(DEVICE)
            ref_embeddings.append(model(t).cpu())
    ref_bank = torch.cat(ref_embeddings, dim=0)  # Shape: (N_train_ok, EMBEDDING_DIM)
    bank_path = WEIGHTS_DIR / f"reference_bank_{tag}.pt"
    torch.save(ref_bank, bank_path)
    print(f"[Stage 3b Saved] Model: {best_model_path} | Reference Bank ({ref_bank.shape[0]} anchors): {bank_path}")

if __name__ == "__main__":
    finetune_on_tapal(use_impact_pretrain=True)
    finetune_on_tapal(use_impact_pretrain=False)  # Automatically runs the ablation model