import os
import pickle
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from config import CROPS_DIR, SYNTHETIC_DIR, WEIGHTS_DIR, DEVICE, BATCH_SIZE, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# 1. CLASSICAL BASELINE: HOG + LBP + SVM
# ==============================================================================
def extract_hog_lbp_features(img_gray: np.ndarray) -> np.ndarray:
    hog_feat = hog(
        img_gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True
    )
    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)
    return np.concatenate([hog_feat, lbp_hist]).astype(np.float32)

def load_binary_split(split_name: str):
    ok_paths = sorted((CROPS_DIR / split_name).glob("*.png"))
    def_paths = sorted((SYNTHETIC_DIR / split_name / "images").glob("*.png"))
    # Subsample synthetic defects to 2x OK count for balanced classifier training
    if len(def_paths) > 2 * len(ok_paths):
        rng = np.random.default_rng(SEED)
        def_paths = list(rng.choice(def_paths, size=2 * len(ok_paths), replace=False))
    paths = [(p, 0) for p in ok_paths] + [(p, 1) for p in def_paths]  # 0 = OK, 1 = Defect
    return paths

def train_hog_lbp_svm():
    print("[Baseline 1] Extracting HOG + LBP features and training RBF-SVM...")
    train_items = load_binary_split("train_ok")
    X_train, y_train = [], []
    for p, label in tqdm(train_items, desc="HOG+LBP Feature Extraction"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            X_train.append(extract_hog_lbp_features(img))
            y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=5.0, probability=True, class_weight="balanced", random_state=SEED))
    ])
    clf.fit(X_train, y_train)

    svm_path = WEIGHTS_DIR / "baseline_hog_lbp_svm.pkl"
    with open(svm_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[Baseline 1 Saved] {svm_path}")

# ==============================================================================
# 2. DEEP CLASSIFIER BASELINE: FINE-TUNED RESNET-18 (OK vs Not_OK)
# ==============================================================================
class BinaryCropDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, label = self.items[idx]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        tensor = tensor.repeat(3, 1, 1)  # 3-channel input for pretrained ResNet
        return tensor, torch.tensor(label, dtype=torch.float32)

class ResNet18BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.net = models.resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_resnet18_classifier(epochs: int = 20):
    print("[Baseline 2] Training ResNet-18 binary classifier...")
    train_ds = BinaryCropDataset(load_binary_split("train_ok"))
    val_ds = BinaryCropDataset(load_binary_split("val_ok"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet18BinaryClassifier(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    out_weights = WEIGHTS_DIR / "baseline_resnet18_classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item() * len(y)

        val_loss /= max(1, len(val_ds))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_weights)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss/len(train_ds):.4f} | Val Loss: {val_loss:.4f}")

    print(f"[Baseline 2 Saved] {out_weights}")

if __name__ == "__main__":
    train_hog_lbp_svm()
    train_resnet18_classifier()