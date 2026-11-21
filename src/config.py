import os
from pathlib import Path
import torch

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
REPO_ROOT = Path(r"C:\Mowahhid\Projects_Repo\package-defect-detection")

TAPAL_OK_RAW_DIR = Path(
    r"C:\Mowahhid\LUMS\MS-Thesis\Project_Repo\Mowahhid_Project\FinalVersion"
    r"\datasets\Pattern_detection\ok-20251130T193428Z-1-001-20251222T074643Z-1-001"
    r"\ok-20251130T193428Z-1-001\ok"
)
TAPAL_NOT_OK_RAW_DIR = Path(
    r"C:\Mowahhid\LUMS\MS-Thesis\Project_Repo\Mowahhid_Project\FinalVersion"
    r"\datasets\Pattern_detection\not_ok-20251219T070643Z-1-001\not_ok"
)

DATA_DIR = REPO_ROOT / "data"
RAW_FILTERED_DIR = DATA_DIR / "tapal_filtered"
YOLO_DATASET_DIR = DATA_DIR / "yolo_print_region"
EXPDATE_DIR = DATA_DIR / "expdate"
IMPACT_DIR = DATA_DIR / "impact_words"

CROPS_DIR = DATA_DIR / "crops_normalized"       # Stage 2 outputs (64px height, grayscale)
SYNTHETIC_DIR = DATA_DIR / "synthetic_benchmark" # 6 defect types x 3 severities + masks

WEIGHTS_DIR = REPO_ROOT / "weights"
RESULTS_DIR = REPO_ROOT / "results"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"

for d in [DATA_DIR, RAW_FILTERED_DIR, YOLO_DATASET_DIR, EXPDATE_DIR, IMPACT_DIR,
          CROPS_DIR, SYNTHETIC_DIR, WEIGHTS_DIR, RESULTS_DIR, HEATMAPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. PIPELINE HYPERPARAMETERS
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Stage 0 & Stage 1 (YOLOv8n Locator)
STAGE0_IMG_SIZE = (640, 640)
YOLO_EPOCHS = 40
YOLO_BATCH_SIZE = 16
YOLO_TARGET_MAP50 = 0.95

# Stage 2 (Crop Normalisation)
NORM_HEIGHT = 64
NORM_MAX_WIDTH = 256  # Aspect ratio is preserved; right-padded to NORM_MAX_WIDTH for batching

# Synthetic Defect Taxonomy (Lu et al. 2021 / Pedersen et al. 2016)
DEFECT_TYPES = [
    "fade",
    "smudge",
    "broken_strokes",
    "missing_line",
    "character_swap",
    "spacing_shift",
]
SEVERITIES = [1, 2, 3]  # 1: Mild, 2: Medium, 3: Severe

# Stage 3b (ResNet-34 Embedding Verifier)
EMBEDDING_DIM = 128
TRIPLET_MARGIN = 0.5
PRETRAIN_EPOCHS = 15
FINETUNE_EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Stage 4 (Decision Threshold Target)
TARGET_FAR_VAL = 0.05  # Choose validation threshold for 95% specificity / 5% false reject on OK