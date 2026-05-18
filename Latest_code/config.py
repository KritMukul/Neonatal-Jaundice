# =============================================================================
# config.py  —  Central configuration for all steps
# Paper: "Neonatal jaundice detection using a vision transformer-based DL model"
# Server: root@dgxhnode5:/workspace/Diya/Latest_code/
# =============================================================================

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR          = "/workspace/Diya"
IMAGE_DIR         = os.path.join(BASE_DIR, "NeoJaundice", "NeoJaundice", "images")
CSV_PATH          = os.path.join(BASE_DIR, "NeoJaundice", "NeoJaundice", "chd_jaundice_published_2.csv")
PREPROCESSED_DIR  = os.path.join(BASE_DIR, "preprocessed_data")
MODELS_DIR        = os.path.join(BASE_DIR, "models")
RESULTS_DIR       = os.path.join(BASE_DIR, "results")
CODE_DIR          = os.path.join(BASE_DIR, "Latest_code")

# Split CSVs saved here
TRAIN_CSV         = os.path.join(CODE_DIR, "train_split.csv")
VAL_CSV           = os.path.join(CODE_DIR, "val_split.csv")
TEST_CSV          = os.path.join(CODE_DIR, "test_split.csv")

# ── Clinical threshold (paper Section: Ground truth labeling) ──────────────
BILIRUBIN_THRESHOLD = 15.0   # mg/dL  →  >=15 = jaundiced (label 1)

# ── Image ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224            # both T2T-ViT and ResNet expect 224×224

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 0.0001        # Adam lr
EPOCHS        = 100           # Larger dataset (6705 images) converges faster;
                              # early stopping handles the rest
TRAIN_SPLIT   = 0.70          # 70% train / 10% val / 20% test  (patient-level)
VAL_SPLIT     = 0.10
RANDOM_SEED   = 42
EARLY_STOPPING_PATIENCE = 20  # halt if val AUC doesn't improve for 20 epochs

# ── Class imbalance weights ────────────────────────────────────────────────
# Dataset: 1302 non-jaundice patients | 933 jaundice patients
# pos_weight up-weights minority (jaundice) class in BCEWithLogitsLoss
POS_WEIGHT    = 1302.0 / 933.0   # ≈ 1.396

# ── T2T-ViT-14 architecture (paper Section: Tokens-to-Token ViT) ───────────
# timm model name — auto-resolved at runtime in step3, this is the preferred name
# Fallback order: t2t_vit_14 → t2t_vit_t_14 → t2t_vit_14_resnext → vit_base_patch16_224
T2TVT_MODEL_NAME   = "t2t_vit_14"
T2TVT_PATCH_SIZES  = [7, 3, 3]      # overlapping patch embedding
T2TVT_STRIDES      = [1, 1, 3]
T2TVT_HIDDEN_DIM   = 384
T2TVT_MLP_SIZE     = 1536
T2TVT_NUM_LAYERS   = 14

# ── SVM / k-NN (paper Section: Baseline models) ────────────────────────────
SVM_PARAM_GRID = {
    "C":     [0.01, 0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1, "scale"],
}
SVM_CV_FOLDS  = 5
KNN_K_RANGE   = list(range(1, 11))   # k = 1 … 10

# ── Skin detection thresholds (HSV + YCbCr) ────────────────────────────────
# OpenCV: H 0-180, S 0-255, V 0-255
HSV_LOWER  = (0,   20,  60)
HSV_UPPER  = (25, 200, 255)

# YCbCr (OpenCV)
YCBCR_LOWER = (80,  77, 133)
YCBCR_UPPER = (240, 127, 173)

# ── CLAHE ──────────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# ── Data augmentation (paper Section: Data augmentation techniques) ─────────
AUG_ROTATION_DEG  = 15
AUG_BRIGHTNESS    = 0.3
AUG_CONTRAST      = 0.3
AUG_SATURATION    = 0.2    # skin colour variation across cameras
AUG_HUE           = 0.05   # ±hue shift — critical for jaundice yellow detection
AUG_SCALE_MIN     = 0.75   # wider zoom range for RandomResizedCrop
AUG_SCALE_MAX     = 1.00
AUG_SHEAR         = 10     # RandomAffine shear — skin angle variation
AUG_BLUR_SIGMA    = (0.1, 1.0)   # GaussianBlur — simulates camera focus variation
AUG_ERASING_PROB  = 0.2    # RandomErasing — simulates partial occlusion

# ── ImageNet normalisation (for pretrained models) ─────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Evaluation ─────────────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.5
