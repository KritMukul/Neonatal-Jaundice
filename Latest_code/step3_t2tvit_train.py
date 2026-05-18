#!/bin/bash
# =============================================================================
# run_all.sh  —  Run the complete pipeline end-to-end
# Server: root@dgxhnode5:/workspace/Diya/Latest_code/
#
# Usage:
#   cd /workspace/Diya/Latest_code
#   chmod +x run_all.sh
#   ./run_all.sh
# =============================================================================

set -e   # exit immediately on error

CODE_DIR="/workspace/Diya/Latest_code"
LOG_DIR="/workspace/Diya/results"
mkdir -p "$LOG_DIR"

cd "$CODE_DIR"

echo "======================================================"
echo " Neonatal Jaundice Detection — Full Pipeline"
echo " $(date)"
echo "======================================================"

# ── Step 1: Dataset preparation ─────────────────────────────────────────
echo ""
echo "[1/6] Dataset preparation (80/20 split, 15 mg/dL threshold)..."
python step1_prepare_dataset.py 2>&1 | tee "$LOG_DIR/step1.log"

# ── Step 2: Preprocessing ───────────────────────────────────────────────
echo ""
echo "[2/6] Image preprocessing (color calibration, CLAHE, skin segmentation)..."
python step2_preprocessing.py 2>&1 | tee "$LOG_DIR/step2.log"

# ── Step 3: T2T-ViT training ────────────────────────────────────────────
echo ""
echo "[3/6] Training T2T-ViT-14 (main model, 300 epochs)..."
python step3_t2tvit_train.py 2>&1 | tee "$LOG_DIR/step3_t2tvit.log"

# ── Step 4: ResNet-50 training ───────────────────────────────────────────
echo ""
echo "[4/6] Training ResNet-50 (baseline, 300 epochs)..."
python step4_resnet50_train.py 2>&1 | tee "$LOG_DIR/step4_resnet50.log"

# ── Step 5: SVM + k-NN training ──────────────────────────────────────────
echo ""
echo "[5/6] Training SVM and k-NN (classical baselines)..."
python step5_svm_knn.py 2>&1 | tee "$LOG_DIR/step5_classical.log"

# ── Step 6: Evaluation ───────────────────────────────────────────────────
echo ""
echo "[6/6] Evaluating all models and generating reports..."
python step6_evaluation.py 2>&1 | tee "$LOG_DIR/step6_evaluation.log"

echo ""
echo "======================================================"
echo " Pipeline complete! $(date)"
echo " Results saved to: /workspace/Diya/results/"
echo " Models  saved to: /workspace/Diya/models/"
echo "======================================================"
