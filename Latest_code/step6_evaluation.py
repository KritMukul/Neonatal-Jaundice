# =============================================================================
# step6_evaluation.py
# Purpose : Evaluate all 4 models on the test set and reproduce all paper
#           tables and figures.
#
# Metrics (paper Section "Results"):
#   Accuracy, Precision, Specificity, Recall (Sensitivity), F1-Score, MCC, AUC
#   95% CI:
#     Accuracy    → Wilson method
#     Sensitivity / Specificity → Clopper-Pearson
#   Confusion matrices  (paper Fig. 12)
#   ROC curves          (paper Fig. 11)
#   Bar chart           (paper Fig. 13)
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
import joblib
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

from config import (TRAIN_CSV, TEST_CSV, MODELS_DIR, RESULTS_DIR,
                    IMAGE_SIZE, BATCH_SIZE, T2TVT_MODEL_NAME,
                    IMAGENET_MEAN, IMAGENET_STD, DECISION_THRESHOLD,
                    RANDOM_SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Dataset & transform ───────────────────────────────────────────────────

class NeonatalDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        if "preprocessed_path" in self.df.columns:
            self.df["_path"] = self.df["preprocessed_path"]
        else:
            from config import IMAGE_DIR
            self.df["_path"] = self.df["image_idx"].apply(
                lambda x: os.path.join(IMAGE_DIR, x))
        self.df = self.df[self.df["_path"].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["_path"]).convert("RGB")
        label = float(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Confidence interval helpers ───────────────────────────────────────────

def wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95):
    """Wilson score interval for accuracy (paper: Wilson method)."""
    if n_total == 0:
        return 0.0, 0.0
    z   = stats.norm.ppf(1 - (1 - confidence) / 2)
    p   = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95):
    """Clopper-Pearson exact CI for sensitivity/specificity (paper)."""
    alpha = 1 - confidence
    if k == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lower, upper


# ── Get predictions from deep learning models ─────────────────────────────

@torch.no_grad()
def get_dl_predictions(model, loader, device):
    """Returns (y_true, y_prob) numpy arrays."""
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


# ── Feature extraction for classical models ───────────────────────────────

def extract_features_single(image_path: str) -> np.ndarray | None:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    ycrcb  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_ch   = ycrcb[:, :, 0].ravel()
    cr_ch  = ycrcb[:, :, 1].ravel()
    cb_ch  = ycrcb[:, :, 2].ravel()
    feats = []
    for ch in range(3):
        c = hsv[:, :, ch].ravel()
        feats += [c.mean(), c.std()]
    feats += [y_ch.mean(), y_ch.std(), cb_ch.mean(), cb_ch.std(),
              cr_ch.mean(), cr_ch.std()]
    feats += [float(skew(y_ch)), float(kurtosis(y_ch))]
    return np.array(feats, dtype=np.float32)


def load_features_for_eval(csv_path: str):
    df = pd.read_csv(csv_path)
    if "preprocessed_path" in df.columns:
        paths = df["preprocessed_path"].tolist()
    else:
        from config import IMAGE_DIR
        paths = [os.path.join(IMAGE_DIR, x) for x in df["image_idx"].tolist()]
    X, y = [], []
    for path, label in zip(paths, df["label"].tolist()):
        feat = extract_features_single(path)
        if feat is not None:
            X.append(feat)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Compute full metric suite ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = DECISION_THRESHOLD) -> dict:
    """
    Paper metrics: Accuracy, Precision, Specificity, Recall, F1, MCC, AUC
    + 95% CIs (Wilson for Accuracy; Clopper-Pearson for Sensitivity/Specificity)
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc        = accuracy_score(y_true, y_pred)
    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)       # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1         = f1_score(y_true, y_pred, zero_division=0)
    mcc        = matthews_corrcoef(y_true, y_pred)
    auc        = roc_auc_score(y_true, y_prob)

    n_total    = len(y_true)
    n_correct  = int((y_pred == y_true).sum())

    # 95% CIs
    acc_lo, acc_hi   = wilson_ci(n_correct, n_total)
    sens_lo, sens_hi = clopper_pearson_ci(int(tp), int(tp + fn))
    spec_lo, spec_hi = clopper_pearson_ci(int(tn), int(tn + fp))

    return {
        "Accuracy":      acc,
        "Acc_CI":       (acc_lo, acc_hi),
        "Precision":    precision,
        "Specificity":  specificity,
        "Spec_CI":      (spec_lo, spec_hi),
        "Recall":       recall,
        "Recall_CI":    (sens_lo, sens_hi),
        "F1":           f1,
        "MCC":          mcc,
        "AUC":          auc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ── Plotting helpers ──────────────────────────────────────────────────────

def plot_confusion_matrices(all_results: dict):
    """Paper Fig. 12 — 2×2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    model_names = list(all_results.keys())

    for ax, name in zip(axes, model_names):
        r  = all_results[name]
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Jaundiced", "Jaundiced"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=13, fontweight="bold")

    plt.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_roc_curves(roc_data: dict):
    """Paper Fig. 11 — ROC curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"T2T-ViT": "blue", "ResNet-50": "orange",
               "SVM": "green", "k-NN": "red"}

    # Left: T2T-ViT and ResNet-50
    for name in ["T2T-ViT", "ResNet-50"]:
        if name in roc_data:
            fpr, tpr, auc_val = roc_data[name]
            axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})",
                         color=colors[name], linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title("ROC Curves — Deep Learning Models", fontsize=12)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(True)

    # Right: all models together
    for name in ["T2T-ViT", "ResNet-50", "SVM", "k-NN"]:
        if name in roc_data:
            fpr, tpr, auc_val = roc_data[name]
            axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})",
                         color=colors[name], linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set_title("ROC Curves — All Models", fontsize=12)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_metric_bars(summary_df: pd.DataFrame):
    """Paper Fig. 13 — Bar chart comparing all models across metrics."""
    metric_cols = ["Accuracy", "Precision", "Specificity", "Recall", "F1", "MCC", "AUC"]
    data = summary_df.set_index("Model")[metric_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    x     = np.arange(len(data.index))
    width = 0.11
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336",
              "#9C27B0", "#00BCD4", "#FF5722"]

    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width, data[col], width, label=col, color=colors[i])

    ax.set_xticks(x + width * len(metric_cols) / 2)
    ax.set_xticklabels(data.index, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance Comparison — All Models",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "metric_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── T2T-ViT architecture loader (mirrors step3 fallback logic) ────────────

T2TVIT_REPO         = "/workspace/Diya/T2T-ViT"
T2TVIT_WEIGHTS_PATH = "/workspace/Diya/models/T2T_ViT_14_pretrained.pth.tar"


def _load_t2tvit_for_eval() -> nn.Module | None:
    """Load T2T-ViT-14 architecture (no pretrained weights — we load our checkpoint)."""
    # Try timm first
    candidates = ["t2t_vit_14", "t2t_vit_t_14", "t2t_vit_14_resnext", "t2t_vit_7"]
    available  = timm.list_models()
    t2t_name   = next((n for n in candidates if n in available), None)
    if t2t_name is not None:
        print(f"  Architecture via timm: {t2t_name}")
        return timm.create_model(t2t_name, pretrained=False, num_classes=1)

    # Fall back to official repo
    import sys
    if not os.path.isdir(T2TVIT_REPO):
        print(f"  ERROR: T2T-ViT repo not found at {T2TVIT_REPO}")
        return None
    if T2TVIT_REPO not in sys.path:
        sys.path.insert(0, T2TVIT_REPO)
    try:
        from models.t2t_vit import t2t_vit_14
        print(f"  Architecture via official repo: {T2TVIT_REPO}")
        model = t2t_vit_14()
        model.head = nn.Linear(model.head.in_features, 1)
        return model
    except ImportError as e:
        print(f"  ERROR loading from official repo: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 6 — Evaluation (All Models)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    test_ds     = NeonatalDataset(TEST_CSV, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)
    y_true_dl   = np.array([test_ds.df.iloc[i]["label"] for i in range(len(test_ds))])

    all_metrics = {}
    roc_data    = {}

    # patient_id column — aligned with test_ds.df (shuffle=False keeps order)
    test_patient_ids = test_ds.df["patient_id"].values

    def patient_fusion_metrics(y_true_img, y_prob_img, patient_ids):
        """
        Average the 3 body-region probabilities per patient, then compute
        metrics on the resulting patient-level predictions.
        """
        df_img = pd.DataFrame({
            "patient_id": patient_ids,
            "prob":       y_prob_img,
            "label":      y_true_img,
        })
        df_pat = (df_img.groupby("patient_id")
                        .agg(prob=("prob", "mean"), label=("label", "first"))
                        .reset_index())
        return (compute_metrics(df_pat["label"].values, df_pat["prob"].values),
                df_pat["label"].values,
                df_pat["prob"].values)

    # ── T2T-ViT ────────────────────────────────────────────────────
    print("── Evaluating T2T-ViT …")
    t2t_path = os.path.join(MODELS_DIR, "t2tvit_best.pth")
    if os.path.exists(t2t_path):
        t2t_model = _load_t2tvit_for_eval()
        if t2t_model is not None:
            t2t_model.load_state_dict(torch.load(t2t_path, map_location=device))
            t2t_model.to(device)
            _, y_prob_t2t = get_dl_predictions(t2t_model, test_loader, device)

            # Image-level
            m = compute_metrics(y_true_dl, y_prob_t2t)
            all_metrics["T2T-ViT"] = m
            fpr, tpr, _ = roc_curve(y_true_dl, y_prob_t2t)
            roc_data["T2T-ViT"] = (fpr, tpr, m["AUC"])
            print(f"  T2T-ViT  [image]   Acc={m['Accuracy']:.4f}  AUC={m['AUC']:.4f}")

            # Patient-level fusion (average 3 body-region probs per patient)
            m_pat, y_true_pat, y_prob_pat = patient_fusion_metrics(
                y_true_dl, y_prob_t2t, test_patient_ids)
            all_metrics["T2T-ViT (patient)"] = m_pat
            fpr_p, tpr_p, _ = roc_curve(y_true_pat, y_prob_pat)
            roc_data["T2T-ViT (patient)"] = (fpr_p, tpr_p, m_pat["AUC"])
            print(f"  T2T-ViT  [patient] Acc={m_pat['Accuracy']:.4f}  AUC={m_pat['AUC']:.4f}")
        else:
            print("  SKIP: could not load T2T-ViT architecture")
    else:
        print(f"  SKIP: {t2t_path} not found")

    # ── ResNet-50 ──────────────────────────────────────────────────
    print("── Evaluating ResNet-50 …")
    rn_path = os.path.join(MODELS_DIR, "resnet50_best.pth")
    if os.path.exists(rn_path):
        rn_model = models.resnet50(weights=None)
        rn_model.fc = nn.Linear(rn_model.fc.in_features, 1)
        rn_model.load_state_dict(torch.load(rn_path, map_location=device))
        rn_model.to(device)
        _, y_prob_rn = get_dl_predictions(rn_model, test_loader, device)

        # Image-level
        m = compute_metrics(y_true_dl, y_prob_rn)
        all_metrics["ResNet-50"] = m
        fpr, tpr, _ = roc_curve(y_true_dl, y_prob_rn)
        roc_data["ResNet-50"] = (fpr, tpr, m["AUC"])
        print(f"  ResNet-50 [image]   Acc={m['Accuracy']:.4f}  AUC={m['AUC']:.4f}")

        # Patient-level fusion
        m_pat, y_true_pat, y_prob_pat = patient_fusion_metrics(
            y_true_dl, y_prob_rn, test_patient_ids)
        all_metrics["ResNet-50 (patient)"] = m_pat
        fpr_p, tpr_p, _ = roc_curve(y_true_pat, y_prob_pat)
        roc_data["ResNet-50 (patient)"] = (fpr_p, tpr_p, m_pat["AUC"])
        print(f"  ResNet-50 [patient] Acc={m_pat['Accuracy']:.4f}  AUC={m_pat['AUC']:.4f}")
    else:
        print(f"  SKIP: {rn_path} not found")

    # ── SVM + k-NN (classical) ─────────────────────────────────────
    print("── Evaluating SVM & k-NN …")
    svm_path     = os.path.join(MODELS_DIR, "svm_model.pkl")
    knn_path     = os.path.join(MODELS_DIR, "knn_model.pkl")
    scaler_path  = os.path.join(MODELS_DIR, "feature_scaler.pkl")

    if os.path.exists(svm_path) and os.path.exists(scaler_path):
        X_test, y_true_ml = load_features_for_eval(TEST_CSV)
        scaler    = joblib.load(scaler_path)
        X_test_sc = scaler.transform(X_test)

        svm_model   = joblib.load(svm_path)
        y_prob_svm  = svm_model.predict_proba(X_test_sc)[:, 1]
        m = compute_metrics(y_true_ml, y_prob_svm)
        all_metrics["SVM"] = m
        fpr, tpr, _ = roc_curve(y_true_ml, y_prob_svm)
        roc_data["SVM"] = (fpr, tpr, m["AUC"])
        print(f"  SVM       Acc={m['Accuracy']:.4f}  AUC={m['AUC']:.4f}")

        if os.path.exists(knn_path):
            knn_model  = joblib.load(knn_path)
            y_prob_knn = knn_model.predict_proba(X_test_sc)[:, 1]
            m = compute_metrics(y_true_ml, y_prob_knn)
            all_metrics["k-NN"] = m
            fpr, tpr, _ = roc_curve(y_true_ml, y_prob_knn)
            roc_data["k-NN"] = (fpr, tpr, m["AUC"])
            print(f"  k-NN      Acc={m['Accuracy']:.4f}  AUC={m['AUC']:.4f}")
    else:
        print("  SKIP: classical models not found")

    # ── Build summary tables (paper Tables 3, 4, 5, 6) ────────────
    rows = []
    for model_name, m in all_metrics.items():
        acc_lo, acc_hi     = m["Acc_CI"]
        sens_lo, sens_hi   = m["Recall_CI"]
        spec_lo, spec_hi   = m["Spec_CI"]
        rows.append({
            "Model":       model_name,
            "Accuracy":    round(m["Accuracy"],    3),
            "Acc_95CI":    f"({acc_lo:.3f}–{acc_hi:.3f})",
            "Precision":   round(m["Precision"],   3),
            "Specificity": round(m["Specificity"], 3),
            "Spec_95CI":   f"({spec_lo:.3f}–{spec_hi:.3f})",
            "Recall":      round(m["Recall"],      3),
            "Recall_95CI": f"({sens_lo:.3f}–{sens_hi:.3f})",
            "F1":          round(m["F1"],   3),
            "MCC":         round(m["MCC"],  3),
            "AUC":         round(m["AUC"],  3),
        })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("RESULTS — Performance Summary (paper Table 4)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Save to CSV
    summary_csv = os.path.join(RESULTS_DIR, "all_models_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    # ── Plots ──────────────────────────────────────────────────────
    if all_metrics:
        plot_confusion_matrices(all_metrics)
        plot_roc_curves(roc_data)
        plot_metric_bars(summary_df)

    print("\nSTEP 6 complete. All results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
