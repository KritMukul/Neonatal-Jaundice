# =============================================================================
# step5_svm_knn.py
# Purpose : Extract statistical features from preprocessed ROIs, then
#           train SVM (RBF) and k-NN classifiers.
#
# Paper (Section "Baseline models — SVM & k-NN"):
#   Features : mean, std, skewness, kurtosis of HSV and YCbCr channels
#              (skewness + kurtosis only on Y/luminance component)
#   Normalise: min-max scaling
#   SVM      : RBF kernel, grid search C & γ with 5-fold CV
#   k-NN     : Euclidean distance, distance-weighted, k = 1…10 → best k
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from config import (TRAIN_CSV, TEST_CSV, MODELS_DIR,
                    SVM_PARAM_GRID, SVM_CV_FOLDS, KNN_K_RANGE,
                    RANDOM_SEED)

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Feature extraction ────────────────────────────────────────────────────

def extract_features(image_path: str) -> np.ndarray | None:
    """
    Extract first-order statistical descriptors from the preprocessed ROI.

    Paper: "Feature vectors … consisted of first-order statistical descriptors,
            including the mean and standard deviation of HSV and YCbCr
            channels, as well as skewness and kurtosis of the luminance (Y)
            component."

    Feature vector (12 values):
        HSV   : mean_H, mean_S, mean_V, std_H, std_S, std_V
        YCbCr : mean_Y, mean_Cb, mean_Cr, std_Y, std_Cb, std_Cr
        Y-only: skewness_Y, kurtosis_Y
        → total 14 features
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_feats = []
    for ch in range(3):
        channel = hsv[:, :, ch].ravel()
        hsv_feats += [channel.mean(), channel.std()]

    # ── YCbCr (OpenCV: YCrCb → Y=0, Cr=1, Cb=2)
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_ch  = ycrcb[:, :, 0].ravel()
    cr_ch = ycrcb[:, :, 1].ravel()
    cb_ch = ycrcb[:, :, 2].ravel()
    ycbcr_feats = [
        y_ch.mean(),  y_ch.std(),
        cb_ch.mean(), cb_ch.std(),
        cr_ch.mean(), cr_ch.std(),
    ]

    # ── Skewness & kurtosis of Y (luminance)
    sk = float(skew(y_ch))
    ku = float(kurtosis(y_ch))   # Fisher definition (normal=0)

    feat_vec = np.array(hsv_feats + ycbcr_feats + [sk, ku], dtype=np.float32)
    return feat_vec


def load_features(csv_path: str):
    """Build feature matrix X and label vector y from a split CSV."""
    df = pd.read_csv(csv_path)

    if "preprocessed_path" in df.columns:
        paths = df["preprocessed_path"].tolist()
    else:
        from config import IMAGE_DIR
        paths = [os.path.join(IMAGE_DIR, x) for x in df["image_idx"].tolist()]

    X, y, valid_idx = [], [], []
    for i, (path, label) in enumerate(zip(paths, df["label"].tolist())):
        feat = extract_features(path)
        if feat is not None:
            X.append(feat)
            y.append(label)
            valid_idx.append(i)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── SVM ───────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    """
    Paper: "SVM with a radial basis function (RBF) kernel …
            hyperparameters (C and γ) were optimized through grid search
            with five-fold cross-validation."
    class_weight='balanced' automatically adjusts weights inversely
    proportional to class frequencies — equivalent to the paper's
    targeted augmentation on the minority class.
    """
    print("\n── SVM: Grid search (5-fold CV) ──────────────────────")
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED,
              class_weight="balanced")
    gs  = GridSearchCV(svm, SVM_PARAM_GRID, cv=SVM_CV_FOLDS,
                       scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best SVM params : {gs.best_params_}")
    print(f"Best CV F1      : {gs.best_score_*100:.2f}%")
    return gs.best_estimator_


# ── k-NN ──────────────────────────────────────────────────────────────────

def train_knn(X_train, y_train):
    """
    Paper: "value of k was systematically varied between 1 and 10, and
            the optimal setting was determined according to validation
            accuracy … distance-weighted voting was incorporated."
    Scored on F1 (not accuracy) to avoid imbalance bias in k selection.
    """
    print("\n── k-NN: Searching best k (1-10) ─────────────────────")
    best_k, best_f1 = 1, 0.0
    for k in KNN_K_RANGE:
        knn    = KNeighborsClassifier(n_neighbors=k,
                                      weights="distance",
                                      metric="euclidean")
        scores = cross_val_score(knn, X_train, y_train,
                                 cv=5, scoring="f1")
        mean_f1 = scores.mean()
        print(f"  k={k:2d}  CV F1={mean_f1*100:.2f}%")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k  = k

    print(f"Best k  : {best_k}  (CV F1={best_f1*100:.2f}%)")
    final_knn = KNeighborsClassifier(n_neighbors=best_k,
                                     weights="distance",
                                     metric="euclidean")
    final_knn.fit(X_train, y_train)
    return final_knn, best_k


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 5 — SVM & k-NN (Classical Baselines)")
    print("=" * 60)

    # ── Load features
    print("\nExtracting features from training set …")
    X_train, y_train = load_features(TRAIN_CSV)
    print(f"Train shape : X={X_train.shape}  y={y_train.shape}")
    print(f"  Jaundiced : {y_train.sum()}  |  Non-jaundiceed : {(y_train==0).sum()}")

    print("\nExtracting features from test set …")
    X_test, y_test = load_features(TEST_CSV)
    print(f"Test  shape : X={X_test.shape}   y={y_test.shape}")

    # ── Balance training set (paper: targeted augmentation on minority class)
    # Oversample the minority (jaundiced) class to match majority count
    X_maj = X_train[y_train == 0]
    y_maj = y_train[y_train == 0]
    X_min = X_train[y_train == 1]
    y_min = y_train[y_train == 1]

    if len(X_maj) != len(X_min):
        X_min_up, y_min_up = resample(X_min, y_min,
                                      replace=True,
                                      n_samples=len(X_maj),
                                      random_state=RANDOM_SEED)
        X_train = np.vstack([X_maj, X_min_up])
        y_train = np.hstack([y_maj, y_min_up])
        print(f"Balanced train  : {(y_train==0).sum()} non-jaundiced  "
              f"| {(y_train==1).sum()} jaundiced")

    # ── Min-max normalisation (paper: "All features were normalized using
    #    min–max scaling before classification.")
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train SVM
    svm_model = train_svm(X_train, y_train)
    svm_acc   = accuracy_score(y_test, svm_model.predict(X_test))
    print(f"\nSVM Test Accuracy : {svm_acc*100:.2f}%")

    # ── Train k-NN
    knn_model, best_k = train_knn(X_train, y_train)
    knn_acc   = accuracy_score(y_test, knn_model.predict(X_test))
    print(f"\nk-NN (k={best_k}) Test Accuracy : {knn_acc*100:.2f}%")

    # ── Save models + scaler
    joblib.dump(svm_model,  os.path.join(MODELS_DIR, "svm_model.pkl"))
    joblib.dump(knn_model,  os.path.join(MODELS_DIR, "knn_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODELS_DIR, "feature_scaler.pkl"))
    print(f"\nSaved: {MODELS_DIR}/svm_model.pkl")
    print(f"Saved: {MODELS_DIR}/knn_model.pkl")
    print(f"Saved: {MODELS_DIR}/feature_scaler.pkl")

    # Save best k for reference
    with open(os.path.join(MODELS_DIR, "knn_best_k.txt"), "w") as f:
        f.write(str(best_k))

    print("\nSTEP 5 complete.\n")


if __name__ == "__main__":
    main()
