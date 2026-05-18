"""
Model 5: Random Forest
Enhanced Version with Feature Importance Analysis
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error
)
import os

print("="*70)
print("MODEL 5: RANDOM FOREST (100 trees)")
print("="*70)

# ============================================================
# LOAD DATA
# ============================================================

with open('preprocessed_data/data_split.pkl', 'rb') as f:
    split_data = pickle.load(f)

color_data = np.load('preprocessed_data/color_features.npz')
# Load clinical features (age + gender)
clinical_data = np.load('preprocessed_data/clinical_features.npz')

# Combine color + clinical features
train_features = np.hstack([color_data['train_features'], clinical_data['train_clinical']])
val_features = np.hstack([color_data['val_features'], clinical_data['val_clinical']])
test_features = np.hstack([color_data['test_features'], clinical_data['test_clinical']])

X_full = np.vstack([train_features, val_features])
full_df = pd.concat([split_data['train_df'], split_data['val_df']], ignore_index=True)
test_df = split_data['test_df']

y_full_reg = full_df['blood(mg/dL)'].values
y_full_clf = (full_df['blood(mg/dL)'] >= 15).astype(int).values
y_test_reg = test_df['blood(mg/dL)'].values
y_test_clf = (test_df['blood(mg/dL)'] >= 15).astype(int).values

X_test = test_features

print(f"\nDataset: {X_full.shape[0]} samples with {X_full.shape[1]} features")
print(f"Test: {X_test.shape[0]} samples")

# ============================================================
# REGRESSION
# ============================================================

print("\n" + "="*70)
print("RANDOM FOREST REGRESSION")
print("="*70)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

cv_pred = cross_val_predict(rf_reg, X_full, y_full_reg, cv=10)
cv_r = np.corrcoef(y_full_reg, cv_pred)[0, 1]
cv_rmse = np.sqrt(mean_squared_error(y_full_reg, cv_pred))

print(f"\n10-Fold CV: R={cv_r:.4f}, RMSE={cv_rmse:.4f}")

rf_reg.fit(X_full, y_full_reg)
y_test_pred = rf_reg.predict(X_test)

test_r = np.corrcoef(y_test_reg, y_test_pred)[0, 1]
test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))

print(f"Test Set: R={test_r:.4f}, RMSE={test_rmse:.4f}")

# ============================================================
# CLASSIFICATION
# ============================================================

print("\n" + "="*70)
print("RANDOM FOREST CLASSIFICATION")
print("="*70)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

cv_pred_clf = cross_val_predict(rf_clf, X_full, y_full_clf, cv=10)
cv_pred_proba = cross_val_predict(
    rf_clf, X_full, y_full_clf, cv=10, method='predict_proba'
)[:, 1]

cv_acc = accuracy_score(y_full_clf, cv_pred_clf)
cv_sens = recall_score(y_full_clf, cv_pred_clf)
cm = confusion_matrix(y_full_clf, cv_pred_clf)
cv_spec = cm[0,0] / (cm[0,0] + cm[0,1])
cv_f1 = f1_score(y_full_clf, cv_pred_clf)
cv_auc = roc_auc_score(y_full_clf, cv_pred_proba)

print(f"\n10-Fold CV:")
print(f"Accuracy={cv_acc:.4f}, Sensitivity={cv_sens:.4f}, Specificity={cv_spec:.4f}, F1={cv_f1:.4f}, AUC={cv_auc:.4f}")

rf_clf.fit(X_full, y_full_clf)

y_test_pred_clf = rf_clf.predict(X_test)
y_test_proba = rf_clf.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test_clf, y_test_pred_clf)
test_sens = recall_score(y_test_clf, y_test_pred_clf)
cm_test = confusion_matrix(y_test_clf, y_test_pred_clf)
test_spec = cm_test[0,0] / (cm_test[0,0] + cm_test[0,1])
test_f1 = f1_score(y_test_clf, y_test_pred_clf)
test_auc = roc_auc_score(y_test_clf, y_test_proba)

print(f"\nTest Set:")
print(f"Accuracy={test_acc:.4f}, Sensitivity={test_sens:.4f}, Specificity={test_spec:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
print(f"\nConfusion Matrix:\n{cm_test}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_names = [
    "B_mean","G_mean","R_mean",
    "H_mean","S_mean","V_mean",
    "L_mean","A_mean","LAB_b_mean",
    "Y_mean","Cr_mean","Cb_mean",
    "B_std","G_std","R_std",
    "H_std","S_std","V_std",
    "L_std","A_std","LAB_b_std",
    "Yellow_strength","R/B_ratio","G/B_ratio","R/G_ratio",
    "LAB_b_yellow","LAB_b/L_ratio","Cr_minus_Cb","SxV_product"
]

importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 10 Important Features:")
for i in range(10):
    idx = indices[i]
    print(f"{i+1}. {feature_names[idx]} → {importances[idx]:.4f}")

plt.figure(figsize=(10,6))
plt.title("Top 10 Feature Importances")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("RANDOM FOREST COMPLETED!")
print("="*70)