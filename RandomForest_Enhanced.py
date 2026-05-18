"""
Model 5: Random Forest - ENHANCED VERSION
Improvements:
✅ Automatic threshold optimization using ROC curve (Youden Index)
✅ StratifiedKFold for classification cross-validation
✅ Hyperparameter tuning with RandomizedSearchCV
✅ class_weight='balanced' for handling class imbalance
✅ Comprehensive reporting with optimal threshold
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, mean_squared_error
)
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL 5: RANDOM FOREST - ENHANCED VERSION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

with open('preprocessed_data/data_split.pkl', 'rb') as f:
    split_data = pickle.load(f)

color_data = np.load('preprocessed_data/color_features.npz')

train_features = color_data['train_features']
val_features = color_data['val_features']
test_features = color_data['test_features']

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
print(f"\nClass Distribution (Train+Val):")
print(f"  Class 0 (No Jaundice): {np.sum(y_full_clf == 0)} ({np.sum(y_full_clf == 0)/len(y_full_clf)*100:.1f}%)")
print(f"  Class 1 (Jaundice): {np.sum(y_full_clf == 1)} ({np.sum(y_full_clf == 1)/len(y_full_clf)*100:.1f}%)")

# ============================================================================
# REGRESSION - Keep unchanged as requested
# ============================================================================

print("\n" + "="*80)
print("RANDOM FOREST REGRESSION - BILIRUBIN PREDICTION")
print("="*80)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

cv_pred = cross_val_predict(rf_reg, X_full, y_full_reg, cv=10, n_jobs=-1)
cv_r = np.corrcoef(y_full_reg, cv_pred)[0, 1]
cv_rmse = np.sqrt(mean_squared_error(y_full_reg, cv_pred))

print(f"\n10-Fold CV: R={cv_r:.4f}, RMSE={cv_rmse:.4f}")

rf_reg.fit(X_full, y_full_reg)
y_test_pred = rf_reg.predict(X_test)

test_r = np.corrcoef(y_test_reg, y_test_pred)[0, 1]
test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))

print(f"Test Set: R={test_r:.4f}, RMSE={test_rmse:.4f}")

# ============================================================================
# CLASSIFICATION - ENHANCED WITH ALL IMPROVEMENTS
# ============================================================================

print("\n" + "="*80)
print("RANDOM FOREST CLASSIFICATION - JAUNDICE DETECTION (ENHANCED)")
print("="*80)

# 1️⃣ HYPERPARAMETER TUNING
print("\n[1/4] Hyperparameter Tuning with RandomizedSearchCV...")
print("-" * 80)

param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use StratifiedKFold for hyperparameter search
skf_search = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    base_rf,
    param_distributions=param_distributions,
    n_iter=20,
    cv=skf_search,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

random_search.fit(X_full, y_full_clf)

best_params = random_search.best_params_
print(f"Best parameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"Best CV AUC: {random_search.best_score_:.4f}")

# Use best parameters for final model
rf_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

# 2️⃣ STRATIFIED 10-FOLD CROSS-VALIDATION
print("\n[2/4] Performing Stratified 10-Fold Cross-Validation...")
print("-" * 80)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Get CV predictions with probabilities
cv_pred_proba = cross_val_predict(
    rf_clf, X_full, y_full_clf, cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]

# 3️⃣ OPTIMIZE THRESHOLD USING ROC CURVE (YOUDEN INDEX)
print("\n[3/4] Optimizing Classification Threshold...")
print("-" * 80)

fpr, tpr, thresholds = roc_curve(y_full_clf, cv_pred_proba)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold_cv = thresholds[optimal_idx]

print(f"Optimal threshold (Youden Index): {optimal_threshold_cv:.4f}")
print(f"  At this threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")

# Apply optimal threshold to CV predictions
cv_pred_clf_optimized = (cv_pred_proba >= optimal_threshold_cv).astype(int)

# Calculate CV metrics with optimized threshold
cv_acc = accuracy_score(y_full_clf, cv_pred_clf_optimized)
cv_sens = recall_score(y_full_clf, cv_pred_clf_optimized, zero_division=0)
cm_cv = confusion_matrix(y_full_clf, cv_pred_clf_optimized)
cv_spec = cm_cv[0,0] / (cm_cv[0,0] + cm_cv[0,1]) if (cm_cv[0,0] + cm_cv[0,1]) > 0 else 0
cv_f1 = f1_score(y_full_clf, cv_pred_clf_optimized, zero_division=0)
cv_auc = roc_auc_score(y_full_clf, cv_pred_proba)

print(f"\n10-Fold Stratified CV Results (Optimized Threshold):")
print(f"  Accuracy:    {cv_acc:.4f}")
print(f"  Sensitivity: {cv_sens:.4f}")
print(f"  Specificity: {cv_spec:.4f}")
print(f"  F1-Score:    {cv_f1:.4f}")
print(f"  AUC:         {cv_auc:.4f}")
print(f"\nCV Confusion Matrix:")
print(cm_cv)

# 4️⃣ TEST SET EVALUATION
print("\n[4/4] Testing on Hold-out Test Set...")
print("-" * 80)

rf_clf.fit(X_full, y_full_clf)
y_test_proba = rf_clf.predict_proba(X_test)[:, 1]

# Apply optimal threshold from CV to test set
y_test_pred_clf = (y_test_proba >= optimal_threshold_cv).astype(int)

test_acc = accuracy_score(y_test_clf, y_test_pred_clf)
test_sens = recall_score(y_test_clf, y_test_pred_clf, zero_division=0)
cm_test = confusion_matrix(y_test_clf, y_test_pred_clf)
test_spec = cm_test[0,0] / (cm_test[0,0] + cm_test[0,1]) if (cm_test[0,0] + cm_test[0,1]) > 0 else 0
test_f1 = f1_score(y_test_clf, y_test_pred_clf, zero_division=0)
test_auc = roc_auc_score(y_test_clf, y_test_proba)

print(f"Test Set Results (Optimized Threshold = {optimal_threshold_cv:.4f}):")
print(f"  Accuracy:    {test_acc:.4f}")
print(f"  Sensitivity: {test_sens:.4f}")
print(f"  Specificity: {test_spec:.4f}")
print(f"  F1-Score:    {test_f1:.4f}")
print(f"  AUC:         {test_auc:.4f}")
print(f"\nTest Confusion Matrix:")
print(cm_test)
print(f"  TN={cm_test[0,0]}, FP={cm_test[0,1]}")
print(f"  FN={cm_test[1,0]}, TP={cm_test[1,1]}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Save models
with open('models/rf_regressor_enhanced.pkl', 'wb') as f:
    pickle.dump(rf_reg, f)
with open('models/rf_classifier_enhanced.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

# Save results
results = pd.DataFrame({
    'Model': ['RandomForest_Regression', 'RandomForest_Classification_Enhanced'],
    'CV_R/Acc': [cv_r, cv_acc],
    'CV_RMSE': [cv_rmse, '-'],
    'CV_Sens': ['-', cv_sens],
    'CV_Spec': ['-', cv_spec],
    'CV_F1': ['-', cv_f1],
    'CV_AUC': ['-', cv_auc],
    'Test_R/Acc': [test_r, test_acc],
    'Test_RMSE': [test_rmse, '-'],
    'Test_Sens': ['-', test_sens],
    'Test_Spec': ['-', test_spec],
    'Test_F1': ['-', test_f1],
    'Test_AUC': ['-', test_auc],
    'Optimal_Threshold': ['-', f'{optimal_threshold_cv:.4f}']
})
results.to_csv('results/RandomForest_Enhanced_results.csv', index=False)

# Save hyperparameters
hyperparams_df = pd.DataFrame([best_params])
hyperparams_df.to_csv('results/RandomForest_Enhanced_hyperparameters.csv', index=False)

print("✅ Models saved to models/")
print("✅ Results saved to results/RandomForest_Enhanced_results.csv")
print("✅ Hyperparameters saved to results/RandomForest_Enhanced_hyperparameters.csv")

print("\n" + "="*80)
print("RANDOM FOREST ENHANCED TRAINING COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f"  Regression:")
print(f"    CV R: {cv_r:.4f}, Test R: {test_r:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"  Classification (Enhanced):")
print(f"    CV Acc: {cv_acc:.4f}, Test Acc: {test_acc:.4f}")
print(f"    Test Sens: {test_sens:.4f}, Test Spec: {test_spec:.4f}")
print(f"    Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
print(f"    Optimal Threshold: {optimal_threshold_cv:.4f}")
print("="*80)
