"""
Model 3: ElasticNet Regression
Research Paper Implementation:
- ElasticNet with alpha=1.0, l1_ratio=0.5
- 10-Fold Cross-Validation
- Regression: Predict bilirubin (TsB)
- Classification: Logistic Regression with ElasticNet penalty
- Metrics: Accuracy, Sensitivity, Specificity, F1, AUC, Correlation (R), RMSE
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
import os

print("="*70)
print("MODEL 3: ELASTIC NET (alpha=1.0, l1_ratio=0.5)")
print("="*70)

# Load data
with open('preprocessed_data/data_split.pkl', 'rb') as f:
    split_data = pickle.load(f)

train_df = split_data['train_df']
val_df = split_data['val_df']
test_df = split_data['test_df']

# Load color features
color_data = np.load('preprocessed_data/color_features.npz')
# Load clinical features (age + gender)
clinical_data = np.load('preprocessed_data/clinical_features.npz')

# Combine color + clinical features
train_features = np.hstack([color_data['train_features'], clinical_data['train_clinical']])
val_features = np.hstack([color_data['val_features'], clinical_data['val_clinical']])
test_features = np.hstack([color_data['test_features'], clinical_data['test_clinical']])

# Combine train+val
X_full = np.vstack([train_features, val_features])
full_df = pd.concat([train_df, val_df], ignore_index=True)

# Get targets
y_full_reg = full_df['blood(mg/dL)'].values
y_full_clf = (full_df['blood(mg/dL)'] >= 15).astype(int).values
y_test_reg = test_df['blood(mg/dL)'].values
y_test_clf = (test_df['blood(mg/dL)'] >= 15).astype(int).values

X_test = test_features

print(f"\nDataset: {X_full.shape[0]} samples with {X_full.shape[1]} color features")
print(f"Test: {test_features.shape[0]} samples")
print(f"\nClass Distribution (Train+Val):")
print(f"  Class 0 (No Jaundice): {np.sum(y_full_clf == 0)} ({np.sum(y_full_clf == 0)/len(y_full_clf)*100:.1f}%)")
print(f"  Class 1 (Jaundice): {np.sum(y_full_clf == 1)} ({np.sum(y_full_clf == 1)/len(y_full_clf)*100:.1f}%)")

# REGRESSION
print("\n" + "="*70)
print("ELASTICNET REGRESSION")
print("="*70)

en_reg = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

cv_pred = cross_val_predict(en_reg, X_full, y_full_reg, cv=10)
cv_r = np.corrcoef(y_full_reg, cv_pred)[0, 1]
cv_rmse = np.sqrt(mean_squared_error(y_full_reg, cv_pred))
cv_mae = mean_absolute_error(y_full_reg, cv_pred)

print(f"\n10-Fold CV: R={cv_r:.4f}, RMSE={cv_rmse:.4f}, MAE={cv_mae:.4f}")

en_reg.fit(X_full, y_full_reg)
y_test_pred = en_reg.predict(X_test)
test_r = np.corrcoef(y_test_reg, y_test_pred)[0, 1]
test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))
test_mae = mean_absolute_error(y_test_reg, y_test_pred)

print(f"Test Set: R={test_r:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

# CLASSIFICATION
print("\n" + "="*70)
print("LOGISTIC REGRESSION WITH ELASTIC NET PENALTY")
print("="*70)
print("Note: Using class_weight='balanced' for handling class imbalance")

en_clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, 
                             C=1.0, max_iter=5000, random_state=42, class_weight='balanced')

cv_pred_clf = cross_val_predict(en_clf, X_full, y_full_clf, cv=10)
en_clf.fit(X_full, y_full_clf)
cv_pred_proba = cross_val_predict(en_clf, X_full, y_full_clf, cv=10, method='predict_proba')[:, 1]

cv_acc = accuracy_score(y_full_clf, cv_pred_clf)
cv_sens = recall_score(y_full_clf, cv_pred_clf, zero_division=0)
cm = confusion_matrix(y_full_clf, cv_pred_clf)
cv_spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
cv_f1 = f1_score(y_full_clf, cv_pred_clf, zero_division=0)
cv_auc = roc_auc_score(y_full_clf, cv_pred_proba)

print(f"\n10-Fold CV: Acc={cv_acc:.4f}, Sens={cv_sens:.4f}, Spec={cv_spec:.4f}, F1={cv_f1:.4f}, AUC={cv_auc:.4f}")

y_test_pred_clf = en_clf.predict(X_test)
y_test_proba = en_clf.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test_clf, y_test_pred_clf)
test_sens = recall_score(y_test_clf, y_test_pred_clf, zero_division=0)
cm_test = confusion_matrix(y_test_clf, y_test_pred_clf)
test_spec = cm_test[0,0] / (cm_test[0,0] + cm_test[0,1]) if (cm_test[0,0] + cm_test[0,1]) > 0 else 0
test_f1 = f1_score(y_test_clf, y_test_pred_clf, zero_division=0)
test_auc = roc_auc_score(y_test_clf, y_test_proba)

print(f"Test Set: Acc={test_acc:.4f}, Sens={test_sens:.4f}, Spec={test_spec:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
print(f"\nConfusion Matrix:\n{cm_test}")

# Save
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
with open('models/elasticnet_regressor.pkl', 'wb') as f:
    pickle.dump(en_reg, f)
with open('models/elasticnet_classifier.pkl', 'wb') as f:
    pickle.dump(en_clf, f)

results = pd.DataFrame({
    'Model': ['ElasticNet_Regression', 'ElasticNet_Classification'],
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
    'Test_AUC': ['-', test_auc]
})
results.to_csv('results/ElasticNet_results.csv', index=False)

print("\n" + "="*70)
print("ELASTICNET COMPLETED!")
print(f"Regression - Test R: {test_r:.4f}, RMSE: {test_rmse:.4f}")
print(f"Classification - Test Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
print("="*70)
