"""
Model 8: Vision Transformer (ViT)
Deep Learning Implementation:
- Pretrained ViT-Base model (16x16 patches, 224x224 input)
- Fine-tuned for neonatal jaundice detection
- Transfer learning from ImageNet
- Clinical features (age + gender) fused with image features
- Both regression (bilirubin prediction) and classification (jaundice detection)
- 10-Fold Cross-Validation
- Metrics: Accuracy, Sensitivity, Specificity, F1, AUC, Correlation (R), RMSE
"""

import pandas as pd
import numpy as np
import pickle
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
from torchvision import transforms
from timm import create_model
import warnings
warnings.filterwarnings('ignore')

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*70)
print("MODEL 8: VISION TRANSFORMER (ViT-Base)")
print("="*70)
print(f"Device: {device}")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'MODEL_NAME': 'vit_base_patch16_224',  # Pretrained ViT-Base
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'EPOCHS_REG': 30,
    'EPOCHS_CLF': 30,
    'LR_REG': 1e-4,
    'LR_CLF': 1e-4,
    'WEIGHT_DECAY': 1e-4,
    'N_FOLDS': 10,
    'RANDOM_SEED': 42,
    'IMAGES_PATH': 'NeoJaundice/NeoJaundice/images/',
    'CLINICAL_DIM': 2,   # gender + age
    'VIT_FEAT_DIM': 768, # ViT-Base output dimension
}

torch.manual_seed(CONFIG['RANDOM_SEED'])
np.random.seed(CONFIG['RANDOM_SEED'])

# ============================================================================
# 2. DATASET CLASS
# ============================================================================

class JaundiceDataset(Dataset):
    """Dataset: returns (image, clinical_features, target)"""

    def __init__(self, df, images_path, clinical_features, transform=None, target_type='regression'):
        self.df = df.reset_index(drop=True)
        self.images_path = images_path
        self.clinical_features = clinical_features  # (N, 2) numpy array
        self.transform = transform
        self.target_type = target_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_path, row['image_idx'])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))

        if self.transform:
            image = self.transform(image)

        clinical = torch.tensor(self.clinical_features[idx], dtype=torch.float32)

        bilirubin = row['blood(mg/dL)']
        if self.target_type == 'regression':
            target = torch.tensor(bilirubin, dtype=torch.float32)
        else:
            target = torch.tensor(1 if bilirubin >= 15 else 0, dtype=torch.long)

        return image, clinical, target

# ============================================================================
# 3. DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 4. MODEL DEFINITION
# ============================================================================

class ViTRegressor(nn.Module):
    """ViT backbone + clinical feature fusion for bilirubin regression."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = create_model(CONFIG['MODEL_NAME'], pretrained=pretrained, num_classes=0)
        fused_dim = CONFIG['VIT_FEAT_DIM'] + CONFIG['CLINICAL_DIM']
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, clinical):
        img_features = self.vit(x)
        combined = torch.cat([img_features, clinical], dim=1)
        return self.regressor(combined).squeeze(-1)


class ViTClassifier(nn.Module):
    """ViT backbone + clinical feature fusion for jaundice classification."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = create_model(CONFIG['MODEL_NAME'], pretrained=pretrained, num_classes=0)
        fused_dim = CONFIG['VIT_FEAT_DIM'] + CONFIG['CLINICAL_DIM']
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x, clinical):
        img_features = self.vit(x)
        combined = torch.cat([img_features, clinical], dim=1)
        return self.classifier(combined)

# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================

def train_epoch_regression(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, clinical, targets in dataloader:
        images, clinical, targets = images.to(device), clinical.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images, clinical)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_regression(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for images, clinical, targets in dataloader:
            images, clinical = images.to(device), clinical.to(device)
            outputs = model(images, clinical)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    r = np.corrcoef(actuals, predictions)[0, 1]
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    return r, rmse, mae, predictions


def train_epoch_classification(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, clinical, targets in dataloader:
        images, clinical, targets = images.to(device), clinical.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images, clinical)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_classification(model, dataloader):
    model.eval()
    predictions, probabilities, actuals = [], [], []
    with torch.no_grad():
        for images, clinical, targets in dataloader:
            images, clinical = images.to(device), clinical.to(device)
            outputs = model(images, clinical)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            actuals.extend(targets.numpy())
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    actuals = np.array(actuals)
    acc = accuracy_score(actuals, predictions)
    sens = recall_score(actuals, predictions, zero_division=0)
    cm = confusion_matrix(actuals, predictions)
    spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    f1 = f1_score(actuals, predictions, zero_division=0)
    auc = roc_auc_score(actuals, probabilities) if len(np.unique(actuals)) > 1 else 0
    return acc, sens, spec, f1, auc, predictions, probabilities

# ============================================================================
# 6. LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

with open('preprocessed_data/data_split.pkl', 'rb') as f:
    split_data = pickle.load(f)

train_df = split_data['train_df']
val_df   = split_data['val_df']
test_df  = split_data['test_df']

# Load clinical features (age + gender, already scaled)
clinical_data = np.load('preprocessed_data/clinical_features.npz')
train_clinical = clinical_data['train_clinical']
val_clinical   = clinical_data['val_clinical']
test_clinical  = clinical_data['test_clinical']

# Combine train+val for 10-fold CV
full_df       = pd.concat([train_df, val_df], ignore_index=True)
full_clinical = np.vstack([train_clinical, val_clinical])

print(f"\nDataset: {len(full_df)} samples (train+val), {full_clinical.shape[1]} clinical features")
print(f"Test: {len(test_df)} samples")
print(f"\nClass Distribution (Train+Val):")
print(f"  Class 0 (No Jaundice): {len(full_df[full_df['blood(mg/dL)'] < 15])} ({len(full_df[full_df['blood(mg/dL)'] < 15])/len(full_df)*100:.1f}%)")
print(f"  Class 1 (Jaundice): {len(full_df[full_df['blood(mg/dL)'] >= 15])} ({len(full_df[full_df['blood(mg/dL)'] >= 15])/len(full_df)*100:.1f}%)")

# ============================================================================
# 7. REGRESSION - 10-FOLD CROSS-VALIDATION
# ============================================================================

print("\n" + "="*70)
print("VISION TRANSFORMER REGRESSION - BILIRUBIN PREDICTION")
print("="*70)

kfold = KFold(n_splits=CONFIG['N_FOLDS'], shuffle=True, random_state=CONFIG['RANDOM_SEED'])

cv_predictions_reg = np.zeros(len(full_df))
cv_r_scores, cv_rmse_scores = [], []

for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df), 1):
    print(f"\n--- Fold {fold}/{CONFIG['N_FOLDS']} ---")

    train_fold_df = full_df.iloc[train_idx]
    val_fold_df   = full_df.iloc[val_idx]

    train_dataset = JaundiceDataset(train_fold_df, CONFIG['IMAGES_PATH'],
                                    full_clinical[train_idx], train_transform, 'regression')
    val_dataset   = JaundiceDataset(val_fold_df,   CONFIG['IMAGES_PATH'],
                                    full_clinical[val_idx],   val_transform,   'regression')

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model     = ViTRegressor(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR_REG'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_REG'])

    best_val_r, patience, patience_counter = -np.inf, 5, 0

    for epoch in range(CONFIG['EPOCHS_REG']):
        train_loss = train_epoch_regression(model, train_loader, criterion, optimizer)
        val_r, val_rmse, _, _ = evaluate_regression(model, val_loader)
        scheduler.step()

        if val_r > best_val_r:
            best_val_r = val_r
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == CONFIG['EPOCHS_REG'] - 1:
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS_REG']}: Loss={train_loss:.4f}, Val R={val_r:.4f}, Val RMSE={val_rmse:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    _, val_rmse, _, fold_preds = evaluate_regression(model, val_loader)
    cv_predictions_reg[val_idx] = fold_preds
    cv_r_scores.append(best_val_r)
    cv_rmse_scores.append(val_rmse)
    print(f"Fold {fold} - R: {best_val_r:.4f}, RMSE: {val_rmse:.4f}")

cv_r    = np.corrcoef(full_df['blood(mg/dL)'].values, cv_predictions_reg)[0, 1]
cv_rmse = np.sqrt(mean_squared_error(full_df['blood(mg/dL)'].values, cv_predictions_reg))
cv_mae  = mean_absolute_error(full_df['blood(mg/dL)'].values, cv_predictions_reg)

print(f"\n10-Fold Cross-Validation Results:")
print(f"  Correlation (R): {cv_r:.4f} (±{np.std(cv_r_scores):.4f})")
print(f"  RMSE: {cv_rmse:.4f} (±{np.std(cv_rmse_scores):.4f})")
print(f"  MAE: {cv_mae:.4f}")

# Final model trained on all train+val
print("\nTraining final model on full train+val data...")
full_dataset = JaundiceDataset(full_df, CONFIG['IMAGES_PATH'], full_clinical, train_transform, 'regression')
full_loader = DataLoader(
    full_dataset,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)



test_dataset = JaundiceDataset(test_df, CONFIG['IMAGES_PATH'], test_clinical, val_transform, 'regression')
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

final_model_reg = ViTRegressor(pretrained=True).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(final_model_reg.parameters(), lr=CONFIG['LR_REG'], weight_decay=CONFIG['WEIGHT_DECAY'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_REG'])

for epoch in range(CONFIG['EPOCHS_REG']):
    train_loss = train_epoch_regression(final_model_reg, full_loader, criterion, optimizer)
    scheduler.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS_REG']}: Loss={train_loss:.4f}")

test_r, test_rmse, test_mae, test_preds_reg = evaluate_regression(final_model_reg, test_loader)

print(f"\nTest Set Results:")
print(f"  Correlation (R): {test_r:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE: {test_mae:.4f}")

os.makedirs('models', exist_ok=True)
torch.save(final_model_reg.state_dict(), 'models/vit_regressor.pth')
print("\n✅ Regression model saved to models/vit_regressor.pth")

# ============================================================================
# 8. CLASSIFICATION - 10-FOLD CROSS-VALIDATION
# ============================================================================

print("\n" + "="*70)
print("VISION TRANSFORMER CLASSIFICATION - JAUNDICE DETECTION")
print("="*70)

cv_predictions_clf  = np.zeros(len(full_df))
cv_probabilities_clf = np.zeros(len(full_df))
cv_acc_scores, cv_auc_scores = [], []

for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df), 1):
    print(f"\n--- Fold {fold}/{CONFIG['N_FOLDS']} ---")

    train_fold_df = full_df.iloc[train_idx]
    val_fold_df   = full_df.iloc[val_idx]

    train_dataset = JaundiceDataset(train_fold_df, CONFIG['IMAGES_PATH'],
                                    full_clinical[train_idx], train_transform, 'classification')
    val_dataset   = JaundiceDataset(val_fold_df,   CONFIG['IMAGES_PATH'],
                                    full_clinical[val_idx],   val_transform,   'classification')

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = ViTClassifier(pretrained=True).to(device)

    y_train      = (train_fold_df['blood(mg/dL)'] >= 15).astype(int).values
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([len(y_train)/class_counts[0], len(y_train)/class_counts[1]]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR_CLF'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_CLF'])

    best_val_auc, patience, patience_counter = 0, 5, 0

    for epoch in range(CONFIG['EPOCHS_CLF']):
        train_loss = train_epoch_classification(model, train_loader, criterion, optimizer)
        val_acc, _, _, _, val_auc, _, _ = evaluate_classification(model, val_loader)
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == CONFIG['EPOCHS_CLF'] - 1:
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS_CLF']}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    val_acc, val_sens, val_spec, val_f1, val_auc, fold_preds, fold_probs = evaluate_classification(model, val_loader)
    cv_predictions_clf[val_idx]  = fold_preds
    cv_probabilities_clf[val_idx] = fold_probs
    cv_acc_scores.append(val_acc)
    cv_auc_scores.append(val_auc)
    print(f"Fold {fold} - Acc: {val_acc:.4f}, Sens: {val_sens:.4f}, Spec: {val_spec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

y_full_clf = (full_df['blood(mg/dL)'] >= 15).astype(int).values
cv_acc  = accuracy_score(y_full_clf, cv_predictions_clf)
cv_sens = recall_score(y_full_clf, cv_predictions_clf, zero_division=0)
cm_cv   = confusion_matrix(y_full_clf, cv_predictions_clf)
cv_spec = cm_cv[0,0] / (cm_cv[0,0] + cm_cv[0,1]) if (cm_cv[0,0] + cm_cv[0,1]) > 0 else 0
cv_f1   = f1_score(y_full_clf, cv_predictions_clf, zero_division=0)
cv_auc  = roc_auc_score(y_full_clf, cv_probabilities_clf)

print(f"\n10-Fold Cross-Validation Results:")
print(f"  Accuracy: {cv_acc:.4f} (±{np.std(cv_acc_scores):.4f})")
print(f"  Sensitivity: {cv_sens:.4f}")
print(f"  Specificity: {cv_spec:.4f}")
print(f"  F1-Score: {cv_f1:.4f}")
print(f"  AUC: {cv_auc:.4f} (±{np.std(cv_auc_scores):.4f})")

# Final model
print("\nTraining final model on full train+val data...")
full_dataset = JaundiceDataset(full_df, CONFIG['IMAGES_PATH'], full_clinical, train_transform, 'classification')
full_loader = DataLoader(
    full_dataset,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)



test_dataset = JaundiceDataset(test_df, CONFIG['IMAGES_PATH'], test_clinical, val_transform, 'classification')
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['BATCH_SIZE'],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

final_model_clf = ViTClassifier(pretrained=True).to(device)

y_full        = (full_df['blood(mg/dL)'] >= 15).astype(int).values
class_counts  = np.bincount(y_full)
class_weights = torch.FloatTensor([len(y_full)/class_counts[0], len(y_full)/class_counts[1]]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(final_model_clf.parameters(), lr=CONFIG['LR_CLF'], weight_decay=CONFIG['WEIGHT_DECAY'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_CLF'])

for epoch in range(CONFIG['EPOCHS_CLF']):
    train_loss = train_epoch_classification(final_model_clf, full_loader, criterion, optimizer)
    scheduler.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS_CLF']}: Loss={train_loss:.4f}")

y_test_clf = (test_df['blood(mg/dL)'] >= 15).astype(int).values
test_acc, test_sens, test_spec, test_f1, test_auc, test_preds_clf, test_probs_clf = evaluate_classification(
    final_model_clf, test_loader)

print(f"\nTest Set Results:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Sensitivity: {test_sens:.4f}")
print(f"  Specificity: {test_spec:.4f}")
print(f"  F1-Score: {test_f1:.4f}")
print(f"  AUC: {test_auc:.4f}")

torch.save(final_model_clf.state_dict(), 'models/vit_classifier.pth')
print("\n✅ Classification model saved to models/vit_classifier.pth")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

os.makedirs('results', exist_ok=True)

results = pd.DataFrame({
    'Model': ['ViT_Regression', 'ViT_Classification'],
    'CV_R/Acc': [cv_r, cv_acc],
    'CV_Sens':  ['-', cv_sens],
    'CV_Spec':  ['-', cv_spec],
    'CV_F1':    ['-', cv_f1],
    'CV_AUC':   ['-', cv_auc],
    'CV_RMSE':  [cv_rmse, '-'],
    'Test_R/Acc': [test_r, test_acc],
    'Test_Sens':  ['-', test_sens],
    'Test_Spec':  ['-', test_spec],
    'Test_F1':    ['-', test_f1],
    'Test_AUC':   ['-', test_auc],
    'Test_RMSE':  [test_rmse, '-']
})

results.to_csv('results/ViT_results.csv', index=False)
print("✅ Results saved to results/ViT_results.csv")

print("\n" + "="*70)
print("VISION TRANSFORMER TRAINING COMPLETE!")
print("="*70)
print("\nSummary:")
print(f"  Regression    - CV R: {cv_r:.4f}, Test R: {test_r:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"  Classification - CV Acc: {cv_acc:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
