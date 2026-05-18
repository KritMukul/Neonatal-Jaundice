import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========================
# CONFIG
# ========================
CSV_PATH = "/workspace/RohitSir /NeoJaundice/chd_jaundice_published_2.csv"
IMG_DIR = "/workspace/RohitSir /NeoJaundice/images"

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# LOAD CSV
# ========================
df = pd.read_csv(CSV_PATH)

df = df.rename(columns={
    "image_idx": "image_name",
    "blood(mg/dL)": "bilirubin"
})

# Encode gender
df["gender"] = df["gender"].map({"M": 1, "F": 0})

# ========================
# DATASET
# ========================
class JaundiceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(IMG_DIR, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["bilirubin"], dtype=torch.float32)

        return image, label

# ========================
# TRANSFORMS
# ========================
train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ========================
# GROUP SPLIT (CRITICAL)
# ========================
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

print("Train:", len(train_df), "Val:", len(val_df))

# ========================
# DATA LOADERS
# ========================
train_ds = JaundiceDataset(train_df, train_tf)
val_ds = JaundiceDataset(val_df, val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ========================
# MODEL
# ========================
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(DEVICE)

# ========================
# LOSS & OPTIMIZER
# ========================
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========================
# TRAIN LOOP
# ========================
best_mae = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        preds = model(imgs).squeeze()
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ========================
    # VALIDATION
    # ========================
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)

            preds = model(imgs).squeeze().cpu().numpy()

            preds_all.extend(preds)
            labels_all.extend(labels.numpy())

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    # ========================
    # METRICS
    # ========================
    mae = mean_absolute_error(labels_all, preds_all)
    rmse = np.sqrt(mean_squared_error(labels_all, preds_all))
    r2 = r2_score(labels_all, preds_all)

    mape = np.mean(np.abs((labels_all - preds_all) / labels_all)) * 100
    errors = np.abs(labels_all - preds_all)

    within_2 = np.mean(errors <= 2) * 100
    within_3 = np.mean(errors <= 3) * 100

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")

    print("\n--- Regression Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Median Error: {np.median(errors):.4f}")
    print(f"Max Error: {errors.max():.4f}")
    print(f"% within ±2 mg/dL: {within_2:.2f}%")
    print(f"% within ±3 mg/dL: {within_3:.2f}%")

    # SAVE BEST MODEL
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")

print("\n🔥 FINAL BEST MAE:", best_mae)