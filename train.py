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
from sklearn.metrics import r2_score, mean_absolute_error

# ========================
# CONFIG
# ========================
CSV_PATH = "/workspace/RohitSir/NeoJaundice/chd_jaundice_published_2.csv"
IMG_DIR = "/workspace/RohitSir/NeoJaundice/images"

BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
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
# TARGET NORMALIZATION
# ========================
mean_bil = df["bilirubin"].mean()
std_bil = df["bilirubin"].std()

df["bilirubin"] = (df["bilirubin"] - mean_bil) / std_bil

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

        # SAFE FEATURES ONLY
        features = torch.tensor([
            row["gender"],
            row["gestational_age"],
            row["age(day)"],
            row["weight"]
        ], dtype=torch.float32)

        label = torch.tensor(row["bilirubin"], dtype=torch.float32)

        return image, features, label, row["patient_id"]

# ========================
# TRANSFORMS
# ========================
train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(400),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.1,0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ========================
# SPLIT (PATIENT-WISE)
# ========================
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# ========================
# LOADERS
# ========================
train_ds = JaundiceDataset(train_df, train_tf)
val_ds = JaundiceDataset(val_df, val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# MODEL
# ========================
class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.efficientnet_b0(pretrained=True)
        self.cnn.classifier = nn.Identity()

        self.tabular = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.fc = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img, tab):
        img_feat = self.cnn(img)
        tab_feat = self.tabular(tab)

        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.fc(x)

model = MultiModalModel().to(DEVICE)

# ========================
# LOSS
# ========================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ========================
# TRAIN
# ========================
best_r2 = -1

for epoch in range(EPOCHS):
    model.train()

    for imgs, feats, labels, _ in tqdm(train_loader):
        imgs = imgs.to(DEVICE)
        feats = feats.to(DEVICE)
        labels = labels.to(DEVICE)

        preds = model(imgs, feats).squeeze()
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ========================
    # VALIDATION (PATIENT LEVEL)
    # ========================
    model.eval()
    patient_dict = {}

    with torch.no_grad():
        for imgs, feats, labels, pids in val_loader:
            imgs = imgs.to(DEVICE)
            feats = feats.to(DEVICE)

            preds = model(imgs, feats).squeeze().cpu().numpy()

            for pid, pred, label in zip(pids, preds, labels.numpy()):
                if pid not in patient_dict:
                    patient_dict[pid] = {"preds": [], "label": label}
                
                patient_dict[pid]["preds"].append(pred)

    final_preds = []
    final_labels = []

    for pid in patient_dict:
        avg_pred = np.mean(patient_dict[pid]["preds"])
        final_preds.append(avg_pred)
        final_labels.append(patient_dict[pid]["label"])

    final_preds = np.array(final_preds)
    final_labels = np.array(final_labels)

    # DENORMALIZE
    final_preds = final_preds * std_bil + mean_bil
    final_labels = final_labels * std_bil + mean_bil

    mae = mean_absolute_error(final_labels, final_preds)
    r2 = r2_score(final_labels, final_preds)

    print(f"\nEpoch {epoch+1}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Best model saved!")

print("\nFINAL BEST R2:", best_r2)