import os
import warnings
import pickle
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold

from timm import create_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")

# ================= GPU =================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ================= CONFIG =================
CONFIG = {
    "MODEL_NAME": "vit_base_patch16_224",
    "IMG_SIZE": 224,
    "BATCH_SIZE": 16,
    "EPOCHS": 25,
    "LR": 3e-5,
    "WEIGHT_DECAY": 1e-4,
    "N_FOLDS": 10,
    "WARMUP_EPOCHS": 2,
    "IMAGES_PATH": "NeoJaundice/NeoJaundice/images/",
}

# ================= DATASET =================
class JaundiceDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df.reset_index(drop=True)
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.path, row["image_idx"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = 1 if row["blood(mg/dL)"] >= 15 else 0
        return img, torch.tensor(label, dtype=torch.long)

# ================= TRANSFORMS =================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= MODEL =================
class ViTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = create_model(CONFIG["MODEL_NAME"], pretrained=True, num_classes=0)

        self.head = nn.Sequential(
            nn.Linear(self.vit.num_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.vit(x)
        return self.head(x)

# ================= TRAIN =================
def train_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for imgs, labels in loop:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

# ================= EVAL =================
def evaluate(model, loader, epoch):
    model.eval()
    preds, probs, actuals = [], [], []

    loop = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)

    with torch.no_grad():
        for imgs, labels in loop:
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)

            prob = torch.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)

            preds.extend(pred.cpu().numpy())
            probs.extend(prob[:,1].cpu().numpy())
            actuals.extend(labels.numpy())

    acc = accuracy_score(actuals, preds)
    sens = recall_score(actuals, preds, zero_division=0)
    cm = confusion_matrix(actuals, preds)
    spec = cm[0,0] / (cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
    f1 = f1_score(actuals, preds, zero_division=0)
    auc = roc_auc_score(actuals, probs)

    return acc, sens, spec, f1, auc

# ================= LOAD DATA =================
with open("preprocessed_data/data_split.pkl", "rb") as f:
    data = pickle.load(f)

full_df = pd.concat([data["train_df"], data["val_df"]]).reset_index(drop=True)

# ================= K-FOLD =================
kf = KFold(n_splits=CONFIG["N_FOLDS"], shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):

    print(f"\n🚀 ===== Fold {fold+1}/{CONFIG['N_FOLDS']} =====")

    train_loader = DataLoader(
        JaundiceDataset(full_df.iloc[train_idx], CONFIG["IMAGES_PATH"], train_transform),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        JaundiceDataset(full_df.iloc[val_idx], CONFIG["IMAGES_PATH"], val_transform),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = ViTClassifier().to(device)

    # Freeze backbone
    for p in model.vit.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["LR"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_auc = 0

    # Epoch loop
    for epoch in tqdm(range(CONFIG["EPOCHS"]), desc="Epochs"):

        if epoch == CONFIG["WARMUP_EPOCHS"]:
            print("🔓 Unfreezing backbone...")
            for p in model.vit.parameters():
                p.requires_grad = True

        loss = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch+1)
        acc, sens, spec, f1, auc = evaluate(model, val_loader, epoch+1)

        scheduler.step()

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, AUC={auc:.4f}, Acc={acc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"vit_best_fold{fold}.pth")

    torch.cuda.empty_cache()

print("\n✅ TRAINING COMPLETE")