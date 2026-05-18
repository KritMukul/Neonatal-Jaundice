import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, mean_squared_error
)

from timm import create_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ================= SETUP =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

os.makedirs("results", exist_ok=True)

IMG_PATH = "NeoJaundice/NeoJaundice/images/"
CSV_PATH = "NeoJaundice/NeoJaundice/chd_jaundice_published_2.csv"

# ================= LOAD =================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df = df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
df["label"] = (df["blood(mg/dL)"] >= 15).astype(int)

# ================= DATASET =================
class JaundiceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(IMG_PATH, row["image_idx"])
        img = cv2.imread(img_path)

        if img is None:
            img = np.zeros((224,224,3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return (
            img,
            torch.tensor(row["label"]),
            torch.tensor(row["blood(mg/dL)"], dtype=torch.float32)
        )

# ================= TRANSFORMS =================
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= MODEL =================
class MultiTaskSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        dim = self.backbone.num_features
        self.clf = nn.Linear(dim, 2)
        self.reg = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.clf(x), self.reg(x).squeeze()

# ================= TRAIN =================
def train_epoch(model, loader, opt, clf_loss, reg_loss, scaler, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for imgs, y_clf, y_reg in loop:
        imgs = imgs.to(device, non_blocking=True)
        y_clf = y_clf.to(device)
        y_reg = y_reg.to(device)

        opt.zero_grad()

        with autocast(device_type="cuda"):
            out_clf, out_reg = model(imgs)
            loss = clf_loss(out_clf, y_clf) + 0.3 * reg_loss(out_reg, y_reg)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loop.set_postfix(loss=loss.item())

# ================= 10-FOLD =================
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

for fold, (tr, val) in enumerate(kf.split(df, df["label"])):

    print(f"\n🔥 Starting Fold {fold+1}")

    train_loader = DataLoader(
        JaundiceDataset(df.iloc[tr], train_tf),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        JaundiceDataset(df.iloc[val], val_tf),
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )

    model = MultiTaskSwin().to(device)

    opt = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    clf_loss = nn.CrossEntropyLoss()
    reg_loss = nn.SmoothL1Loss()
    scaler = GradScaler()

    # TRAIN
    for ep in range(10):
        train_epoch(model, train_loader, opt, clf_loss, reg_loss, scaler, ep+1)

    # EVAL
    model.eval()

    probs, preds, y_true = [], [], []
    reg_preds, reg_true = [], []

    with torch.no_grad():
        for imgs, y_clf, y_reg in tqdm(val_loader, desc=f"Valid Fold {fold+1}", leave=False):
            imgs = imgs.to(device, non_blocking=True)

            out_clf, out_reg = model(imgs)
            p = torch.softmax(out_clf, dim=1)

            probs.extend(p[:,1].cpu().numpy())
            preds.extend(torch.argmax(p,1).cpu().numpy())
            y_true.extend(y_clf.numpy())

            reg_preds.extend(out_reg.cpu().numpy())
            reg_true.extend(y_reg.numpy())

    # ================= CLASSIFICATION =================
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    f1 = f1_score(y_true, preds)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # ================= REGRESSION =================
    rmse = np.sqrt(mean_squared_error(reg_true, reg_preds))

    # 🔥 CORRELATION (R)
    r = np.corrcoef(reg_true, reg_preds)[0,1]

    # PRINT
    print(f"\n🔥 Fold {fold+1} Results")

    print("📌 Classification:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print("\n📌 Regression:")
    print(f"RMSE: {rmse:.3f}")
    print(f"R (Correlation): {r:.4f}")

    results.append([
        fold+1,
        acc, auc, f1, sensitivity, specificity,
        rmse, r
    ])

# ================= SAVE CSV =================
cols = [
    "Fold",
    "Accuracy","AUC","F1","Sensitivity","Specificity",
    "RMSE","R"
]

df_res = pd.DataFrame(results, columns=cols)
df_res.loc["Mean"] = df_res.mean()

df_res.to_csv("results/swin_results.csv", index=False)

print("\n✅ Results saved → results/swin_results.csv")
print(df_res)