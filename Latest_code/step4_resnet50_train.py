# =============================================================================
# step4_resnet50_train.py
# Purpose : Train ResNet-50 baseline — same config as T2T-ViT.
#
# Paper (Section "ResNet-50"):
#   "network was adapted to binary classification by replacing the final
#    fully connected layer with a single output node followed by sigmoid.
#    ResNet-50 was initialized with ImageNet-pretrained weights and
#    subsequently fine-tuned … under the same training configuration."
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from config import (TRAIN_CSV, VAL_CSV, MODELS_DIR, RESULTS_DIR,
                    IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS,
                    IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED,
                    POS_WEIGHT, EARLY_STOPPING_PATIENCE,
                    AUG_ROTATION_DEG, AUG_BRIGHTNESS, AUG_CONTRAST,
                    AUG_SATURATION, AUG_HUE, AUG_SCALE_MIN, AUG_SCALE_MAX,
                    AUG_SHEAR, AUG_BLUR_SIGMA, AUG_ERASING_PROB)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Dataset (identical to step3) ──────────────────────────────────────────

class NeonatalDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
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
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms (same as T2T-ViT) ─────────────────────────────────────────

train_transform = transforms.Compose([
    # Spatial transforms
    transforms.RandomRotation(AUG_ROTATION_DEG),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, shear=AUG_SHEAR),   # skin angle variation
    transforms.RandomResizedCrop(IMAGE_SIZE,
                                 scale=(AUG_SCALE_MIN, AUG_SCALE_MAX)),

    # Colour transforms — hue jitter is critical for jaundice yellow detection
    transforms.ColorJitter(
        brightness=AUG_BRIGHTNESS,
        contrast=AUG_CONTRAST,
        saturation=AUG_SATURATION,
        hue=AUG_HUE,
    ),

    # Blur — simulates camera focus variation
    transforms.GaussianBlur(kernel_size=3, sigma=AUG_BLUR_SIGMA),

    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

    # Erasing — simulates partial occlusion (applied after normalisation)
    transforms.RandomErasing(p=AUG_ERASING_PROB, scale=(0.02, 0.10),
                             ratio=(0.3, 3.3), value=0),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def make_weighted_sampler(dataset: NeonatalDataset):
    labels        = dataset.df["label"].values
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights     = torch.DoubleTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True,
    )


# ── Model ─────────────────────────────────────────────────────────────────

def build_resnet50():
    """
    Paper: "replacing the final fully connected layer with a single output
            node followed by a sigmoid activation"
    We use BCEWithLogitsLoss so we don't add sigmoid here — it's in the loss.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


# ── Training helpers ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds       = (torch.sigmoid(logits) >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        all_probs.extend(probs.squeeze(1).cpu().numpy().tolist())
        all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / total, correct / total, auc


# ── Early stopping ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best     = -float("inf")
        self.counter  = 0
        self.stop     = False

    def step(self, metric: float) -> bool:
        if metric > self.best:
            self.best    = metric
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


def save_curves(train_losses, val_losses, train_accs, val_accs, val_aucs,
                prefix="resnet50"):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss",  color="blue")
    axes[0].plot(epochs, val_losses,   label="Val Loss",    color="orange")
    axes[0].set_title("Loss Curve (ResNet-50)")
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, train_accs, label="Train Accuracy", color="blue")
    axes[1].plot(epochs, val_accs,   label="Val Accuracy",   color="orange")
    axes[1].set_title("Accuracy Over Epochs (ResNet-50)")
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, val_aucs, label="Val AUC", color="green")
    axes[2].set_title("Validation AUC Over Epochs (ResNet-50)")
    axes[2].set_xlabel("Epochs"); axes[2].set_ylabel("AUC")
    axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{prefix}_learning_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 4 — ResNet-50 Training (Baseline)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = NeonatalDataset(TRAIN_CSV, transform=train_transform)
    val_ds   = NeonatalDataset(VAL_CSV,   transform=test_transform)
    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = build_resnet50().to(device)

    # Weighted loss: up-weights minority (jaundice) class
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer     = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                     weight_decay=1e-4)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    print(f"\nModel       : ResNet-50")
    print(f"Loss        : BCEWithLogitsLoss  pos_weight={POS_WEIGHT:.3f}")
    print(f"Optimizer   : Adam  lr={LEARNING_RATE}  weight_decay=1e-4")
    print(f"Batch       : {BATCH_SIZE}  |  Max epochs: {EPOCHS}")
    print(f"Early stop  : patience={EARLY_STOPPING_PATIENCE} (val AUC)\n")

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    val_aucs                 = []
    best_val_auc             = 0.0
    stopped_epoch            = EPOCHS

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc         = train_one_epoch(model, train_loader,
                                                  criterion, optimizer, device)
        va_loss, va_acc, va_auc = evaluate(model, val_loader,
                                           criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc * 100)
        val_accs.append(va_acc * 100)
        val_aucs.append(va_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}]  "
                  f"Train Loss={tr_loss:.4f}  Acc={tr_acc*100:.2f}%  |  "
                  f"Val Loss={va_loss:.4f}  Acc={va_acc*100:.2f}%  AUC={va_auc:.4f}")

        improved = early_stopper.step(va_auc)
        if improved:
            best_val_auc = va_auc
            ckpt = os.path.join(MODELS_DIR, "resnet50_best.pth")
            torch.save(model.state_dict(), ckpt)

        if early_stopper.stop:
            print(f"\nEarly stopping at epoch {epoch}  "
                  f"(best val AUC={best_val_auc:.4f})")
            break

    final_ckpt = os.path.join(MODELS_DIR, "resnet50_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"\nBest Val AUC : {best_val_auc:.4f}")
    print(f"Saved model  : {final_ckpt}")

    save_curves(train_losses, val_losses, train_accs, val_accs, val_aucs,
                prefix="resnet50")

    curve_df = pd.DataFrame({
        "epoch":      range(1, len(train_losses) + 1),
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc":  train_accs,   "val_acc":  val_accs,
        "val_auc":    val_aucs,
    })
    curve_df.to_csv(os.path.join(RESULTS_DIR, "resnet50_curves.csv"), index=False)

    print("\nSTEP 4 complete.\n")


if __name__ == "__main__":
    main()
