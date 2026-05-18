import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

X = np.load("X.npy")
y_reg = np.load("y_reg.npy")
y_clf = np.load("y_clf.npy")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y_reg = torch.tensor(y_reg, dtype=torch.float32)
y_clf = torch.tensor(y_clf, dtype=torch.float32)

mean = y_reg.mean()
std = y_reg.std()
y_reg = (y_reg - mean) / std

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, loss_fn, epochs=150):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in tqdm(range(epochs)):
        model.train()
        xb = X_train.to(device)
        yb = y_train.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = loss_fn(outputs.squeeze(), yb)
        loss.backward()
        optimizer.step()

    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)

reg_scores = []
clf_scores = []

all_probs = []
all_targets = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    print(f"\nFold {fold+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_reg, y_val_reg = y_reg[train_idx], y_reg[val_idx]
    y_train_clf, y_val_clf = y_clf[train_idx], y_clf[val_idx]

    # Regression
    reg_model = MLP(X.shape[1], 1)
    reg_model = train_model(reg_model, X_train, y_train_reg, nn.MSELoss())

    reg_model.eval()
    preds = reg_model(X_val.to(device)).detach().cpu().numpy()

    preds = preds * std.numpy() + mean.numpy()
    true_vals = y_val_reg.numpy() * std.numpy() + mean.numpy()

    rmse = np.sqrt(mean_squared_error(true_vals, preds))
    reg_scores.append(rmse)

    print("RMSE:", round(rmse, 3))

    # Classification
    clf_model = MLP(X.shape[1], 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

    clf_model = train_model(clf_model, X_train, y_train_clf, loss_fn)

    clf_model.eval()
    outputs = clf_model(X_val.to(device)).squeeze()

    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)

    all_probs.extend(probs)
    all_targets.extend(y_val_clf.numpy())

    acc = accuracy_score(y_val_clf.numpy(), preds)
    auc = roc_auc_score(y_val_clf.numpy(), probs)

    clf_scores.append(auc)

    print("AUC:", round(auc, 4), "| Acc:", round(acc, 4))

print("\nFINAL RESULTS")
print("Avg RMSE:", round(np.mean(reg_scores), 3))
print("Avg AUC:", round(np.mean(clf_scores), 4))

np.save("mlp_preds.npy", np.array(all_probs))
np.save("y_true.npy", np.array(all_targets))

print("Saved mlp_preds.npy & y_true.npy")