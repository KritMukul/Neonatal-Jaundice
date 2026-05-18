import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

# ================= LOAD =================
mlp_preds = np.load("mlp_preds.npy")
swin_preds = np.load("swin_preds.npy")
y_true = np.load("y_true.npy")

# ================= NORMALIZE =================
mlp_preds = (mlp_preds - mlp_preds.min()) / (mlp_preds.max() - mlp_preds.min())
swin_preds = (swin_preds - swin_preds.min()) / (swin_preds.max() - swin_preds.min())

# ================= STACK =================
X_meta = np.vstack([mlp_preds, swin_preds]).T

# ================= MODEL =================
model = XGBClassifier(
    n_estimators=200,
    max_depth=2,          # 🔥 reduce overfitting
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,         # 🔥 regularization
    eval_metric="logloss"
)

# 🔥 TRAIN ON FULL OOF DATA (NO SPLIT)
model.fit(X_meta, y_true)

# ================= PREDICT =================
probs = model.predict_proba(X_meta)[:,1]
preds = (probs > 0.45).astype(int)

# ================= METRICS =================
auc = roc_auc_score(y_true, probs)
acc = accuracy_score(y_true, preds)

print("\n🔥 FINAL META MODEL RESULTS (FIXED)")
print("AUC:", round(auc, 4))
print("Accuracy:", round(acc, 4))