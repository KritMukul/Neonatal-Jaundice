import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, mean_squared_error
)
from sklearn.preprocessing import StandardScaler

# ✅ CREATE RESULTS FOLDER
os.makedirs("results", exist_ok=True)

# ================= LOAD =================
X = np.load("X.npy")
y = np.load("y.npy")

y_clf = (y >= 15).astype(int)

# ✅ IMPORTANT: SCALE FEATURES (ElasticNet NEEDS THIS)
scaler = StandardScaler()
X = scaler.fit_transform(X)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

for fold, (tr, val) in enumerate(tqdm(list(kf.split(X, y_clf)))):

    X_tr, X_val = X[tr], X[val]
    y_tr, y_val = y[tr], y[val]
    y_tr_c, y_val_c = y_clf[tr], y_clf[val]

    try:
        # ================= CLASSIFICATION =================
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr, y_tr_c)

        probs = clf.predict_proba(X_val)[:, 1]
        preds = clf.predict(X_val)

        # ================= REGRESSION =================
        reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
        reg.fit(X_tr, y_tr)

        reg_preds = reg.predict(X_val)

        # ================= METRICS =================
        acc = accuracy_score(y_val_c, preds)
        auc = roc_auc_score(y_val_c, probs)
        f1 = f1_score(y_val_c, preds)

        tn, fp, fn, tp = confusion_matrix(y_val_c, preds).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

        rmse = np.sqrt(mean_squared_error(y_val, reg_preds))

        # 🔥 FIX NaN issue
        if np.std(reg_preds) == 0:
            r = 0
        else:
            r = np.corrcoef(y_val, reg_preds)[0, 1]

        print(f"\nFold {fold+1}")
        print(f"AUC: {auc:.3f}, Acc: {acc:.3f}, RMSE: {rmse:.3f}")

        results.append([fold+1, acc, auc, f1, sens, spec, rmse, r])

    except Exception as e:
        print(f"❌ Fold {fold+1} failed:", e)

# ================= SAVE =================
df_res = pd.DataFrame(
    results,
    columns=["Fold","Acc","AUC","F1","Sens","Spec","RMSE","R"]
)

df_res.to_csv("results/elasticnet.csv", index=False)

print("\n✅ elasticnet.csv saved successfully")