import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMG_PATH = "NeoJaundice/NeoJaundice/images/"
CSV_PATH = "NeoJaundice/NeoJaundice/chd_jaundice_published_2.csv"

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# remove duplicate patients
df = df.drop_duplicates(subset=["patient_id"])

def extract_features(img):
    features = []

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    skin = cv2.bitwise_and(img, img, mask=mask)

    if np.sum(mask) == 0:
        skin = img

    def stats(channel):
        return [
            np.mean(channel),
            np.std(channel),
            np.min(channel),
            np.max(channel)
        ]

    for i in range(3):
        features.extend(stats(skin[:,:,i]))

    hsv_skin = cv2.cvtColor(skin, cv2.COLOR_RGB2HSV)
    for i in range(3):
        features.extend(stats(hsv_skin[:,:,i]))

    lab = cv2.cvtColor(skin, cv2.COLOR_RGB2LAB)
    for i in range(3):
        features.extend(stats(lab[:,:,i]))

    return features

X, y_reg, y_clf = [], [], []

print("Extracting features...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMG_PATH, row["image_idx"])

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    feats = extract_features(img)

    feats.extend([
        row["weight"] if pd.notna(row["weight"]) else 0,
        row["gestational_age"] if pd.notna(row["gestational_age"]) else 0,
        row["age(day)"] if pd.notna(row["age(day)"]) else 0,
        1 if row["gender"] == "M" else 0
    ])

    X.append(feats)
    y_reg.append(row["blood(mg/dL)"])
    y_clf.append(1 if row["blood(mg/dL)"] >= 15 else 0)

X = np.array(X)
y_reg = np.array(y_reg)
y_clf = np.array(y_clf)

mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y_reg = y_reg[mask]
y_clf = y_clf[mask]

np.save("X.npy", X)
np.save("y_reg.npy", y_reg)
np.save("y_clf.npy", y_clf)

print("Done. Shape:", X.shape)