import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMG_PATH = "NeoJaundice/NeoJaundice/images/"
df = pd.read_csv("processed_data.csv")

def white_balance(img):
    # convert to float
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])

    lab[:, :, 1] = lab[:, :, 1] - (avg_a - 128)
    lab[:, :, 2] = lab[:, :, 2] - (avg_b - 128)

    # clip values
    lab = np.clip(lab, 0, 255)

    # convert back to uint8
    lab = lab.astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def extract_features(img):
    feats = []

    def stats(c):
        return [np.mean(c), np.std(c)]

    # RGB
    for i in range(3):
        feats.extend(stats(img[:,:,i]))

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        feats.extend(stats(hsv[:,:,i]))

    return feats  # 12 features

X, y = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join(IMG_PATH, row["image_idx"])

    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = white_balance(img)

    feats = extract_features(img)

    X.append(feats)
    y.append(row["blood(mg/dL)"])

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Features extracted:", X.shape)