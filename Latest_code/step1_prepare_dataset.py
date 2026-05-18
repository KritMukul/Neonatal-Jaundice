# =============================================================================
# step1_prepare_dataset.py
# Purpose : Load CSV, apply 15 mg/dL bilirubin threshold, split 80/20
#           at PATIENT level (prevents data leakage across body-region images).
# Outputs : Latest_code/train_split.csv  &  Latest_code/test_split.csv
# Paper   : Section "Ground truth labeling" + "Data collection"
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import (CSV_PATH, IMAGE_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV,
                    BILIRUBIN_THRESHOLD, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED)

def main():
    print("=" * 60)
    print("STEP 1 — Dataset Preparation")
    print("=" * 60)

    # ── 1. Load CSV ────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV  : {CSV_PATH}")
    print(f"Total rows  : {len(df)}")
    print(f"Columns     : {list(df.columns)}\n")

    # ── 2. Apply bilirubin threshold → binary label ────────────────
    # Paper: "A bilirubin threshold of 15 mg/dL was used to define
    #         jaundice status"
    #  >=15 mg/dL  →  Jaundiced     (label = 1)
    #  < 15 mg/dL  →  Non-jaundiced (label = 0)
    df["label"] = (df["blood(mg/dL)"] >= BILIRUBIN_THRESHOLD).astype(int)

    total_jaundiced     = (df["label"] == 1).sum()
    total_non_jaundiced = (df["label"] == 0).sum()
    print(f"Threshold       : {BILIRUBIN_THRESHOLD} mg/dL")
    print(f"Jaundiced rows  : {total_jaundiced}")
    print(f"Non-jaundiced   : {total_non_jaundiced}")

    # ── 3. Verify image files exist ────────────────────────────────
    df["image_path"] = df["image_idx"].apply(
        lambda x: os.path.join(IMAGE_DIR, str(x))
    )
    missing = df[~df["image_path"].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"\nWARNING: {len(missing)} images not found — removing from dataset.")
        df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    else:
        print(f"Image check     : All {len(df)} images found.")

    # ── 4. Patient-level 70 / 10 / 20 stratified split ───────────────
    # All 3 body-region images of one patient stay in the SAME split
    # to prevent data leakage.
    patient_labels = (
        df.groupby("patient_id")["label"]
        .first()
        .reset_index()
    )

    # Step A: split off 20% test
    trainval_patients, test_patients = train_test_split(
        patient_labels["patient_id"],
        test_size    = 1.0 - TRAIN_SPLIT - VAL_SPLIT,   # 0.20
        stratify     = patient_labels["label"],
        random_state = RANDOM_SEED,
    )

    # Step B: from remaining 80%, carve out val (10% of total = 12.5% of 80%)
    label_map        = patient_labels.set_index("patient_id")["label"]
    trainval_strat   = label_map.loc[trainval_patients].values
    val_fraction     = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)   # 0.10/0.80 = 0.125

    train_patients, val_patients = train_test_split(
        trainval_patients,
        test_size    = val_fraction,
        stratify     = trainval_strat,
        random_state = RANDOM_SEED,
    )

    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df  = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

    # ── 5. Print split statistics ──────────────────────────────────
    def _summary(name, split_df, n_patients):
        j  = split_df["label"].sum()
        nj = (split_df["label"] == 0).sum()
        print(f"{name:6s} images : {len(split_df):5d}  "
              f"(jaundiced={j}, non-jaundiced={nj}, patients={n_patients})")

    print("\n── Split Summary ──────────────────────────────────────")
    _summary("Train", train_df, len(train_patients))
    _summary("Val",   val_df,   len(val_patients))
    _summary("Test",  test_df,  len(test_patients))

    # ── 6. Save splits ────────────────────────────────────────────
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,     index=False)
    test_df.to_csv(TEST_CSV,   index=False)
    print(f"\nSaved: {TRAIN_CSV}")
    print(f"Saved: {VAL_CSV}")
    print(f"Saved: {TEST_CSV}")
    print("\nSTEP 1 complete.\n")


if __name__ == "__main__":
    main()
