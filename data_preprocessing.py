"""
Neonatal Jaundice Detection - Data Preprocessing
Step 3: Data Preprocessing & Preparation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'DATA_PATH': 'NeoJaundice/NeoJaundice/chd_jaundice_published_2.csv',
    'IMAGES_PATH': 'NeoJaundice/NeoJaundice/images/',
    'OUTPUT_DIR': 'preprocessed_data/',
    
    # Image parameters
    'IMG_HEIGHT': 224,
    'IMG_WIDTH': 224,
    'IMG_CHANNELS': 3,
    
    # Split ratios
    'TRAIN_RATIO': 0.70,
    'VAL_RATIO': 0.15,
    'TEST_RATIO': 0.15,
    
    # Other parameters
    'RANDOM_SEED': 14
}

# Create output directory
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

print("="*70)
print("NEONATAL JAUNDICE DETECTION - DATA PREPROCESSING")
print("="*70)
print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# 2. LOAD DATASET
# ============================================================================

print("\n" + "="*70)
print("LOADING DATASET")
print("="*70)

df = pd.read_csv(CONFIG['DATA_PATH'])
print(f"\nDataset loaded: {len(df)} records")
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\nMissing values found:")
    print(missing[missing > 0])
else:
    print(f"\nNo missing values")

# ============================================================================
# 3. PATIENT-LEVEL TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("CREATING TRAIN/VAL/TEST SPLIT")
print("="*70)

# Get unique patients with their labels
patient_labels = df.groupby('patient_id')['Treatment'].first().reset_index()
patient_ids = patient_labels['patient_id'].values
labels = patient_labels['Treatment'].values

print(f"\nTotal patients: {len(patient_ids)}")
print(f"  Class 0 (No Treatment): {sum(labels == 0)}")
print(f"  Class 1 (Treatment): {sum(labels == 1)}")

# First split: Train + Val vs Test (stratified by treatment)
train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
    patient_ids, 
    labels,
    test_size=CONFIG['TEST_RATIO'],
    random_state=CONFIG['RANDOM_SEED'],
    stratify=labels
)

# Second split: Train vs Val
val_ratio_adjusted = CONFIG['VAL_RATIO'] / (CONFIG['TRAIN_RATIO'] + CONFIG['VAL_RATIO'])
train_ids, val_ids, train_labels, val_labels = train_test_split(
    train_val_ids,
    train_val_labels,
    test_size=val_ratio_adjusted,
    random_state=CONFIG['RANDOM_SEED'],
    stratify=train_val_labels
)

print(f"\nSplit completed:")
print(f"  Train: {len(train_ids)} patients ({len(train_ids)/len(patient_ids)*100:.1f}%)")
print(f"    - Class 0: {sum(train_labels == 0)}, Class 1: {sum(train_labels == 1)}")
print(f"  Val: {len(val_ids)} patients ({len(val_ids)/len(patient_ids)*100:.1f}%)")
print(f"    - Class 0: {sum(val_labels == 0)}, Class 1: {sum(val_labels == 1)}")
print(f"  Test: {len(test_ids)} patients ({len(test_ids)/len(patient_ids)*100:.1f}%)")
print(f"    - Class 0: {sum(test_labels == 0)}, Class 1: {sum(test_labels == 1)}")

# Create train, val, test dataframes
train_df = df[df['patient_id'].isin(train_ids)].copy()
val_df = df[df['patient_id'].isin(val_ids)].copy()
test_df = df[df['patient_id'].isin(test_ids)].copy()

print(f"\nDataframes created:")
print(f"  Train images: {len(train_df)}")
print(f"  Val images: {len(val_df)}")
print(f"  Test images: {len(test_df)}")

# ============================================================================
# 4. COLOR FEATURES EXTRACTION FROM IMAGES
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING COLOR FEATURES FROM IMAGES")
print("="*70)

def extract_color_features(image_path):
    """
    Extract 29 jaundice-specific color features from image
    Color spaces: RGB, HSV, LAB, YCrCb
    Returns 29-dimensional feature vector:
    
    ✅ Core Color Means (12 features):
    - RGB mean (3)
    - HSV mean (3)  
    - LAB mean (3)
    - YCrCb mean (3)
    
    ✅ Standard Deviations (9 features):
    - RGB std (3)
    - HSV std (3)
    - LAB std (3)
    
    ✅ Yellow-Dominance Features (8 features):
    - (R+G)/2 - B: Yellow strength
    - R/B: Red-to-blue ratio
    - G/B: Green-to-blue ratio  
    - R/G: Red-to-green ratio
    - LAB_b: Direct yellow axis
    - LAB_b/LAB_L: Normalized yellow
    - Cr-Cb: YCrCb yellow indicator
    - HSV_S×HSV_V: Saturation-value product
    
    Preprocessing steps:
    1. Resize to 224x224
    2. Color calibration using gray patches in calibration card
    3. Extract ONLY skin region from center (exclude calibration card)
    4. Multi-color-space feature extraction from corrected skin
    5. Compute jaundice-specific features based on bilirubin physics
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize to standard size
        img = cv2.resize(img, (CONFIG['IMG_WIDTH'], CONFIG['IMG_HEIGHT']))
        
        # Color calibration using gray world on peripheral regions (calibration card)
        # Sample gray patches from corners where calibration squares are located
        h, w, _ = img.shape
        patch_size = int(h * 0.1)  # 10% of image size
        
        # Extract corner gray patches (top-left, top-right, bottom-left, bottom-right)
        gray_patches = []
        gray_patches.append(img[0:patch_size, 0:patch_size])  # top-left
        gray_patches.append(img[0:patch_size, -patch_size:])  # top-right
        gray_patches.append(img[-patch_size:, 0:patch_size])  # bottom-left
        gray_patches.append(img[-patch_size:, -patch_size:])  # bottom-right
        
        # Calculate average color from gray patches for calibration
        gray_ref = np.mean([patch.mean(axis=(0, 1)) for patch in gray_patches], axis=0)
        avg_gray = gray_ref.mean()
        
        # Apply color correction to entire image based on calibration
        img_float = img.astype(np.float32)
        if gray_ref[0] > 0 and gray_ref[1] > 0 and gray_ref[2] > 0:
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (avg_gray / gray_ref[0]), 0, 255)
            img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (avg_gray / gray_ref[1]), 0, 255)
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (avg_gray / gray_ref[2]), 0, 255)
        
        img_corrected = img_float.astype(np.uint8)
        
        # Extract ONLY center skin region (exclude calibration card borders)
        # Use inner 40% to avoid calibration patches
        skin_roi = img_corrected[int(0.3*h):int(0.7*h), int(0.3*w):int(0.7*w)]
        
        # Extract features from multiple color spaces (SKIN REGION ONLY)
        epsilon = 1e-6  # Avoid division by zero
        
        # ===== 1. CORE COLOR MEANS (12 features) =====
        # RGB features (BGR in OpenCV)
        rgb_mean = skin_roi.mean(axis=(0, 1))  # B, G, R means
        b_mean, g_mean, r_mean = rgb_mean[0], rgb_mean[1], rgb_mean[2]
        
        # HSV features
        hsv = cv2.cvtColor(skin_roi, cv2.COLOR_BGR2HSV)
        hsv_mean = hsv.mean(axis=(0, 1))  # H, S, V means
        h_mean, s_mean, v_mean = hsv_mean[0], hsv_mean[1], hsv_mean[2]
        
        # LAB features
        lab = cv2.cvtColor(skin_roi, cv2.COLOR_BGR2LAB)
        lab_raw = lab.mean(axis=(0, 1))  # L, A, B means (raw from OpenCV)
        l_mean, a_mean, b_lab_raw = lab_raw[0], lab_raw[1], lab_raw[2]
        # Center LAB A and B channels: OpenCV A,B ∈ [0,255] with 128=neutral
        # A_true = A_raw - 128 (negative=green, positive=red)
        # B_true = B_raw - 128 (negative=blue, positive=yellow)
        a_mean = a_mean - 128
        b_lab_mean = b_lab_raw - 128
        lab_mean = np.array([l_mean, a_mean, b_lab_mean])  # Centered LAB
        
        # YCrCb features
        ycrcb = cv2.cvtColor(skin_roi, cv2.COLOR_BGR2YCrCb)
        ycrcb_mean = ycrcb.mean(axis=(0, 1))  # Y, Cr, Cb means
        y_mean, cr_mean, cb_mean = ycrcb_mean[0], ycrcb_mean[1], ycrcb_mean[2]
        
        # ===== 2. STANDARD DEVIATIONS (9 features) =====
        # Captures bilirubin variation across skin
        rgb_std = skin_roi.std(axis=(0, 1))  # B, G, R std
        hsv_std = hsv.std(axis=(0, 1))  # H, S, V std
        lab_std_raw = lab.std(axis=(0, 1))  # L, A, B std (raw)
        # Note: std is unaffected by centering, but for consistency with means
        lab_std = lab_std_raw  # L_std, A_std, B_std
        
        # ===== 3. YELLOW-DOMINANCE FEATURES (8 features) =====
        # These directly model jaundice/bilirubin physics
        
        # Yellow strength: (R+G)/2 - B
        yellow_strength = ((r_mean + g_mean) / 2.0) - b_mean
        
        # R/B ratio: Red-to-blue (high in jaundice)
        r_b_ratio = r_mean / (b_mean + epsilon)
        
        # G/B ratio: Green-to-blue (high in jaundice)
        g_b_ratio = g_mean / (b_mean + epsilon)
        
        # R/G ratio: Red-to-green balance
        r_g_ratio = r_mean / (g_mean + epsilon)
        
        # LAB b-channel: Direct yellow axis (positive = yellow)
        lab_b_yellow = b_lab_mean  # Already centered
        
        # LAB b/L ratio: Normalized yellow relative to lightness
        lab_b_l_ratio = b_lab_mean / (l_mean + epsilon)  # Uses centered b
        
        # YCrCb Cr-Cb difference: Yellow indicator
        cr_cb_diff = cr_mean - cb_mean
        
        # HSV S×V product: Saturation-value interaction
        sv_product = s_mean * v_mean
        
        # ===== COMBINE ALL 29 FEATURES =====
        features = np.concatenate([
            # Core means (12)
            rgb_mean,        # B, G, R (3)
            hsv_mean,        # H, S, V (3)
            lab_mean,        # L, A, B (3)
            ycrcb_mean,      # Y, Cr, Cb (3)
            
            # Standard deviations (9)
            rgb_std,         # B_std, G_std, R_std (3)
            hsv_std,         # H_std, S_std, V_std (3)
            lab_std,         # L_std, A_std, B_std (3)
            
            # Yellow-dominance features (8)
            [yellow_strength, r_b_ratio, g_b_ratio, r_g_ratio,
             lab_b_yellow, lab_b_l_ratio, cr_cb_diff, sv_product]
        ])
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_features_from_df(df_subset, images_path):
    """Extract color features for all images in dataframe"""
    features_list = []
    valid_indices = []
    
    for idx, row in df_subset.iterrows():
        image_name = row['image_idx']
        image_path = os.path.join(images_path, image_name)
        
        features = extract_color_features(image_path)
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)
    
    return np.array(features_list), valid_indices

# Extract features from images
print("\nExtracting color features from training images...")
train_features, train_valid_idx = extract_features_from_df(train_df, CONFIG['IMAGES_PATH'])
print(f"  Extracted: {train_features.shape[0]}/{len(train_df)} images")

print("Extracting color features from validation images...")
val_features, val_valid_idx = extract_features_from_df(val_df, CONFIG['IMAGES_PATH'])
print(f"  Extracted: {val_features.shape[0]}/{len(val_df)} images")

print("Extracting color features from test images...")
test_features, test_valid_idx = extract_features_from_df(test_df, CONFIG['IMAGES_PATH'])
print(f"  Extracted: {test_features.shape[0]}/{len(test_df)} images")

# Filter dataframes to only valid images (use .loc for label-based indexing)
train_df = train_df.loc[train_valid_idx].reset_index(drop=True)
val_df = val_df.loc[val_valid_idx].reset_index(drop=True)
test_df = test_df.loc[test_valid_idx].reset_index(drop=True)

# Normalize color features
print("\nNormalizing color features...")
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

print(f"\nColor features extracted and normalized:")
print(f"  Train: {train_features.shape}")
print(f"  Val: {val_features.shape}")
print(f"  Test: {test_features.shape}")
print(f"  ✅ Core Means: RGB(3), HSV(3), LAB(3), YCrCb(3) = 12 features")
print(f"  ✅ Std Dev: RGB_std(3), HSV_std(3), LAB_std(3) = 9 features")
print(f"  ✅ Yellow-Dominance: Yellow_strength, R/B, G/B, R/G, LAB_b, LAB_b/L, Cr-Cb, S×V = 8 features")
print(f"  Total: 29 jaundice-specific features")
print(f"  Note: LAB A,B channels centered at 128 (yellow axis properly scaled)")
print(f"  Note: Calibration card color correction + Skin-only ROI extraction applied")

# ============================================================================
# 4b. EXTRACT CLINICAL FEATURES (Age + Gender)
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING CLINICAL FEATURES (AGE + GENDER)")
print("="*70)

def extract_clinical_features(df_subset, age_fill=None, gender_fill=None):
    """
    Extract age and gender from clinical data.
    gender: M=1, F=0  (NaN filled with gender_fill)
    age: age(day) as numeric  (NaN filled with age_fill)
    Returns (N, 2) array: [gender_encoded, age_day]
    """
    gender = df_subset['gender'].str.upper().fillna(gender_fill)
    gender_encoded = (gender == 'M').astype(float).values

    age = df_subset['age(day)'].astype(float)
    if age_fill is not None:
        age = age.fillna(age_fill)
    age = age.values

    return np.column_stack([gender_encoded, age])

# Compute fill values from training set only
train_age_median = train_df['age(day)'].astype(float).median()
train_gender_mode = train_df['gender'].str.upper().mode()[0]  # 'M' or 'F'

print(f"\n  Missing value fills (from train set):")
print(f"    age(day) median: {train_age_median}")
print(f"    gender mode:     {train_gender_mode}")

train_clinical_raw = extract_clinical_features(train_df, age_fill=train_age_median, gender_fill=train_gender_mode)
val_clinical_raw   = extract_clinical_features(val_df,   age_fill=train_age_median, gender_fill=train_gender_mode)
test_clinical_raw  = extract_clinical_features(test_df,  age_fill=train_age_median, gender_fill=train_gender_mode)

# Scale clinical features (fit on train only)
clinical_scaler = StandardScaler()
train_clinical = clinical_scaler.fit_transform(train_clinical_raw)
val_clinical = clinical_scaler.transform(val_clinical_raw)
test_clinical = clinical_scaler.transform(test_clinical_raw)

print(f"\nClinical features extracted and normalized:")
print(f"  Train: {train_clinical.shape}")
print(f"  Val:   {val_clinical.shape}")
print(f"  Test:  {test_clinical.shape}")
print(f"  Features: [gender (M=1/F=0), age(day)]")
print(f"  Normalization: StandardScaler (fit on train)")

# ============================================================================
# 5. SAVE PREPROCESSING ARTIFACTS
# ============================================================================

print("\n" + "="*70)
print("SAVING PREPROCESSING ARTIFACTS")
print("="*70)

# Save train/val/test split with reset index to avoid pickle issues
split_data = {
    'train_ids': train_ids,
    'val_ids': val_ids,
    'test_ids': test_ids,
    'train_df': train_df.reset_index(drop=True),
    'val_df': val_df.reset_index(drop=True),
    'test_df': test_df.reset_index(drop=True)
}

with open(os.path.join(CONFIG['OUTPUT_DIR'], 'data_split.pkl'), 'wb') as f:
    pickle.dump(split_data, f, protocol=4)
print(f"Saved: data_split.pkl")

# Save scaler
with open(os.path.join(CONFIG['OUTPUT_DIR'], 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved: scaler.pkl")

# Save clinical scaler + imputation fill values
clinical_meta = {
    'scaler': clinical_scaler,
    'age_fill': train_age_median,
    'gender_fill': train_gender_mode
}
with open(os.path.join(CONFIG['OUTPUT_DIR'], 'clinical_scaler.pkl'), 'wb') as f:
    pickle.dump(clinical_meta, f)
print(f"Saved: clinical_scaler.pkl")

# Save color features
np.savez(os.path.join(CONFIG['OUTPUT_DIR'], 'color_features.npz'),
         train_features=train_features,
         val_features=val_features,
         test_features=test_features)
print(f"Saved: color_features.npz")

# Save clinical features
np.savez(os.path.join(CONFIG['OUTPUT_DIR'], 'clinical_features.npz'),
         train_clinical=train_clinical,
         val_clinical=val_clinical,
         test_clinical=test_clinical)
print(f"Saved: clinical_features.npz")

# Save configuration
with open(os.path.join(CONFIG['OUTPUT_DIR'], 'config.pkl'), 'wb') as f:
    pickle.dump(CONFIG, f)
print(f"Saved: config.pkl")

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DATA PREPROCESSING SUMMARY")
print("="*70)

print(f"\n1. DATA SPLIT:")
print(f"   Training Set:")
print(f"     - Patients: {len(train_ids)} ({len(train_ids)/len(patient_ids)*100:.1f}%)")
print(f"     - Images: {len(train_df)}")
print(f"     - Class 0: {sum(train_labels == 0)}, Class 1: {sum(train_labels == 1)}")
print(f"\n   Validation Set:")
print(f"     - Patients: {len(val_ids)} ({len(val_ids)/len(patient_ids)*100:.1f}%)")
print(f"     - Images: {len(val_df)}")
print(f"     - Class 0: {sum(val_labels == 0)}, Class 1: {sum(val_labels == 1)}")
print(f"\n   Test Set:")
print(f"     - Patients: {len(test_ids)} ({len(test_ids)/len(patient_ids)*100:.1f}%)")
print(f"     - Images: {len(test_df)}")
print(f"     - Class 0: {sum(test_labels == 0)}, Class 1: {sum(test_labels == 1)}")

print(f"\n2. JAUNDICE-SPECIFIC COLOR FEATURES (Image-based only, no clinical data):")
print(f"   - Feature vector size: {train_features.shape[1]} features")
print(f"   - ✅ Core Means (12): RGB, HSV, LAB, YCrCb")
print(f"   - ✅ Standard Deviations (9): RGB_std, HSV_std, LAB_std")
print(f"   - ✅ Yellow-Dominance (8): Yellow_strength, R/B, G/B, R/G, LAB_b, LAB_b/L, Cr-Cb, S×V")
print(f"   - Preprocessing: Calibration card color correction → Skin ROI extraction")
print(f"   - LAB Scaling: A,B channels centered at 128 (yellow axis properly scaled)")
print(f"   - Calibration: Using gray patches from card corners")
print(f"   - ROI: Center 40% (skin only, excludes calibration card)")
print(f"   - Normalization: StandardScaler")

print(f"\n3. SAVED ARTIFACTS:")
print(f"   - Location: {CONFIG['OUTPUT_DIR']}")
print(f"   - Files: data_split.pkl, scaler.pkl, color_features.npz, config.pkl")

print(f"\nDATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print(f"\nNext step: Run model_training.py")
print("="*70)
