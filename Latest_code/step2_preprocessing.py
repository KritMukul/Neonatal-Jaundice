# =============================================================================
# step2_preprocessing.py
# Purpose : Full preprocessing pipeline as described in the paper.
#           Input  : raw images from IMAGE_DIR
#           Output : 224×224 preprocessed images in PREPROCESSED_DIR
#
# KEY DATASET NOTE:
#   In the NeoJaundice dataset, the color calibration card is a SQUARE
#   FRAME/BORDER surrounding the skin patch — the baby's skin is in the
#   CENTER HOLE of the card. Strategy:
#     1. Detect the colorful card border (high-saturation outer ring)
#     2. Extract the inner rectangle (the skin patch in the center)
#     3. Apply preprocessing only on that inner skin region
#
# Pipeline (paper Fig. 4):
#   1. Load image
#   2. Detect & crop inner skin patch (remove card frame)
#   3. White-balance / color calibration using card border reference
#   4. CLAHE contrast enhancement
#   5. Color space conversion RGB → HSV & RGB → YCbCr
#   6. Skin segmentation (HSV + YCbCr thresholding + morphology)
#   7. ROI refinement & resize to 224×224
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import (TRAIN_CSV, VAL_CSV, TEST_CSV, IMAGE_DIR, PREPROCESSED_DIR,
                    IMAGE_SIZE,
                    HSV_LOWER, HSV_UPPER, YCBCR_LOWER, YCBCR_UPPER,
                    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID)


# ── Step A: Extract the inner skin patch from inside the card frame ────────

def extract_inner_patch(image_rgb: np.ndarray) -> np.ndarray:
    """
    The calibration card is a colorful square FRAME surrounding the skin.
    This function finds the inner rectangle (the skin region inside the frame).

    Strategy:
      1. Detect the card frame using its highly saturated, multi-colored border
      2. Build a mask for NON-card (inner + background) regions
      3. Find the largest compact central region → that is the skin patch
      4. Crop it with padding
    Falls back to a centre-crop if detection fails.
    """
    h, w = image_rgb.shape[:2]
    hsv  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # ── Card border detection
    # The card squares are highly saturated (S > 80) and moderately bright
    # The checkerboard adds black/white regions — treat those separately
    sat   = hsv[:, :, 1]
    val   = hsv[:, :, 2]

    # High-saturation mask → colored card squares
    mask_colored = ((sat > 80) & (val > 40)).astype(np.uint8) * 255

    # Black checkerboard squares on the card
    mask_black = (val < 50).astype(np.uint8) * 255

    # White checkerboard squares on the card
    mask_white = ((val > 200) & (sat < 40)).astype(np.uint8) * 255

    # Card mask = colored + black + white
    card_mask = cv2.bitwise_or(mask_colored, mask_black)
    card_mask = cv2.bitwise_or(card_mask, mask_white)

    # Dilate card mask to close gaps between patches
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    card_mask  = cv2.dilate(card_mask, kernel_lg, iterations=3)

    # Inner region = everything NOT part of the card
    inner_mask = cv2.bitwise_not(card_mask)

    # Remove very small noise blobs
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN,
                                  kernel_sm, iterations=2)

    # ── Find the largest inner contour (the skin patch)
    contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours: must be near the image centre and reasonably large
    cx_img, cy_img = w // 2, h // 2
    min_area = (w * h) * 0.03   # at least 3% of image area
    centre_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        mx, my, mw, mh = cv2.boundingRect(cnt)
        # Centre of this contour
        cx_c = mx + mw // 2
        cy_c = my + mh // 2
        # Must be within 40% of image centre
        if (abs(cx_c - cx_img) < w * 0.40 and
                abs(cy_c - cy_img) < h * 0.40):
            centre_contours.append((area, cnt))

    if centre_contours:
        # Pick the largest central contour
        _, best_cnt = max(centre_contours, key=lambda x: x[0])
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        patch = image_rgb[y1:y2, x1:x2]
        if patch.size > 0:
            return patch

    # ── Fallback: crop the central 40% of the image
    margin_x = int(w * 0.30)
    margin_y = int(h * 0.30)
    fallback  = image_rgb[margin_y:h - margin_y, margin_x:w - margin_x]
    if fallback.size == 0:
        return image_rgb
    return fallback


# ── Step B: White balance using gray-world on the skin patch ───────────────

def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """
    White-balance via the Gray World assumption applied to skin patch.
    Paper: "Histogram matching and color correction algorithms were applied,
            with white balance and luminance adjusted according to the
            reference patches on the card."
    """
    img = image_rgb.astype(np.float32)
    mean_r   = img[:, :, 0].mean()
    mean_g   = img[:, :, 1].mean()
    mean_b   = img[:, :, 2].mean()
    mean_all = (mean_r + mean_g + mean_b) / 3.0

    if mean_r > 0:
        img[:, :, 0] *= mean_all / mean_r
    if mean_g > 0:
        img[:, :, 1] *= mean_all / mean_g
    if mean_b > 0:
        img[:, :, 2] *= mean_all / mean_b

    return np.clip(img, 0, 255).astype(np.uint8)


# ── Step C: CLAHE contrast enhancement ────────────────────────────────────

def apply_clahe(image_rgb: np.ndarray,
                clip_limit: float = CLAHE_CLIP_LIMIT,
                tile_grid: tuple  = CLAHE_TILE_GRID) -> np.ndarray:
    """
    CLAHE on LAB L-channel.
    Paper: "Contrast Limited Adaptive Histogram Equalization (CLAHE) …
            increasing local contrast while avoiding excessive noise."
    """
    lab   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ── Step D: Skin segmentation on the extracted patch ──────────────────────

def build_skin_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Skin segmentation by combining HSV and YCbCr thresholds.
    Paper: "The script combined thresholding in both HSV and YCbCr spaces,
            followed by morphological operations to suppress non-skin areas."
    Applied AFTER the card frame has already been removed, so we are only
    looking at the skin patch — greatly reducing false positives.
    """
    hsv      = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask_hsv = cv2.inRange(hsv,
                           np.array(HSV_LOWER, dtype=np.uint8),
                           np.array(HSV_UPPER, dtype=np.uint8))

    ycrcb    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    y   = ycrcb[:, :, 0]
    cr  = ycrcb[:, :, 1]
    cb  = ycrcb[:, :, 2]

    mask_y   = (y  >= YCBCR_LOWER[0]) & (y  <= YCBCR_UPPER[0])
    mask_cb  = (cb >= YCBCR_LOWER[1]) & (cb <= YCBCR_UPPER[1])
    mask_cr  = (cr >= YCBCR_LOWER[2]) & (cr <= YCBCR_UPPER[2])
    mask_ycbcr = (mask_y & mask_cb & mask_cr).astype(np.uint8) * 255

    combined = cv2.bitwise_and(mask_hsv, mask_ycbcr)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel, iterations=1)
    return cleaned


# ── Step E: Final ROI refinement & resize ─────────────────────────────────

def refine_and_resize(patch_rgb: np.ndarray,
                      mask: np.ndarray,
                      target_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Within the already-extracted skin patch, find the largest skin contour
    for final refinement, then resize to target_size × target_size.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        # Only use contour if it covers at least 15% of the patch
        if area > patch_rgb.shape[0] * patch_rgb.shape[1] * 0.15:
            x, y, bw, bh = cv2.boundingRect(largest)
            pad = 5
            h_p, w_p = patch_rgb.shape[:2]
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w_p, x + bw + pad); y2 = min(h_p, y + bh + pad)
            roi = patch_rgb[y1:y2, x1:x2]
            if roi.size > 0:
                return cv2.resize(roi, (target_size, target_size),
                                  interpolation=cv2.INTER_AREA)

    # Fallback: just resize the whole patch
    return cv2.resize(patch_rgb, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


# ── Full pipeline ──────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray | None:
    """
    Full pipeline for one image. Returns 224×224 RGB numpy array or None.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # A: Detect card frame → extract inner skin patch
    patch = extract_inner_patch(img_rgb)

    # B: White balance on the skin patch
    patch_wb = gray_world_white_balance(patch)

    # C: CLAHE contrast enhancement
    patch_ce = apply_clahe(patch_wb)

    # D: Skin segmentation (HSV + YCbCr)
    mask = build_skin_mask(patch_ce)

    # E: Refine ROI and resize to 224×224
    roi = refine_and_resize(patch_ce, mask)

    return roi   # RGB, uint8, (224, 224, 3)


# ── Main ──────────────────────────────────────────────────────────────────

def process_split(csv_path: str, split_name: str) -> None:
    df      = pd.read_csv(csv_path)
    out_dir = os.path.join(PREPROCESSED_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Preprocessing [{split_name}]"):
        src_path = os.path.join(IMAGE_DIR, row["image_idx"])
        img_name = os.path.splitext(row["image_idx"])[0] + ".png"
        dst_path = os.path.join(out_dir, img_name)

        if os.path.exists(dst_path):
            continue

        roi = preprocess_image(src_path)
        if roi is None:
            print(f"  SKIP (unreadable): {src_path}")
            skipped += 1
            continue

        cv2.imwrite(dst_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    print(f"[{split_name}] done. Skipped: {skipped}  → saved to {out_dir}\n")

    df["preprocessed_path"] = df["image_idx"].apply(
        lambda x: os.path.join(out_dir, os.path.splitext(x)[0] + ".png")
    )
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with preprocessed_path column.")


def main():
    print("=" * 60)
    print("STEP 2 — Image Preprocessing Pipeline")
    print("(Card-frame aware: extracts inner skin patch)")
    print("=" * 60)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Delete old preprocessed images so they get regenerated cleanly
    import shutil
    for split in ["train", "val", "test"]:
        old_dir = os.path.join(PREPROCESSED_DIR, split)
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
            print(f"Cleared old preprocessed dir: {old_dir}")

    for split, csv in [("train", TRAIN_CSV), ("val", VAL_CSV), ("test", TEST_CSV)]:
        if not os.path.exists(csv):
            print(f"ERROR: {csv} not found. Run step1 first.")
            continue
        process_split(csv, split)

    print("STEP 2 complete.\n")


if __name__ == "__main__":
    main()
