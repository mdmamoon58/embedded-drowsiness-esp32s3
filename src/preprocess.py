#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------

# Root where Sub1, Sub2, Sub3 live
RAW_ROOT = Path("data/raw")

# Where processed images will be stored
PROC_ROOT = Path("data/processed")

# Final CSV with all metadata
LABELS_CSV = PROC_ROOT / "labels_subjects.csv"

# Image size for the model
IMG_SIZE = (96, 96)  # (width, height)

# Allowed class names (case-insensitive)
ALLOWED_CLASSES = {"eyeclose", "yawn", "happy", "neutral"}


# ---------- HELPERS ----------

def collect_image_paths() -> List[Tuple[Path, str, str]]:
    """
    Walks data/raw and collects (filepath, subject, label) triples.

    Assumes structure like:
        data/raw/Sub1/Eyeclose/*.jpg
        data/raw/Sub2/Yawn/*.jpg
    """
    samples: List[Tuple[Path, str, str]] = []

    if not RAW_ROOT.is_dir():
        print(f"[ERROR] RAW_ROOT does not exist: {RAW_ROOT}")
        return samples

    for subj_dir in RAW_ROOT.iterdir():
        if not subj_dir.is_dir():
            continue

        subject = subj_dir.name  # e.g., "Sub1"
        print(f"[INFO] Scanning subject folder: {subject}")

        for class_dir in subj_dir.iterdir():
            if not class_dir.is_dir():
                continue

            raw_label = class_dir.name  # e.g., "Eyeclose"
            label_lower = raw_label.lower()

            if label_lower not in ALLOWED_CLASSES:
                print(f"[WARN] Skipping unknown class folder: {class_dir}")
                continue

            # Normalize label name (e.g., "Eyeclose" -> "eyeclose")
            label = label_lower

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in class_dir.glob(ext):
                    samples.append((img_path, subject, label))

    print(f"[INFO] Found {len(samples)} images total.")
    return samples


def preprocess_and_save(
    img_path: Path,
    subject: str,
    label: str,
    proc_root: Path,
) -> Path:
    """
    Reads an image, converts to grayscale, resizes, and saves it.

    Output path:
        data/processed/<label>/<subject>_<origname>.png
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # BGR -> grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Ensure uint8 0–255
    resized_u8 = np.clip(resized, 0, 255).astype(np.uint8)

    # Output dir by label
    out_dir = proc_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Example filename: Sub1_175.png
    out_name = f"{subject}_{img_path.stem}.png"
    out_path = out_dir / out_name

    cv2.imwrite(str(out_path), resized_u8)

    return out_path


def main():
    PROC_ROOT.mkdir(parents=True, exist_ok=True)

    samples = collect_image_paths()
    if not samples:
        print("[ERROR] No images found. Check your folder structure.")
        return

    records = []

    for img_path, subject, label in samples:
        try:
            out_path = preprocess_and_save(img_path, subject, label, PROC_ROOT)
            records.append(
                {
                    "orig_path": str(img_path),
                    "proc_path": str(out_path),
                    "subject": subject,
                    "label": label,
                }
            )
        except Exception as e:
            print(f"[ERROR] Preprocessing failed for {img_path}: {e}")

    df = pd.DataFrame(records)
    print(f"[INFO] Successfully processed {len(df)} images.")

    # Train/val/test split (stratified by label)
    if len(df) > 0:
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df["label"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
        )

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    else:
        full_df = df
        full_df["split"] = "train"

    full_df.to_csv(LABELS_CSV, index=False)
    print(f"[INFO] Saved labels CSV to {LABELS_CSV}")

    print("\n[INFO] Class distribution by split:")
    print(full_df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
