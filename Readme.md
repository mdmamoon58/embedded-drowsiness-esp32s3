# Embedded Drowsiness Detection (ESP32-S3 + TinyML)

This project is an embedded ML pipeline for **driver drowsiness detection**:

- You collect face images for a few subjects in different states:
  - `eyeclose`, `yawn`, `neutral`, `happy`
- You preprocess them to **96×96 grayscale**.
- You train:
  1. A **4-class baseline CNN** (reference, not embedded-friendly).
  2. A **tiny 4-class DS-CNN**.
  3. A **tiny binary DS-CNN** that classifies **`alert` vs `drowsy`** (this is the main model for ESP32-S3).

The ESP32-S3 integration (TFLite Micro + camera + LED) is the next step on top of this repo.

---

## Repository Layout (what matters)

```text
embedded-drowsiness-esp32s3/
├─ data/
│  ├─ raw/                 # raw subject images (you put these here)
│  │  ├─ Sub1/
│  │  │  ├─ Eyeclose/
│  │  │  ├─ Yawn/
│  │  │  ├─ Happy/
│  │  │  └─ Neutral/
│  │  ├─ Sub2/ ...
│  │  └─ Sub3/ ...
│  └─ processed/
│     ├─ eyeclose/         # preprocessed 96x96 grayscale images
│     ├─ yawn/
│     ├─ happy/
│     ├─ neutral/
│     └─ labels_subjects.csv  # metadata + train/val/test split
├─ models/
│  ├─ baseline_cnn.h5        # 4-class large CNN
│  ├─ dscnn_tiny_plus.h5     # 4-class DS-CNN (tiny)
│  └─ dscnn_binary.h5        # binary DS-CNN (alert vs drowsy)
├─ tflite/                   # (optional) exported TFLite models
├─ esp32/                    # placeholder for ESP32 firmware
├─ preprocess_subjects.py    # raw → processed images + CSV
├─ train_baseline_cnn.py     # 4-class baseline CNN
├─ train_dscnn.py            # 4-class tiny DS-CNN
├─ train_dscnn_binary.py     # binary DS-CNN (main deployment model)
├─ analyze_binary_thresholds.py  # threshold sweep for binary model
├─ convert_dscnn_to_tflite.py    # (optional) DS-CNN → TFLite
├─ .gitignore
└─ README.md
