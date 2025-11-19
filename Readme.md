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

driver_drowsiness_project/
│
├── data/
│   ├── raw/              # original images/frames/sensors
│   ├── processed/        # normalized, resized, MFCCs, etc.
│
├── src/
│   ├── preprocess.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── dscnn.py
│   │   ├── dscnn2.py
│   │
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_dscnn.py
│   │   ├── train_dscnn2.py
│   │
│   ├── convert/
│   │   ├── convert_dscnn.py     # TFLite / TFLM .cc conversion
│   │   ├── convert_baseline.py
│
├── esp32/
│   ├── model.cc                 # final TFLM model for ESP32
│   ├── main.ino / main.cpp      # ESP32 inference code
│   ├── camera_config.h
│   ├── tflite_micro_runtime/
│
├── mlflow_runs/
│   ├── run_baseline.py
│   ├── run_dscnn.py
│   ├── run_dscnn2.py
│   ├── compare_models.py
│
├── requirements.txt
└── README.md
