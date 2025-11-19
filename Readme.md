Below is a **clean, professional, GitHub-friendly README.md** version of what you wrote — polished, formatted, and clear for recruiters, hackathons, and collaborators.

You can paste this directly into your repo’s `README.md`.

---

# 🚗 **Embedded Driver Drowsiness Detection (ESP32-S3 + TinyML)**

Real-time, on-device drowsiness detection using **ESP32-S3**, **TensorFlow Lite Micro**, and **TinyML-optimized CNNs**.

This project implements a fully embedded ML pipeline capable of detecting:

* **Eye Close**
* **Yawn**
* **Neutral**
* **Happy**

And additionally produces a **binary `alert vs drowsy` model** optimized for deployment on microcontrollers.

---

## 🧠 **Project Overview**

This system is built around a TinyML workflow:

1. **Collect Dataset**
   Capture face images (your own subjects) in multiple states:
   `eyeclose`, `yawn`, `neutral`, `happy`.

2. **Preprocess**
   Convert all images to **96×96 grayscale** for lightweight inference.

3. **Train Models**

   * **Baseline CNN (4-class)** – reference model (non-embedded).
   * **Tiny DS-CNN (4-class)** – optimized for MCU but multi-class.
   * **Tiny DS-CNN (Binary: alert/drowsy)** – final deployable model.

4. **Convert to TensorFlow Lite Micro**
   Generate:

   * `model.tflite`
   * `model.cc` (C-array for ESP32-S3)

5. **Deploy on ESP32-S3**
   Run real-time inference using:

   * ESP32-S3 camera
   * TensorFlow Lite Micro
   * LED/Buzzer alerts when the driver is drowsy

---

# 📁 **Repository Structure**

```
driver_drowsiness_project/
│
├── data/
│   ├── raw/                 # Original captured frames
│   ├── processed/           # 96x96 grayscale images
│
├── src/
│   ├── preprocess.py        # Dataset preprocessing pipeline
│
│   ├── models/
│   │   ├── baseline.py      # Baseline 4-class CNN
│   │   ├── dscnn.py         # Tiny DS-CNN (4-class)
│   │   ├── dscnn2.py        # Tiny DS-CNN (binary alert/drowsy)
│
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_dscnn.py
│   │   ├── train_dscnn2.py
│
│   ├── convert/
│   │   ├── convert_dscnn.py     # TFLite + TFLM conversion
│   │   ├── convert_baseline.py
│
├── esp32/
│   ├── model.cc                 # Final TFLM C array
│   ├── main.cpp / main.ino      # ESP32 real-time inference code
│   ├── camera_config.h
│   ├── tflite_micro_runtime/    # TFLM runtime for ESP32
│
├── mlflow_runs/
│   ├── run_baseline.py
│   ├── run_dscnn.py
│   ├── run_dscnn2.py
│   ├── compare_models.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ **Technologies Used**

### 📦 ML & Model Training

* TensorFlow / Keras
* TensorFlow Lite & TFLite Micro
* MLflow (experiment tracking)
* Python, NumPy, OpenCV

### 🧩 Embedded Deployment

* ESP32-S3
* esp-idf / Arduino framework
* TFLite Micro runtime
* OV2640 / ESP32-S3 camera module

---

# 🚀 **How It Works on ESP32-S3**

1. ESP32-S3 captures real-time camera frames.
2. The frame is resized & normalized to 96×96 grayscale.
3. The TinyML inferencing engine runs the DS-CNN model.
4. If `drowsy` is detected:

   * LED alert
   * Buzzer
   * Serial message
   * Optional: vibration motor

All inference happens **on-device** — no WiFi, no cloud, no external compute.

---

# 📊 **MLflow Integration**

This repository uses MLflow for:

✔ Logging training runs
✔ Comparing baseline vs DS-CNN models
✔ Tracking accuracy/loss
✔ Storing `.tflite` and `.cc` artifacts

Example:

```bash
python mlflow_runs/run_dscnn2.py
mlflow ui
```

---

# 🎯 **Current Status**

* [x] Dataset captured
* [x] Preprocessing pipeline implemented
* [x] CNN + DS-CNN models trained
* [x] Binary alert/drowsy DS-CNN optimized
* [x] TFLite + TFLM conversion
* [ ] ESP32-S3 deployment
* [ ] Real-time detection + hardware alerts
* [ ] Final testing in car environment
