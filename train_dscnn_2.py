#!/usr/bin/env python3
"""
Train a binary depthwise-separable CNN (DS-CNN) for drowsiness detection.

We collapse 4 labels into 2:

  - alert  = {neutral, happy}
  - drowsy = {eyeclose, yawn}

This is more aligned with the actual device behavior (LED alert vs no alert)
and is much easier for a tiny model than 4-way classification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIG ----------------

LABELS_CSV = Path("data/processed/labels_subjects.csv")
IMG_SIZE = (96, 96)  # (width, height)
BATCH_SIZE = 32
EPOCHS = 40
MODEL_OUT = Path("models/dscnn_binary.h5")


# ---------------- DATASET HELPERS ----------------

def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map original 4 labels -> 2 labels:
        alert  = neutral, happy
        drowsy = eyeclose, yawn
    """
    def _map(lbl: str) -> str:
        lbl = lbl.lower()
        if lbl in ("neutral", "happy"):
            return "alert"
        elif lbl in ("eyeclose", "yawn"):
            return "drowsy"
        else:
            raise ValueError(f"Unknown label: {lbl}")

    df = df.copy()
    df["binary_label"] = df["label"].apply(_map)
    return df


def load_metadata():
    df = pd.read_csv(LABELS_CSV)
    df = add_binary_label(df)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    class_names = sorted(train_df["binary_label"].unique())  # ['alert', 'drowsy']
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    print("[INFO] Binary class names:", class_names)
    print("[INFO] Train size:", len(train_df))
    print("[INFO] Val size:", len(val_df))
    print("[INFO] Test size:", len(test_df))

    # Also show per-class distribution
    print("\n[INFO] Train distribution (binary):")
    print(train_df["binary_label"].value_counts())

    return train_df, val_df, test_df, class_names, label_to_idx


def make_dataset(df, label_to_idx, batch_size, shuffle=False, augment=False):
    """
    Create a tf.data pipeline that:
    - loads PNGs from disk
    - converts to grayscale float32 in [0,1]
    - applies optional augmentation (train only)
    """
    paths = df["proc_path"].values
    labels = df["binary_label"].map(label_to_idx).values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_png(img_bytes, channels=1)  # grayscale
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)  # 0–1

        if augment:
            # Horizontal flip
            img = tf.image.random_flip_left_right(img)
            # Brightness & contrast jitter
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            # Small random crop + resize back (translation/zoom)
            pad_h = IMG_SIZE[1] + 4
            pad_w = IMG_SIZE[0] + 4
            img = tf.image.resize_with_crop_or_pad(img, pad_h, pad_w)
            img = tf.image.random_crop(img, size=(IMG_SIZE[1], IMG_SIZE[0], 1))

        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------- MODEL (DS-CNN) ----------------

def conv_block_ds(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """Depthwise-separable conv block: SeparableConv2D + BN + ReLU."""
    x = tf.keras.layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_dscnn_binary() -> tf.keras.Model:
    """
    Binary DS-CNN: still tiny, but slightly wider than the absolute minimum.
    """
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 1))

    # Block 1
    x = conv_block_ds(inputs, 24)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)   # 48x48x24

    # Block 2
    x = conv_block_ds(x, 48)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)   # 24x24x48

    # Block 3
    x = conv_block_ds(x, 64)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)   # 12x12x64

    x = tf.keras.layers.GlobalAveragePooling2D()(x)         # 64
    x = tf.keras.layers.Dense(48, activation="relu")(x)

    # Binary output: 1 logit with sigmoid
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dscnn_binary")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ---------------- MAIN ----------------

def main():
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, class_names, label_to_idx = load_metadata()

    # Datasets (augmentation only on train)
    train_ds = make_dataset(
        train_df, label_to_idx, BATCH_SIZE, shuffle=True, augment=True
    )
    val_ds = make_dataset(
        val_df, label_to_idx, BATCH_SIZE, shuffle=False, augment=False
    )
    test_ds = make_dataset(
        test_df, label_to_idx, BATCH_SIZE, shuffle=False, augment=False
    )

    # Class weights for binary labels
    train_labels_int = train_df["binary_label"].map(label_to_idx).values
    classes = np.arange(len(class_names))  # [0,1]
    class_weight_vals = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels_int,
    )
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weight_vals)}
    print("[INFO] Binary class weights:", class_weight)

    model = build_dscnn_binary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[RESULT] Binary test loss: {test_loss:.4f}, Binary test accuracy: {test_acc:.4f}")

    # Collect predictions
    y_true, y_pred = [], []

    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)  # shape [B,1], sigmoid
        preds_bin = (preds[:, 0] >= 0.5).astype(int)  # threshold at 0.5
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(preds_bin.tolist())

    # Convert idx->name for reporting
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    print("\n[RESULT] Binary classification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=target_names, digits=4
        )
    )

    print("[RESULT] Binary confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    model.save(MODEL_OUT)
    print(f"[INFO] Saved binary DS-CNN model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
