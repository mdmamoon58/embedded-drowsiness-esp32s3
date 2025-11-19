#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------

LABELS_CSV = Path("data/processed/labels_subjects.csv")
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_OUT = Path("models/baseline_cnn.h5")


# ---------------- DATASET HELPERS ----------------

def load_metadata():
    df = pd.read_csv(LABELS_CSV)

    # Split into train/val/test
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Class names
    class_names = sorted(df["label"].unique())
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    print("[INFO] Class names:", class_names)
    print("[INFO] Train size:", len(train_df))
    print("[INFO] Val size:", len(val_df))
    print("[INFO] Test size:", len(test_df))

    return train_df, val_df, test_df, class_names, label_to_idx


def make_dataset(df, label_to_idx, batch_size, shuffle=False):
    paths = df["proc_path"].values
    labels = df["label"].map(label_to_idx).values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        # path: string tensor
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_png(img_bytes, channels=1)  # grayscale
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)  # 0–1
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------- MODEL ----------------

def build_model(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ---------------- MAIN ----------------

def main():
    # Ensure output dir exists
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, class_names, label_to_idx = load_metadata()

    train_ds = make_dataset(train_df, label_to_idx, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(val_df, label_to_idx, BATCH_SIZE, shuffle=False)
    test_ds = make_dataset(test_df, label_to_idx, BATCH_SIZE, shuffle=False)

    model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"[RESULT] Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Classification report
    y_true = []
    y_pred = []

    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        preds_idx = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(preds_idx.tolist())

    print("\n[RESULT] Classification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=4
        )
    )

    print("[RESULT] Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save model
    model.save(MODEL_OUT)
    print(f"[INFO] Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
