import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.logger import logging
# Structured logger
logger = logging.getLogger("train_dscnn_2")
logger.info(f"Project Root: {ROOT}")
# Import the correct DSCNN2 model
from src.models.dscnn2 import create_dscnn_model

# Paths
DATA_DIR = Path("data/processed")
LABELS_CSV = DATA_DIR / "labels_subjects.csv"
MODEL_OUT = Path("models/dscnn_2")


# ---------------------------------------------------------
# Load Metadata
# ---------------------------------------------------------
def load_metadata():
    logger.info("Loading labels_subjects.csv ...")
    if not LABELS_CSV.exists():
        logger.error(f"Labels CSV not found at {LABELS_CSV}")
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)

    # Validate required columns
    required_cols = {"proc_path", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"labels_subjects.csv missing columns: {sorted(missing)}")
        raise ValueError(f"labels_subjects.csv missing columns: {sorted(missing)}")

    # Map multiclass labels to binary
    label_map = {
        "neutral": "alert",
        "happy": "alert",
        "eyeclose": "drowsy",
        "yawn": "drowsy",
    }
    df["binary_label"] = df["label"].map(label_map)
    if df["binary_label"].isna().any():
        bad = sorted(df.loc[df["binary_label"].isna(), "label"].unique())
        logger.error(f"Unknown labels encountered for binary mapping: {bad}")
        raise ValueError(f"Unknown labels for binary mapping: {bad}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    logger.info(f"Train samples (orig): {len(train_df)}")
    logger.info(f"Val samples (orig):   {len(val_df)}")
    logger.info(f"Test samples (orig):  {len(test_df)}")

    return train_df, val_df, test_df

# ---------------------------------------------------------
# Load sample from file path
# ---------------------------------------------------------
def load_npy(file_path):
    arr = np.load(file_path)
    return arr.astype(np.float32)

# ---------------------------------------------------------
# Create TF dataset
# ---------------------------------------------------------
def create_dataset(df, label_to_idx, batch_size=32, shuffle=True):
    # Filter out any rows with missing paths
    total_before = len(df)
    exists_mask = df["proc_path"].apply(lambda p: os.path.exists(p))
    if not bool(exists_mask.all()):
        missing = total_before - int(exists_mask.sum())
        logger.warning(f"Filtering out {missing} samples with missing files from {total_before}")
        df = df[exists_mask].reset_index(drop=True)

    paths = df["proc_path"].astype(str).values
    labels = df["binary_label"].map(label_to_idx).values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_png(img_bytes, channels=1)
        img = tf.image.resize(img, (96, 96))
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------
# Main training
# ---------------------------------------------------------
def main():
    logger.info("Starting DSCNN training ...")

    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # Load metadata and build binary datasets
    train_df, val_df, test_df = load_metadata()

    class_names = ["alert", "drowsy"]
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    train_ds = create_dataset(train_df, label_to_idx)
    val_ds = create_dataset(val_df, label_to_idx, shuffle=False)
    test_ds = create_dataset(test_df, label_to_idx, shuffle=False)

    input_shape = (96, 96, 1)
    logger.info(f"Input shape: {input_shape}")

    # Create DSCNN model
    model = create_dscnn_model(input_shape, num_classes=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Log model summary via logger
    model.summary(print_fn=lambda line: logger.info(line))

    # Train
    class LogMetrics(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            loss = logs.get("loss", float('nan'))
            acc = logs.get("accuracy", float('nan'))
            vloss = logs.get("val_loss", float('nan'))
            vacc = logs.get("val_accuracy", float('nan'))
            logger.info(
                f"Epoch {epoch+1}: loss={loss:.4f} acc={acc:.4f} val_loss={vloss:.4f} val_acc={vacc:.4f}"
            )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT / "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True
        ),
        LogMetrics(),
    ]

    logger.info("Training starting ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks
    )

    # Save final model
    model.save(MODEL_OUT / "final_model.keras")
    logger.info("Training completed!")
    logger.info(f"Model saved at: {MODEL_OUT}")

    # Evaluate
    logger.info("Evaluating on test set ...")
    test_loss, test_acc = model.evaluate(test_ds)
    logger.info(f"Test loss={test_loss:.4f}, Test accuracy={test_acc * 100:.2f}%")

# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Log full stacktrace and exit non-zero
        logger.exception("Unhandled exception in train_dscnn_2")
        sys.exit(1)
