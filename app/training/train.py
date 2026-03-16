"""
Training script for the Smart Trash Bin classifier.

Downloads the TrashNet dataset from GitHub, remaps the 6 original classes
into 3 categories (biodegradable, non_biodegradable, hazardous), then
trains a MobileNetV2 transfer-learning model.

Mapping:
    biodegradable       ← cardboard, paper
    non_biodegradable   ← glass, metal, plastic
    hazardous           ← trash

Usage:
    python -m app.training.train
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
DATASET_DIR = "data/trashnet"
REMAPPED_DIR = "data/trashnet_remapped"
MODEL_SAVE_PATH = "data/models/trashnet_mobilenetv2.keras"
TFLITE_SAVE_PATH = "data/models/trashnet_mobilenetv2.tflite"

# TrashNet → 3-category mapping
CATEGORY_MAP = {
    "cardboard": "biodegradable",
    "paper":     "biodegradable",
    "glass":     "non_biodegradable",
    "metal":     "non_biodegradable",
    "plastic":   "non_biodegradable",
    "trash":     "hazardous",
}


def download_dataset() -> str:
    """Download and extract TrashNet dataset if not already present."""
    extract_dir = DATASET_DIR
    zip_path = os.path.join("data", "dataset-resized.zip")

    if os.path.isdir(os.path.join(extract_dir, "dataset-resized")):
        print(f"Dataset already exists at '{extract_dir}/dataset-resized'. Skipping download.")
        return os.path.join(extract_dir, "dataset-resized")

    os.makedirs(extract_dir, exist_ok=True)

    print(f"Downloading TrashNet dataset from {DATASET_URL} ...")
    subprocess.check_call(
        ["curl", "-L", "-o", zip_path, DATASET_URL],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print("Extracting dataset ...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    os.remove(zip_path)
    print(f"Dataset extracted to '{extract_dir}/dataset-resized'")
    return os.path.join(extract_dir, "dataset-resized")


def remap_dataset(original_path: str) -> str:
    """
    Remap the 6-class TrashNet dataset into 3 categories by creating
    symlinks (or copies) in a new directory structure.
    """
    if os.path.isdir(REMAPPED_DIR) and len(os.listdir(REMAPPED_DIR)) == 3:
        print(f"Remapped dataset already exists at '{REMAPPED_DIR}'. Skipping.")
        return REMAPPED_DIR

    print("\nRemapping dataset to 3 categories ...")
    # Clean slate
    if os.path.exists(REMAPPED_DIR):
        shutil.rmtree(REMAPPED_DIR)

    for new_cat in set(CATEGORY_MAP.values()):
        os.makedirs(os.path.join(REMAPPED_DIR, new_cat), exist_ok=True)

    count = 0
    for old_cat, new_cat in CATEGORY_MAP.items():
        src_dir = os.path.join(original_path, old_cat)
        if not os.path.isdir(src_dir):
            print(f"  ⚠ Skipping missing folder: {src_dir}")
            continue
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(REMAPPED_DIR, new_cat, f"{old_cat}_{fname}")
            shutil.copy2(src, dst)
            count += 1

    print(f"Remapped {count} images into {REMAPPED_DIR}")
    for cat in sorted(set(CATEGORY_MAP.values())):
        n = len(os.listdir(os.path.join(REMAPPED_DIR, cat)))
        print(f"  {cat}: {n} images")
    return REMAPPED_DIR


def build_model(num_classes: int = 3):
    """Build a MobileNetV2-based transfer learning model."""
    from keras import layers, models, optimizers
    from keras.applications import MobileNetV2

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train() -> None:
    """Full training pipeline."""
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.mobilenet_v2 import preprocess_input

    original_path = download_dataset()
    dataset_path = remap_dataset(original_path)

    print(f"\nLoading images from '{dataset_path}' ...")

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    print(f"\nClasses found: {train_gen.class_indices}")
    num_classes = len(train_gen.class_indices)
    assert num_classes == 3, f"Expected 3 classes, got {num_classes}"

    model = build_model(num_classes)
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs ...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
    )

    # Evaluate
    loss, acc = model.evaluate(val_gen)
    print(f"\nValidation loss: {loss:.4f}  |  Validation accuracy: {acc:.4f}")

    # Save Keras model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nKeras model saved to '{MODEL_SAVE_PATH}'")

    # Convert to TFLite for Raspberry Pi deployment
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_SAVE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to '{TFLITE_SAVE_PATH}'")
    print("\nCopy the .tflite file to your Raspberry Pi and restart the app.")
    print("You can now POST images to /classify")


if __name__ == "__main__":
    train()

