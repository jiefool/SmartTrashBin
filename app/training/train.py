"""
Training script for the Smart Trash Bin classifier.

Uses the local dataset at data/trashnet_remapped/ with 3 categories:
    biodegradable, non_biodegradable, hazardous

Trains a MobileNetV2 transfer-learning model and exports to TFLite.

Usage:
    python -m app.training.train
"""

from __future__ import annotations

import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "data/trashnet_remapped"
MODEL_SAVE_PATH = "data/models/trashnet_mobilenetv2.keras"
TFLITE_SAVE_PATH = "data/models/trashnet_mobilenetv2.tflite"


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

    dataset_path = DATASET_DIR

    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset not found at '{dataset_path}'")
        print("Expected 3 subfolders: biodegradable, non_biodegradable, hazardous")
        return

    # Print dataset summary
    for cat in sorted(os.listdir(dataset_path)):
        cat_path = os.path.join(dataset_path, cat)
        if os.path.isdir(cat_path):
            count = len(os.listdir(cat_path))
            print(f"  {cat}: {count} images")

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

