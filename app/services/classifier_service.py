from __future__ import annotations

"""
Trash classification service using TensorFlow + MobileNetV2.

Uses transfer learning on the TrashNet dataset (6 categories:
cardboard, glass, metal, paper, plastic, trash).
"""

import os
from io import BytesIO
from typing import List

import numpy as np
from loguru import logger
from PIL import Image

from app.config import settings
from app.models.classification import (
    CategoryPrediction,
    ClassificationResult,
    TrashCategory,
)

# Image dimensions expected by MobileNetV2
IMG_SIZE = (224, 224)

# Category labels in the same order used during training
CATEGORY_LABELS: List[TrashCategory] = [
    TrashCategory.CARDBOARD,
    TrashCategory.GLASS,
    TrashCategory.METAL,
    TrashCategory.PAPER,
    TrashCategory.PLASTIC,
    TrashCategory.TRASH,
]


class TrashClassifier:
    """Loads a trained MobileNetV2 model and classifies trash images."""

    def __init__(self) -> None:
        self._model = None

    def load_model(self) -> None:
        """Load the trained model from disk."""
        import tensorflow as tf

        model_path = settings.model_path
        if not os.path.exists(model_path):
            logger.warning(
                f"Trained model not found at '{model_path}'. "
                "Classification will be unavailable until you run the training script: "
                "python -m app.training.train"
            )
            return

        logger.info(f"Loading trash classification model from '{model_path}' ...")
        self._model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Convert raw image bytes into a normalised tensor for MobileNetV2."""
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)  # batch dimension
        arr = preprocess_input(arr)
        return arr

    def classify(self, image_bytes: bytes) -> ClassificationResult:
        """
        Classify a single image and return predictions for all categories.

        Raises RuntimeError if the model has not been loaded yet.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Model not loaded. Train the model first: python -m app.training.train"
            )

        tensor = self._preprocess(image_bytes)
        predictions = self._model.predict(tensor, verbose=0)[0]  # shape (6,)

        # Build per-category predictions sorted by confidence (desc)
        all_preds = sorted(
            [
                CategoryPrediction(
                    category=cat,
                    confidence=round(float(conf), 4),
                )
                for cat, conf in zip(CATEGORY_LABELS, predictions)
            ],
            key=lambda p: p.confidence,
            reverse=True,
        )

        top = all_preds[0]
        logger.info(
            f"Classification: {top.category.value} ({top.confidence:.2%})"
        )

        return ClassificationResult(
            predicted_category=top.category,
            confidence=top.confidence,
            all_predictions=all_preds,
        )


# Singleton instance
classifier = TrashClassifier()

