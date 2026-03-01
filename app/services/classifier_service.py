from __future__ import annotations

"""
Trash classification service using TFLite Runtime.

Uses a TFLite model converted from MobileNetV2 trained on the TrashNet
dataset (6 categories: cardboard, glass, metal, paper, plastic, trash).

Works on Raspberry Pi without full TensorFlow installed.
"""

import os
from io import BytesIO
from typing import List

from loguru import logger

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
    """Loads a TFLite model and classifies trash images."""

    def __init__(self) -> None:
        self._interpreter = None
        self._input_details = None
        self._output_details = None

    def load_model(self) -> None:
        """Load the TFLite model from disk."""
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        model_path = settings.model_path
        if not os.path.exists(model_path):
            logger.warning(
                f"Trained model not found at '{model_path}'. "
                "Classification will be unavailable until you run the training script: "
                "python -m app.training.train"
            )
            return

        logger.info(f"Loading TFLite model from '{model_path}' ...")
        self._interpreter = Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("Model loaded successfully.")

    @property
    def is_ready(self) -> bool:
        return self._interpreter is not None

    def _preprocess(self, image_bytes: bytes):
        """Convert raw image bytes into a normalised tensor for MobileNetV2."""
        import numpy as np
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)  # batch dimension
        # MobileNetV2 preprocess: scale pixels from [0,255] to [-1,1]
        arr = (arr / 127.5) - 1.0
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

        self._interpreter.set_tensor(self._input_details[0]["index"], tensor)
        self._interpreter.invoke()
        predictions = self._interpreter.get_tensor(self._output_details[0]["index"])[0]

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

