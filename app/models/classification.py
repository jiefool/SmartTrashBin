from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class TrashCategory(str, Enum):
    """TrashNet dataset categories."""
    CARDBOARD = "cardboard"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    PLASTIC = "plastic"
    TRASH = "trash"


class CategoryPrediction(BaseModel):
    """A single category prediction with confidence score."""
    category: TrashCategory
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")


class ClassificationResult(BaseModel):
    """Full classification result for an uploaded image."""
    predicted_category: TrashCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_predictions: List[CategoryPrediction] = Field(
        default_factory=list,
        description="Predictions for all categories sorted by confidence",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "1.0.0"

