from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TrashCategory(str, Enum):
    """Garbage classification categories."""
    BIODEGRADABLE = "biodegradable"
    NON_BIODEGRADABLE = "non_biodegradable"
    HAZARDOUS = "hazardous"


class CategoryPrediction(BaseModel):
    """A single category prediction with confidence score."""
    category: TrashCategory
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")


class ClassificationResult(BaseModel):
    """Full classification result for an uploaded image."""
    model_config = ConfigDict(protected_namespaces=())

    predicted_category: TrashCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_predictions: List[CategoryPrediction] = Field(
        default_factory=list,
        description="Predictions for all categories sorted by confidence",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "1.0.0"

