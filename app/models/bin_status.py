from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FillStatus(str, Enum):
    EMPTY = "EMPTY"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    FULL = "FULL"


class BinReading(BaseModel):
    """A single ultrasonic sensor reading from a bin."""

    bin_id: str
    distance_cm: float = Field(..., ge=0, description="Distance from sensor to waste surface (cm)")
    capacity_cm: float = Field(..., gt=0, description="Total depth of the bin (cm)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def fill_level_percent(self) -> float:
        """Percentage of bin that is filled (0–100)."""
        filled = self.capacity_cm - self.distance_cm
        return round(max(0.0, min(100.0, (filled / self.capacity_cm) * 100)), 2)

    @property
    def fill_status(self) -> FillStatus:
        level = self.fill_level_percent
        if level < 10:
            return FillStatus.EMPTY
        elif level < 40:
            return FillStatus.LOW
        elif level < 70:
            return FillStatus.MEDIUM
        elif level < 90:
            return FillStatus.HIGH
        return FillStatus.FULL


class BinAlert(BaseModel):
    """Alert emitted when a bin exceeds the configured threshold."""

    bin_id: str
    fill_level_percent: float
    fill_status: FillStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

