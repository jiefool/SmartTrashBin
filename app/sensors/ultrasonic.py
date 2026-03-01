from __future__ import annotations

"""
Ultrasonic sensor module.

In production this would interface with real hardware (e.g. HC-SR04 via GPIO).
For now it provides a simulator so the application can run without physical sensors.
"""

import random
from abc import ABC, abstractmethod

from loguru import logger

from app.config import settings
from app.models.bin_status import BinReading


class BaseSensor(ABC):
    """Abstract base class for all bin sensors."""

    def __init__(self, bin_id: str, capacity_cm: float | None = None) -> None:
        self.bin_id = bin_id
        self.capacity_cm = capacity_cm or settings.bin_capacity_cm

    @abstractmethod
    def read(self) -> BinReading:
        """Return the current sensor reading."""


class SimulatedUltrasonicSensor(BaseSensor):
    """
    Simulates an HC-SR04 ultrasonic distance sensor.

    The sensor measures the distance (in cm) from the top of the bin
    to the surface of the waste.  As the bin fills up the distance decreases.
    """

    def __init__(
        self,
        bin_id: str,
        capacity_cm: float | None = None,
        initial_fill_percent: float = 0.0,
        noise_cm: float = 1.5,
    ) -> None:
        super().__init__(bin_id, capacity_cm)
        self._fill_percent = max(0.0, min(100.0, initial_fill_percent))
        self._noise_cm = noise_cm

    def _simulate_fill_increase(self) -> None:
        """Randomly increase the fill level to mimic waste being added."""
        delta = random.uniform(0.0, 3.0)
        self._fill_percent = min(100.0, self._fill_percent + delta)

    def read(self) -> BinReading:
        self._simulate_fill_increase()
        filled_cm = (self._fill_percent / 100.0) * self.capacity_cm
        raw_distance = self.capacity_cm - filled_cm
        # Add sensor noise
        noise = random.uniform(-self._noise_cm, self._noise_cm)
        distance_cm = max(0.0, raw_distance + noise)

        reading = BinReading(
            bin_id=self.bin_id,
            distance_cm=round(distance_cm, 2),
            capacity_cm=self.capacity_cm,
        )
        logger.debug(
            f"[{self.bin_id}] fill={reading.fill_level_percent}% "
            f"distance={reading.distance_cm}cm status={reading.fill_status}"
        )
        return reading

