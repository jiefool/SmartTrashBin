from __future__ import annotations

"""
Ultrasonic sensor module.

Supports:
  - Real HC-SR04 sensor via GPIO (Raspberry Pi)
  - Simulated sensor for development without hardware
"""

import random
import time
from abc import ABC, abstractmethod

from loguru import logger

from app.config import settings
from app.models.bin_status import BinReading

# Conversion factor: 1 inch = 2.54 cm
INCHES_TO_CM = 2.54


class BaseSensor(ABC):
    """Abstract base class for all bin sensors."""

    def __init__(self, bin_id: str, capacity_cm: float | None = None) -> None:
        self.bin_id = bin_id
        self.capacity_cm = capacity_cm or settings.bin_capacity_cm

    @abstractmethod
    def read(self) -> BinReading:
        """Return the current sensor reading."""

    def read_distance_cm(self) -> float:
        """Return only the distance in cm (convenience method)."""
        return self.read().distance_cm

    def read_distance_inches(self) -> float:
        """Return only the distance in inches."""
        return self.read_distance_cm() / INCHES_TO_CM


class GPIOUltrasonicSensor(BaseSensor):
    """
    Real HC-SR04 ultrasonic distance sensor via Raspberry Pi GPIO.

    Wiring:
        TRIG → GPIO 24 (Pin 18)
        ECHO → GPIO 23 (Pin 16) — use voltage divider (3.3V)
        VCC  → 5V (Pin 2)
        GND  → GND (Pin 6)
    """

    def __init__(
        self,
        bin_id: str,
        echo_pin: int | None = None,
        trig_pin: int | None = None,
        capacity_cm: float | None = None,
    ) -> None:
        super().__init__(bin_id, capacity_cm)
        self._echo_pin = echo_pin or settings.gpio_echo_pin
        self._trig_pin = trig_pin or settings.gpio_trig_pin
        self._sensor = None
        self._init_gpio()

    def _init_gpio(self) -> None:
        """Initialise the GPIO sensor."""
        try:
            from gpiozero import DistanceSensor

            self._sensor = DistanceSensor(
                echo=self._echo_pin,
                trigger=self._trig_pin,
                max_distance=4,  # ~4 metres max
            )
            logger.info(
                f"[{self.bin_id}] GPIO ultrasonic sensor initialised "
                f"(ECHO=GPIO{self._echo_pin}, TRIG=GPIO{self._trig_pin})"
            )
        except Exception as exc:
            logger.error(f"[{self.bin_id}] Failed to init GPIO sensor: {exc}")
            raise

    def read(self) -> BinReading:
        if self._sensor is None:
            raise RuntimeError("GPIO sensor not initialised.")

        distance_cm = round(self._sensor.distance * 100, 2)  # gpiozero returns metres

        reading = BinReading(
            bin_id=self.bin_id,
            distance_cm=distance_cm,
            capacity_cm=self.capacity_cm,
        )
        logger.debug(
            f"[{self.bin_id}] distance={distance_cm}cm "
            f"({distance_cm / INCHES_TO_CM:.1f}in) "
            f"fill={reading.fill_level_percent}%"
        )
        return reading

    def release(self) -> None:
        """Release GPIO resources."""
        if self._sensor:
            self._sensor.close()
            self._sensor = None
            logger.info(f"[{self.bin_id}] GPIO sensor released.")


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

