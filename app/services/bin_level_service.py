from __future__ import annotations

"""
Bin level monitoring service.

Uses 3 ultrasonic sensors (one per bin) to measure fill level.
When a bin's fill level reaches the threshold (≤12 inches from sensor),
the corresponding LED is turned ON and the relay is blocked.

    Sensor 1 (biodegradable)      → Trig GPIO 10, Echo GPIO 9
    Sensor 2 (non_biodegradable)  → Trig GPIO 5,  Echo GPIO 11
    Sensor 3 (hazardous)          → Trig GPIO 6,  Echo GPIO 13
"""

import threading
import time
from typing import Optional

from loguru import logger

from app.config import settings
from app.sensors.ultrasonic import INCHES_TO_CM


class BinSensor:
    """A single ultrasonic sensor for measuring bin fill level."""

    MIN_VALID_CM = 2.0
    NO_OBJECT_CM = 400.0
    SAMPLE_COUNT = 3

    def __init__(self, name: str, trig_pin: int, echo_pin: int) -> None:
        self.name = name
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self._sensor = None
        self._available = False
        self._init_gpio()

    def _init_gpio(self) -> None:
        try:
            from gpiozero import DistanceSensor

            self._sensor = DistanceSensor(
                echo=self.echo_pin,
                trigger=self.trig_pin,
                max_distance=4,
            )
            time.sleep(0.3)
            self._available = True
            logger.info(
                f"[{self.name}] Bin sensor initialised "
                f"(Trig=GPIO{self.trig_pin}, Echo=GPIO{self.echo_pin})"
            )
        except Exception as exc:
            logger.warning(f"[{self.name}] Bin sensor unavailable ({exc}), simulated.")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def read_distance_inches(self) -> float:
        """Read filtered distance in inches. Returns large value if no object."""
        if not self._available:
            return 999.0

        samples = []
        for _ in range(self.SAMPLE_COUNT):
            raw_cm = self._sensor.distance * 100
            if raw_cm >= self.MIN_VALID_CM:
                samples.append(raw_cm)
            time.sleep(0.03)

        if not samples:
            return self.NO_OBJECT_CM / INCHES_TO_CM

        samples.sort()
        median_cm = samples[len(samples) // 2]
        return median_cm / INCHES_TO_CM

    def release(self) -> None:
        if self._sensor:
            self._sensor.close()
            self._sensor = None


class BinLevelService:
    """Monitors 3 bin sensors, updates LEDs, and reports which bins are full."""

    def __init__(self) -> None:
        self._sensors: dict[str, BinSensor] = {}
        self._full_threshold_inches: float = settings.bin_full_threshold_inches
        self._distances: dict[str, float] = {}
        self._bin_full: dict[str, bool] = {"bin_1": False, "bin_2": False, "bin_3": False}
        self._led_service = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def configure(self, led_service) -> None:
        self._led_service = led_service

    def initialise(self) -> None:
        self._sensors = {
            "bin_1": BinSensor("Biodegradable", settings.bin_sensor_1_trig, settings.bin_sensor_1_echo),
            "bin_2": BinSensor("NonBiodegradable", settings.bin_sensor_2_trig, settings.bin_sensor_2_echo),
            "bin_3": BinSensor("Hazardous", settings.bin_sensor_3_trig, settings.bin_sensor_3_echo),
        }
        logger.info(f"BinLevelService initialised (threshold={self._full_threshold_inches}in)")

    def is_bin_full(self, key: str) -> bool:
        return self._bin_full.get(key, False)

    def get_distances(self) -> dict[str, float]:
        return dict(self._distances)

    def get_status(self) -> dict:
        return {
            key: {
                "name": sensor.name,
                "distance_inches": self._distances.get(key),
                "is_full": self._bin_full.get(key, False),
                "threshold_inches": self._full_threshold_inches,
                "available": sensor.is_available,
            }
            for key, sensor in self._sensors.items()
        }

    def _poll_loop(self) -> None:
        led_map = {"bin_1": "led_1", "bin_2": "led_2", "bin_3": "led_3"}

        while self._running:
            for key, sensor in self._sensors.items():
                try:
                    dist = sensor.read_distance_inches()
                    self._distances[key] = round(dist, 2)
                    is_full = dist <= self._full_threshold_inches
                    was_full = self._bin_full.get(key, False)
                    self._bin_full[key] = is_full

                    if self._led_service:
                        self._led_service.set_led(led_map[key], is_full)

                    if is_full and not was_full:
                        logger.warning(f"[{sensor.name}] BIN FULL ({dist:.1f}in)")
                    elif not is_full and was_full:
                        logger.info(f"[{sensor.name}] Bin has space ({dist:.1f}in)")
                except Exception as exc:
                    logger.error(f"[{sensor.name}] Read error: {exc}")

            time.sleep(settings.bin_level_poll_interval)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("BinLevelService started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("BinLevelService stopped.")

    def release(self) -> None:
        self.stop()
        for sensor in self._sensors.values():
            sensor.release()


# Singleton
bin_levels = BinLevelService()
