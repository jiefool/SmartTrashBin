from __future__ import annotations

"""
Auto-detection service.

Monitors the ultrasonic sensor continuously. When an object is detected
within the configured distance threshold (default 0–5 inches), it triggers
the camera to capture an image and classifies the trash automatically.
"""

import threading
import time
from datetime import datetime
from typing import List, Optional

from loguru import logger

from app.config import settings
from app.models.classification import ClassificationResult
from app.sensors.ultrasonic import INCHES_TO_CM

# Maximum number of recent detections to keep in memory
MAX_HISTORY = 50


class DetectionService:
    """Watches the ultrasonic sensor and triggers classify on proximity."""

    def __init__(self) -> None:
        self._sensor = None
        self._camera = None
        self._classifier = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._enabled = True
        self._cooldown_seconds: float = settings.detection_cooldown_seconds
        self._threshold_inches: float = settings.detection_threshold_inches
        self._last_detection_time: float = 0
        self._history: List[dict] = []
        self._latest_result: Optional[ClassificationResult] = None
        self._latest_distance_inches: Optional[float] = None
        self._object_detected = False

    def configure(self, sensor, camera_service, classifier_service, actuator_service=None) -> None:
        """Inject dependencies after they are initialised."""
        self._sensor = sensor
        self._camera = camera_service
        self._classifier = classifier_service
        self._actuator_service = actuator_service

    @property
    def is_configured(self) -> bool:
        return (
            self._sensor is not None
            and self._camera is not None
            and self._classifier is not None
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def object_detected(self) -> bool:
        return self._object_detected

    @property
    def latest_distance_inches(self) -> Optional[float]:
        return self._latest_distance_inches

    @property
    def latest_result(self) -> Optional[ClassificationResult]:
        return self._latest_result

    @property
    def history(self) -> List[dict]:
        return list(self._history)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        state = "enabled" if enabled else "disabled"
        logger.info(f"Auto-detection {state}")

    def _poll_loop(self) -> None:
        """Background loop: read sensor → detect → classify."""
        logger.info(
            f"Detection loop started (threshold={self._threshold_inches}in, "
            f"cooldown={self._cooldown_seconds}s)"
        )
        while self._running:
            try:
                if not self._enabled:
                    time.sleep(0.5)
                    continue

                distance_cm = self._sensor.read_distance_cm()
                distance_inches = distance_cm / INCHES_TO_CM
                self._latest_distance_inches = round(distance_inches, 2)

                if 0 <= distance_inches <= self._threshold_inches:
                    self._object_detected = True
                    now = time.time()

                    # Cooldown: don't re-trigger too fast
                    if (now - self._last_detection_time) >= self._cooldown_seconds:
                        self._last_detection_time = now
                        logger.info(
                            f"Object detected at {distance_inches:.1f} inches – classifying..."
                        )
                        self._trigger_classify()
                else:
                    self._object_detected = False

                time.sleep(settings.detection_poll_interval)

            except Exception as exc:
                logger.error(f"Detection loop error: {exc}")
                time.sleep(1)

    def _trigger_classify(self) -> None:
        """Capture image and run classification."""
        try:
            if not self._camera.is_ready:
                logger.warning("Camera not ready, skipping classification.")
                return
            if not self._classifier.is_ready:
                logger.warning("Classifier not ready, skipping classification.")
                return

            image_bytes = self._camera.capture()
            self._camera.save_capture(image_bytes)
            result = self._classifier.classify(image_bytes)
            self._latest_result = result

            # Activate the corresponding actuator to sort the trash
            actuator_name = "none"
            if self._actuator_service:
                actuator_name = self._actuator_service.activate_for_category(
                    result.predicted_category
                )

            entry = {
                "category": result.predicted_category.value,
                "confidence": result.confidence,
                "distance_inches": self._latest_distance_inches,
                "actuator": actuator_name,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._history.insert(0, entry)
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[:MAX_HISTORY]

            logger.info(
                f"Auto-classified: {result.predicted_category.value} "
                f"({result.confidence:.1%}) → actuator: {actuator_name}"
            )
        except Exception as exc:
            logger.error(f"Auto-classify failed: {exc}")

    def start(self) -> None:
        if not self.is_configured:
            logger.warning("DetectionService not configured – skipping start.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("DetectionService started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("DetectionService stopped.")


# Singleton
detection = DetectionService()

