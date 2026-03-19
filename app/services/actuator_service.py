from __future__ import annotations

"""
Actuator service – controls 3 linear actuators via GPIO to sort trash
into the correct bin based on classification result.

Each actuator has 2 control pins (extend / retract):
    Actuator 1 (biodegradable)      → GPIO 2 & 3
    Actuator 2 (non_biodegradable)  → GPIO 4 & 5
    Actuator 3 (hazardous)          → GPIO 6 & 7

Pin behaviour:
    pin_a HIGH, pin_b LOW  → extend  (push trash into bin)
    pin_a LOW,  pin_b HIGH → retract (return to neutral)
    both LOW               → stop / idle
"""

import threading
import time
from typing import Optional

from loguru import logger

from app.config import settings
from app.models.classification import TrashCategory

# Map each category to its actuator config key
CATEGORY_ACTUATOR_MAP = {
    TrashCategory.BIODEGRADABLE: "actuator_1",
    TrashCategory.NON_BIODEGRADABLE: "actuator_2",
    TrashCategory.HAZARDOUS: "actuator_3",
}


class Actuator:
    """Controls a single linear actuator with 2 GPIO pins."""

    def __init__(self, name: str, pin_a: int, pin_b: int) -> None:
        self.name = name
        self.pin_a_num = pin_a
        self.pin_b_num = pin_b
        self._pin_a = None
        self._pin_b = None
        self._available = False
        self._init_gpio()

    def _init_gpio(self) -> None:
        try:
            from gpiozero import OutputDevice

            self._pin_a = OutputDevice(self.pin_a_num, initial_value=True)
            self._pin_b = OutputDevice(self.pin_b_num, initial_value=True)
            self._available = True
            logger.info(
                f"[{self.name}] Actuator initialised "
                f"(pin_a=GPIO{self.pin_a_num}, pin_b=GPIO{self.pin_b_num})"
            )
        except Exception as exc:
            logger.warning(f"[{self.name}] GPIO unavailable ({exc}), actuator simulated.")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def extend(self) -> None:
        """Extend actuator (push trash)."""
        logger.info(f"[{self.name}] Extending...")
        if self._available:
            self._pin_a.off()
            self._pin_b.on()
        else:
            logger.debug(f"[{self.name}] Simulated extend")

    def retract(self) -> None:
        """Retract actuator (return to neutral)."""
        logger.info(f"[{self.name}] Retracting...")
        if self._available:
            self._pin_a.on()
            self._pin_b.off()
        else:
            logger.debug(f"[{self.name}] Simulated retract")

    def stop(self) -> None:
        """Stop actuator (both pins HIGH)."""
        if self._available:
            self._pin_a.on()
            self._pin_b.on()
        logger.debug(f"[{self.name}] Stopped")

    def release(self) -> None:
        """Release GPIO resources."""
        self.stop()
        if self._available:
            self._pin_a.close()
            self._pin_b.close()
            self._available = False
            logger.info(f"[{self.name}] GPIO released.")


class ActuatorService:
    """Manages all 3 actuators and activates the correct one after classification."""

    def __init__(self) -> None:
        self._actuators: dict[str, Actuator] = {}
        self._extend_duration: float = settings.actuator_extend_seconds
        self._lock = threading.Lock()
        self._busy = False

    @property
    def is_busy(self) -> bool:
        """True while any actuator is mid-cycle (extend/retract)."""
        return self._busy

    def initialise(self) -> None:
        """Create all 3 actuators and retract them all to start position."""
        self._actuators = {
            "actuator_1": Actuator("Biodegradable", settings.actuator_1_pin_a, settings.actuator_1_pin_b),
            "actuator_2": Actuator("NonBiodegradable", settings.actuator_2_pin_a, settings.actuator_2_pin_b),
            "actuator_3": Actuator("Hazardous", settings.actuator_3_pin_a, settings.actuator_3_pin_b),
        }
        # Ensure all actuators start in retracted position
        self.retract_all()
        logger.info("ActuatorService initialised (3 actuators – all retracted)")

    def retract_all(self) -> None:
        """Retract all actuators to their home position."""
        for act in self._actuators.values():
            act.retract()
        logger.info("All actuators retracted.")

    def activate_for_category(self, category: TrashCategory) -> str:
        """
        Activate the actuator mapped to the given trash category.

        1. Acquire lock (blocks if another cycle is in progress)
        2. Retract all actuators first
        3. Extend the target actuator for 10 seconds
        4. Retract all actuators
        5. Release lock so next classification can proceed

        Returns the actuator name.
        """
        key = CATEGORY_ACTUATOR_MAP.get(category)
        if not key or key not in self._actuators:
            logger.error(f"No actuator mapped for category: {category}")
            return "unknown"

        with self._lock:
            self._busy = True
            try:
                actuator = self._actuators[key]
                logger.info(f"Sorting '{category.value}' → [{actuator.name}]")

                # Step 1: make sure all actuators are retracted
                self.retract_all()
                time.sleep(0.5)  # brief settle time

                # Step 2: extend the target actuator
                actuator.extend()
                logger.info(f"[{actuator.name}] Extended – waiting {self._extend_duration}s ...")
                time.sleep(self._extend_duration)

                # Step 3: retract all actuators
                self.retract_all()
                time.sleep(0.5)  # settle time

                logger.info(f"[{actuator.name}] Sort complete – all actuators retracted.")
                return actuator.name
            finally:
                self._busy = False

    def get_status(self) -> dict:
        """Return status of all actuators."""
        return {
            "busy": self._busy,
            "actuators": {
                name: {
                    "name": act.name,
                    "pin_a": act.pin_a_num,
                    "pin_b": act.pin_b_num,
                    "available": act.is_available,
                }
                for name, act in self._actuators.items()
            },
        }

    def release(self) -> None:
        """Retract all and release GPIO resources."""
        self.retract_all()
        for act in self._actuators.values():
            act.release()
        logger.info("ActuatorService released.")


# Singleton
actuators = ActuatorService()

