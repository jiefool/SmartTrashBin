from __future__ import annotations

"""
Relay service – controls 3 relays via GPIO to sort trash into the
correct bin based on classification result.

Each relay uses a single GPIO pin:
    Relay 1 (biodegradable)      → GPIO 2
    Relay 2 (non_biodegradable)  → GPIO 3
    Relay 3 (hazardous)          → GPIO 4

Pin behaviour:
    HIGH → relay ON  (activate sorting mechanism)
    LOW  → relay OFF (idle)
"""

import threading
import time

from loguru import logger

from app.config import settings
from app.models.classification import TrashCategory

# Map each category to its relay and bin key
CATEGORY_RELAY_MAP = {
    TrashCategory.BIODEGRADABLE: "relay_1",
    TrashCategory.NON_BIODEGRADABLE: "relay_2",
    TrashCategory.HAZARDOUS: "relay_3",
}

RELAY_TO_BIN_MAP = {
    "relay_1": "bin_1",
    "relay_2": "bin_2",
    "relay_3": "bin_3",
}


class Relay:
    """Controls a single relay with 1 GPIO pin."""

    def __init__(self, name: str, pin: int) -> None:
        self.name = name
        self.pin_num = pin
        self._pin = None
        self._available = False
        self._init_gpio()

    def _init_gpio(self) -> None:
        try:
            from gpiozero import OutputDevice

            self._pin = OutputDevice(self.pin_num, initial_value=True, active_high=False)
            self._available = True
            logger.info(f"[{self.name}] Relay initialised (GPIO{self.pin_num})")
        except Exception as exc:
            logger.warning(f"[{self.name}] GPIO unavailable ({exc}), relay simulated.")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def on(self) -> None:
        """Turn relay ON."""
        logger.info(f"[{self.name}] Relay ON")
        if self._available:
            self._pin.on()
        else:
            logger.debug(f"[{self.name}] Simulated ON")

    def off(self) -> None:
        """Turn relay OFF."""
        logger.info(f"[{self.name}] Relay OFF")
        if self._available:
            self._pin.off()
        else:
            logger.debug(f"[{self.name}] Simulated OFF")

    def release(self) -> None:
        """Turn off and release GPIO."""
        self.off()
        if self._available:
            self._pin.close()
            self._available = False
            logger.info(f"[{self.name}] GPIO released.")


class ActuatorService:
    """Manages all 3 relays and activates the correct one after classification."""

    def __init__(self) -> None:
        self._relays: dict[str, Relay] = {}
        self._on_duration: float = settings.relay_on_seconds
        self._lock = threading.Lock()
        self._busy = False
        self._bin_level_service = None

    def set_bin_level_service(self, bin_level_service) -> None:
        """Inject bin level service to check if bins are full."""
        self._bin_level_service = bin_level_service

    @property
    def is_busy(self) -> bool:
        """True while a relay cycle is in progress."""
        return self._busy

    def initialise(self) -> None:
        """Create all 3 relays and ensure they start OFF."""
        self._relays = {
            "relay_1": Relay("Biodegradable", settings.relay_1_pin),
            "relay_2": Relay("NonBiodegradable", settings.relay_2_pin),
            "relay_3": Relay("Hazardous", settings.relay_3_pin),
        }
        self.all_off()
        logger.info("ActuatorService initialised (3 relays – all OFF)")

    def all_off(self) -> None:
        """Turn all relays OFF."""
        for relay in self._relays.values():
            relay.off()

    def activate_for_category(self, category: TrashCategory) -> str:
        """
        Activate the relay mapped to the given trash category.

        1. Acquire lock (blocks if another cycle is in progress)
        2. Turn all relays OFF first
        3. Turn ON the target relay for configured duration
        4. Turn all relays OFF
        5. Release lock so next classification can proceed

        Returns the relay name.
        """
        key = CATEGORY_RELAY_MAP.get(category)
        if not key or key not in self._relays:
            logger.error(f"No relay mapped for category: {category}")
            return "unknown"

        # Check if the target bin is full
        bin_key = RELAY_TO_BIN_MAP.get(key)
        if bin_key and self._bin_level_service:
            if self._bin_level_service.is_bin_full(bin_key):
                logger.warning(
                    f"BIN FULL – relay '{key}' blocked for '{category.value}'. "
                    f"Cannot sort into full bin."
                )
                return "blocked"

        with self._lock:
            self._busy = True
            try:
                relay = self._relays[key]
                logger.info(f"Sorting '{category.value}' → [{relay.name}]")

                # Step 1: make sure all relays are OFF
                self.all_off()
                time.sleep(0.05)

                # Step 2: turn ON the target relay
                relay.on()
                logger.info(f"[{relay.name}] ON – waiting {self._on_duration}s ...")
                time.sleep(self._on_duration)

                # Step 3: turn all relays OFF
                self.all_off()
                time.sleep(0.05)

                logger.info(f"[{relay.name}] Sort complete – all relays OFF.")
                return relay.name
            finally:
                self._busy = False

    def get_status(self) -> dict:
        """Return status of all relays."""
        return {
            "busy": self._busy,
            "relays": {
                name: {
                    "name": relay.name,
                    "pin": relay.pin_num,
                    "available": relay.is_available,
                }
                for name, relay in self._relays.items()
            },
        }

    def release(self) -> None:
        """Turn all OFF and release GPIO resources."""
        self.all_off()
        for relay in self._relays.values():
            relay.release()
        logger.info("ActuatorService released.")


# Singleton
actuators = ActuatorService()

