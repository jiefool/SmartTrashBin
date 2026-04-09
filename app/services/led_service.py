from __future__ import annotations

"""
LED service – controls 3 indicator LEDs via GPIO.

Each LED indicates that the corresponding bin is FULL:
    LED 1 (biodegradable)      → GPIO 17
    LED 2 (non_biodegradable)  → GPIO 27
    LED 3 (hazardous)          → GPIO 22

LED ON  = bin full (relay blocked)
LED OFF = bin has space (relay allowed)
"""

from loguru import logger

from app.config import settings


class LED:
    """Controls a single LED with 1 GPIO pin."""

    def __init__(self, name: str, pin: int) -> None:
        self.name = name
        self.pin_num = pin
        self._pin = None
        self._available = False
        self._is_on = False
        self._init_gpio()

    def _init_gpio(self) -> None:
        try:
            from gpiozero import OutputDevice

            self._pin = OutputDevice(self.pin_num, initial_value=False)
            self._available = True
            logger.info(f"[{self.name}] LED initialised (GPIO{self.pin_num})")
        except Exception as exc:
            logger.warning(f"[{self.name}] GPIO unavailable ({exc}), LED simulated.")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_on(self) -> bool:
        return self._is_on

    def on(self) -> None:
        """Turn LED ON (bin full)."""
        self._is_on = True
        if self._available:
            self._pin.on()
        logger.info(f"[{self.name}] LED ON – bin full")

    def off(self) -> None:
        """Turn LED OFF (bin has space)."""
        self._is_on = False
        if self._available:
            self._pin.off()
        logger.debug(f"[{self.name}] LED OFF")

    def release(self) -> None:
        """Turn off and release GPIO."""
        self.off()
        if self._available:
            self._pin.close()
            self._available = False


class LEDService:
    """Manages all 3 bin-full indicator LEDs."""

    def __init__(self) -> None:
        self._leds: dict[str, LED] = {}

    def initialise(self) -> None:
        """Create all 3 LEDs."""
        self._leds = {
            "led_1": LED("Biodegradable", settings.led_1_pin),
            "led_2": LED("NonBiodegradable", settings.led_2_pin),
            "led_3": LED("Hazardous", settings.led_3_pin),
        }
        self.all_off()
        logger.info("LEDService initialised (3 LEDs – all OFF)")

    def set_led(self, key: str, state: bool) -> None:
        """Turn a specific LED on or off by key (led_1, led_2, led_3)."""
        led = self._leds.get(key)
        if led:
            led.on() if state else led.off()

    def is_on(self, key: str) -> bool:
        """Check if a specific LED is on."""
        led = self._leds.get(key)
        return led.is_on if led else False

    def all_off(self) -> None:
        """Turn all LEDs OFF."""
        for led in self._leds.values():
            led.off()

    def get_status(self) -> dict:
        """Return status of all LEDs."""
        return {
            name: {
                "name": led.name,
                "pin": led.pin_num,
                "is_on": led.is_on,
                "available": led.is_available,
            }
            for name, led in self._leds.items()
        }

    def release(self) -> None:
        """Release all LED GPIO resources."""
        for led in self._leds.values():
            led.release()
        logger.info("LEDService released.")


# Singleton
leds = LEDService()
