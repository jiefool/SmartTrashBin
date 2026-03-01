from __future__ import annotations

"""
Monitor service – periodically polls all registered sensors and triggers alerts.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from app.config import settings
from app.models.bin_status import BinReading
from app.sensors.ultrasonic import BaseSensor
from app.services.alert_service import AlertService


class MonitorService:
    def __init__(self) -> None:
        self._sensors: list[BaseSensor] = []
        self._alert_service = AlertService()
        self._scheduler = BackgroundScheduler()
        self._latest_readings: dict[str, BinReading] = {}

    def register_sensor(self, sensor: BaseSensor) -> None:
        self._sensors.append(sensor)
        logger.info(f"Registered sensor for bin '{sensor.bin_id}'")

    def get_latest_readings(self) -> dict[str, BinReading]:
        return dict(self._latest_readings)

    def _poll(self) -> None:
        for sensor in self._sensors:
            try:
                reading = sensor.read()
                self._latest_readings[sensor.bin_id] = reading
                self._alert_service.evaluate(reading)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error reading sensor for bin '{sensor.bin_id}': {exc}")

    def start(self) -> None:
        interval = settings.poll_interval_seconds
        self._scheduler.add_job(self._poll, "interval", seconds=interval, id="poll_sensors")
        self._scheduler.start()
        logger.info(f"MonitorService started – polling every {interval}s")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("MonitorService stopped")

