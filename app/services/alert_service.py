from __future__ import annotations

"""
Alert service – decides when a bin needs collection and publishes alerts.
"""

from loguru import logger

from app.config import settings
from app.models.bin_status import BinAlert, BinReading


class AlertService:
    def __init__(self, threshold_percent: float | None = None) -> None:
        self.threshold = threshold_percent or settings.alert_threshold_percent
        self._alerted_bins: set[str] = set()

    def evaluate(self, reading: BinReading) -> BinAlert | None:
        """
        Return a BinAlert if the fill level exceeds the threshold,
        or None if the bin is still within acceptable limits.
        Implements a simple hysteresis: re-alerts only once per fill cycle.
        """
        level = reading.fill_level_percent

        if level >= self.threshold:
            if reading.bin_id not in self._alerted_bins:
                self._alerted_bins.add(reading.bin_id)
                alert = BinAlert(
                    bin_id=reading.bin_id,
                    fill_level_percent=level,
                    fill_status=reading.fill_status,
                    message=(
                        f"Bin '{reading.bin_id}' is {level:.1f}% full "
                        f"({reading.fill_status.value}). Collection required!"
                    ),
                )
                logger.warning(alert.message)
                return alert
        else:
            # Reset so we can alert again next fill cycle
            self._alerted_bins.discard(reading.bin_id)

        return None

