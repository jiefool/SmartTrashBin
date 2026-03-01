"""Unit tests for AlertService."""

import pytest
from app.models.bin_status import BinReading
from app.services.alert_service import AlertService


def make_reading(fill_percent: float, bin_id: str = "BIN-X", capacity_cm: float = 100.0) -> BinReading:
    distance_cm = capacity_cm - (fill_percent / 100.0) * capacity_cm
    return BinReading(bin_id=bin_id, distance_cm=max(0.0, distance_cm), capacity_cm=capacity_cm)


def test_no_alert_below_threshold():
    svc = AlertService(threshold_percent=80.0)
    reading = make_reading(50.0)
    assert svc.evaluate(reading) is None


def test_alert_at_threshold():
    svc = AlertService(threshold_percent=80.0)
    reading = make_reading(80.0)
    alert = svc.evaluate(reading)
    assert alert is not None
    assert alert.fill_level_percent >= 80.0


def test_alert_fires_only_once_per_cycle():
    """Alert should not fire a second time if bin is still full."""
    svc = AlertService(threshold_percent=80.0)
    reading = make_reading(90.0)
    first = svc.evaluate(reading)
    second = svc.evaluate(reading)
    assert first is not None
    assert second is None


def test_alert_resets_after_emptying():
    """After the bin is emptied, it should be able to alert again."""
    svc = AlertService(threshold_percent=80.0)
    full = make_reading(90.0)
    empty = make_reading(10.0)

    svc.evaluate(full)         # alert fires
    svc.evaluate(empty)        # reset
    alert = svc.evaluate(full) # should alert again
    assert alert is not None


def test_alert_message_contains_bin_id():
    svc = AlertService(threshold_percent=80.0)
    reading = make_reading(85.0, bin_id="BIN-Z")
    alert = svc.evaluate(reading)
    assert "BIN-Z" in alert.message

