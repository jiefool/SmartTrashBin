"""Unit tests for BinReading model."""

import pytest
from app.models.bin_status import BinReading, FillStatus


def make_reading(distance_cm: float, capacity_cm: float = 100.0) -> BinReading:
    return BinReading(bin_id="TEST", distance_cm=distance_cm, capacity_cm=capacity_cm)


def test_empty_bin():
    r = make_reading(distance_cm=100.0)
    assert r.fill_level_percent == 0.0
    assert r.fill_status == FillStatus.EMPTY


def test_full_bin():
    r = make_reading(distance_cm=0.0)
    assert r.fill_level_percent == 100.0
    assert r.fill_status == FillStatus.FULL


def test_medium_fill():
    r = make_reading(distance_cm=50.0)
    assert r.fill_level_percent == 50.0
    assert r.fill_status == FillStatus.MEDIUM


def test_high_fill():
    r = make_reading(distance_cm=15.0)
    assert r.fill_level_percent == 85.0
    assert r.fill_status == FillStatus.HIGH


def test_fill_clamped_above_capacity():
    """Distance below 0 should be clamped to 100% fill."""
    r = make_reading(distance_cm=-5.0)
    assert r.fill_level_percent == 100.0


def test_fill_clamped_above_bin_depth():
    """Distance greater than capacity should be clamped to 0% fill."""
    r = make_reading(distance_cm=150.0)
    assert r.fill_level_percent == 0.0

