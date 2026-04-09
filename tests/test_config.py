"""Tests for Settings field validators."""

import pytest
from pydantic import ValidationError

from token0.config import Settings


def test_storage_mode_valid_values():
    assert Settings(storage_mode="lite").storage_mode == "lite"
    assert Settings(storage_mode="full").storage_mode == "full"


def test_storage_mode_invalid_raises():
    with pytest.raises(ValidationError):
        Settings(storage_mode="prod")


def test_text_density_threshold_valid():
    assert Settings(text_density_threshold=0.0).text_density_threshold == 0.0
    assert Settings(text_density_threshold=1.0).text_density_threshold == 1.0
    assert Settings(text_density_threshold=0.52).text_density_threshold == 0.52


def test_text_density_threshold_out_of_range_raises():
    with pytest.raises(ValidationError):
        Settings(text_density_threshold=1.1)
    with pytest.raises(ValidationError):
        Settings(text_density_threshold=-0.1)


def test_port_valid():
    assert Settings(port=8000).port == 8000
    assert Settings(port=1).port == 1
    assert Settings(port=65535).port == 65535


def test_port_out_of_range_raises():
    with pytest.raises(ValidationError):
        Settings(port=0)
    with pytest.raises(ValidationError):
        Settings(port=65536)
