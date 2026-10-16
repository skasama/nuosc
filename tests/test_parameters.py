"""Tests for oscillation parameters."""

import pytest

from nuosc import OscillationParameters


def test_default_parameters_can_be_created() -> None:
    parameters = OscillationParameters()

    assert parameters.dm21 > 0.0
    assert parameters.dm31 != 0.0


def test_negative_dm31_is_allowed() -> None:
    parameters = OscillationParameters(dm31=-2.498e-3)

    assert parameters.dm31 < 0.0


def test_invalid_mixing_angle_raises_error() -> None:
    with pytest.raises(ValueError, match="theta13"):
        OscillationParameters(theta13=-0.1)


def test_negative_dm21_raises_error() -> None:
    with pytest.raises(ValueError, match="dm21"):
        OscillationParameters(dm21=-7.42e-5)
