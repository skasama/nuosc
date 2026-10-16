"""Tests for three-flavor neutrino oscillations in vacuum."""

import numpy as np
import pytest

from nuosc import (
    OscillationParameters,
    vacuum_amplitude_matrix,
    vacuum_probability,
    vacuum_probability_matrix,
)


def test_zero_baseline_amplitude_is_identity() -> None:
    amplitudes = vacuum_amplitude_matrix(
        energy=0.6,
        baseline=0.0,
    )

    np.testing.assert_allclose(
        amplitudes,
        np.eye(3, dtype=np.complex128),
        atol=1.0e-14,
    )


def test_zero_baseline_probability_is_identity() -> None:
    probabilities = vacuum_probability_matrix(
        energy=0.6,
        baseline=0.0,
    )

    np.testing.assert_allclose(
        probabilities,
        np.eye(3),
        atol=1.0e-14,
    )


def test_probabilities_are_conserved() -> None:
    probabilities = vacuum_probability_matrix(
        energy=0.6,
        baseline=295.0,
    )

    np.testing.assert_allclose(
        probabilities.sum(axis=0),
        np.ones(3),
        atol=1.0e-14,
    )


def test_probabilities_are_in_physical_range() -> None:
    probabilities = vacuum_probability_matrix(
        energy=0.6,
        baseline=295.0,
    )

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_single_probability_matches_probability_matrix() -> None:
    probability = vacuum_probability(
        initial="mu",
        final="e",
        energy=0.6,
        baseline=295.0,
    )

    probabilities = vacuum_probability_matrix(
        energy=0.6,
        baseline=295.0,
    )

    np.testing.assert_allclose(
        probability,
        probabilities[0, 1],
        atol=1.0e-14,
    )


def test_no_mixing_gives_identity_probabilities() -> None:
    parameters = OscillationParameters(
        theta12=0.0,
        theta13=0.0,
        theta23=0.0,
        delta_cp=0.0,
    )

    probabilities = vacuum_probability_matrix(
        energy=0.6,
        baseline=295.0,
        parameters=parameters,
    )

    np.testing.assert_allclose(
        probabilities,
        np.eye(3),
        atol=1.0e-14,
    )


def test_antineutrino_probability_is_supported() -> None:
    probability = vacuum_probability(
        initial="mu",
        final="e",
        energy=0.6,
        baseline=295.0,
        antineutrino=True,
    )

    assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize(
    "energy",
    [
        0.0,
        -1.0,
    ],
)
def test_non_positive_energy_raises_error(energy: float) -> None:
    with pytest.raises(ValueError, match="energy must be positive"):
        vacuum_probability_matrix(
            energy=energy,
            baseline=295.0,
        )


def test_negative_baseline_raises_error() -> None:
    with pytest.raises(ValueError, match="baseline must be non-negative"):
        vacuum_probability_matrix(
            energy=0.6,
            baseline=-295.0,
        )


def test_unknown_initial_flavor_raises_error() -> None:
    with pytest.raises(ValueError, match="Unknown flavor"):
        vacuum_probability(
            initial="unknown",  # type: ignore[arg-type]
            final="e",
            energy=0.6,
            baseline=295.0,
        )


def test_unknown_final_flavor_raises_error() -> None:
    with pytest.raises(ValueError, match="Unknown flavor"):
        vacuum_probability(
            initial="mu",
            final="unknown",  # type: ignore[arg-type]
            energy=0.6,
            baseline=295.0,
        )
