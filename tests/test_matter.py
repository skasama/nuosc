"""Tests for oscillations in constant-density matter."""

import numpy as np
import pytest

from nuosc import (
    matter_amplitude_matrix,
    matter_potential,
    matter_probability,
    matter_probability_matrix,
    vacuum_amplitude_matrix,
    vacuum_probability_matrix,
)


def test_zero_baseline_amplitude_is_identity() -> None:
    amplitudes = matter_amplitude_matrix(
        energy=0.6,
        baseline=0.0,
        density=2.6,
    )

    np.testing.assert_allclose(
        amplitudes,
        np.eye(3, dtype=np.complex128),
        atol=1.0e-14,
    )


def test_zero_density_matches_vacuum_amplitude() -> None:
    matter_amplitudes = matter_amplitude_matrix(
        energy=0.6,
        baseline=295.0,
        density=0.0,
    )

    vacuum_amplitudes = vacuum_amplitude_matrix(
        energy=0.6,
        baseline=295.0,
    )

    np.testing.assert_allclose(
        matter_amplitudes,
        vacuum_amplitudes,
        atol=1.0e-13,
    )


def test_zero_density_matches_vacuum_probability() -> None:
    matter_probabilities = matter_probability_matrix(
        energy=0.6,
        baseline=295.0,
        density=0.0,
    )

    vacuum_probabilities = vacuum_probability_matrix(
        energy=0.6,
        baseline=295.0,
    )

    np.testing.assert_allclose(
        matter_probabilities,
        vacuum_probabilities,
        atol=1.0e-13,
    )


def test_probabilities_are_conserved() -> None:
    probabilities = matter_probability_matrix(
        energy=0.6,
        baseline=295.0,
        density=2.6,
    )

    np.testing.assert_allclose(
        probabilities.sum(axis=0),
        np.ones(3),
        atol=1.0e-14,
    )


def test_probabilities_are_in_physical_range() -> None:
    probabilities = matter_probability_matrix(
        energy=0.6,
        baseline=295.0,
        density=2.6,
    )

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_single_probability_matches_matrix() -> None:
    probability = matter_probability(
        initial="mu",
        final="e",
        energy=0.6,
        baseline=295.0,
        density=2.6,
    )

    probabilities = matter_probability_matrix(
        energy=0.6,
        baseline=295.0,
        density=2.6,
    )

    np.testing.assert_allclose(
        probability,
        probabilities[0, 1],
        atol=1.0e-14,
    )


def test_antineutrino_potential_has_opposite_sign() -> None:
    neutrino_potential = matter_potential(
        energy=1.0,
        density=2.6,
        electron_fraction=0.5,
    )

    antineutrino_potential = matter_potential(
        energy=1.0,
        density=2.6,
        electron_fraction=0.5,
        antineutrino=True,
    )

    assert antineutrino_potential == -neutrino_potential


def test_positive_density_changes_probability() -> None:
    matter_value = matter_probability(
        initial="mu",
        final="e",
        energy=0.6,
        baseline=295.0,
        density=2.6,
    )

    vacuum_value = matter_probability(
        initial="mu",
        final="e",
        energy=0.6,
        baseline=295.0,
        density=0.0,
    )

    assert not np.isclose(matter_value, vacuum_value, atol=1.0e-8)


@pytest.mark.parametrize("density", [-1.0, -0.1])
def test_negative_density_raises_error(density: float) -> None:
    with pytest.raises(ValueError, match="density must be non-negative"):
        matter_probability_matrix(
            energy=0.6,
            baseline=295.0,
            density=density,
        )


@pytest.mark.parametrize("electron_fraction", [-0.1, 1.1])
def test_invalid_electron_fraction_raises_error(
    electron_fraction: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="electron_fraction must be between 0 and 1",
    ):
        matter_probability_matrix(
            energy=0.6,
            baseline=295.0,
            density=2.6,
            electron_fraction=electron_fraction,
        )


def test_non_positive_energy_raises_error() -> None:
    with pytest.raises(ValueError, match="energy must be positive"):
        matter_probability_matrix(
            energy=0.0,
            baseline=295.0,
            density=2.6,
        )


def test_negative_baseline_raises_error() -> None:
    with pytest.raises(ValueError, match="baseline must be non-negative"):
        matter_probability_matrix(
            energy=0.6,
            baseline=-295.0,
            density=2.6,
        )
