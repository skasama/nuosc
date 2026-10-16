"""Tests for the PMNS mixing matrix."""

import numpy as np

from nuosc import OscillationParameters, pmns_matrix


def test_pmns_matrix_has_expected_shape() -> None:
    matrix = pmns_matrix(OscillationParameters())

    assert matrix.shape == (3, 3)


def test_pmns_matrix_is_unitary() -> None:
    matrix = pmns_matrix(OscillationParameters())
    identity = np.eye(3, dtype=np.complex128)

    np.testing.assert_allclose(
        matrix @ matrix.conjugate().T,
        identity,
        atol=1.0e-14,
    )


def test_antineutrino_matrix_is_complex_conjugate() -> None:
    parameters = OscillationParameters()

    neutrino_matrix = pmns_matrix(parameters)
    antineutrino_matrix = pmns_matrix(parameters, antineutrino=True)

    np.testing.assert_allclose(
        antineutrino_matrix,
        neutrino_matrix.conjugate(),
        atol=1.0e-14,
    )


def test_zero_cp_phase_produces_real_matrix() -> None:
    parameters = OscillationParameters(delta_cp=0.0)
    matrix = pmns_matrix(parameters)

    np.testing.assert_allclose(
        matrix.imag,
        np.zeros((3, 3)),
        atol=1.0e-14,
    )
