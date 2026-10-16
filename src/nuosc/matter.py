"""Three-flavor neutrino oscillations in constant-density matter."""

from typing import Final

import numpy as np
from numpy.typing import NDArray

from nuosc.parameters import OscillationParameters
from nuosc.pmns import pmns_matrix
from nuosc.vacuum import PHASE_FACTOR, Flavor, _flavor_index

# Matter-potential coefficient for
#
#     a = 2 sqrt(2) G_F N_e E
#
# with density in g/cm^3, energy in GeV, and a in eV^2.
MATTER_POTENTIAL_COEFFICIENT: Final[float] = 1.52e-4


def matter_potential(
    energy: float,
    density: float,
    electron_fraction: float = 0.5,
    *,
    antineutrino: bool = False,
) -> float:
    """Calculate the effective charged-current matter potential.

    Parameters
    ----------
    energy:
        Neutrino energy in GeV.
    density:
        Matter density in g/cm^3.
    electron_fraction:
        Electron fraction ``Y_e = N_e / (N_p + N_n)``.
    antineutrino:
        If True, reverse the sign of the matter potential.

    Returns
    -------
    float
        Effective matter potential in eV^2.
    """

    if energy <= 0.0:
        raise ValueError(f"energy must be positive, got {energy}")

    if density < 0.0:
        raise ValueError(f"density must be non-negative, got {density}")

    if not 0.0 <= electron_fraction <= 1.0:
        raise ValueError(
            f"electron_fraction must be between 0 and 1, got {electron_fraction}"
        )

    potential = MATTER_POTENTIAL_COEFFICIENT * electron_fraction * density * energy

    if antineutrino:
        return -potential

    return potential


def matter_amplitude_matrix(
    energy: float,
    baseline: float,
    density: float,
    parameters: OscillationParameters | None = None,
    *,
    electron_fraction: float = 0.5,
    antineutrino: bool = False,
) -> NDArray[np.complex128]:
    """Calculate the amplitude matrix in constant-density matter.

    Parameters
    ----------
    energy:
        Neutrino energy in GeV.
    baseline:
        Propagation distance in km.
    density:
        Constant matter density in g/cm^3.
    parameters:
        Oscillation parameters. Default parameters are used when omitted.
    electron_fraction:
        Electron fraction ``Y_e`` of the matter.
    antineutrino:
        If True, calculate antineutrino propagation.

    Returns
    -------
    numpy.ndarray
        Complex matrix with shape ``(3, 3)``. Element ``[beta, alpha]``
        is the amplitude for a transition from flavor ``alpha`` to
        flavor ``beta``.
    """

    if baseline < 0.0:
        raise ValueError(f"baseline must be non-negative, got {baseline}")

    if parameters is None:
        parameters = OscillationParameters()

    mixing_matrix = pmns_matrix(
        parameters,
        antineutrino=antineutrino,
    )

    mass_squared = np.diag(
        np.array(
            [
                0.0,
                parameters.dm21,
                parameters.dm31,
            ],
            dtype=np.float64,
        )
    )

    effective_mass_squared = mixing_matrix @ mass_squared @ mixing_matrix.conjugate().T

    potential = matter_potential(
        energy=energy,
        density=density,
        electron_fraction=electron_fraction,
        antineutrino=antineutrino,
    )

    effective_mass_squared[0, 0] += potential

    eigenvalues, eigenvectors = np.linalg.eigh(effective_mass_squared)

    phases = np.exp(-1j * PHASE_FACTOR * eigenvalues * baseline / energy)

    return (eigenvectors * phases) @ eigenvectors.conjugate().T


def matter_probability_matrix(
    energy: float,
    baseline: float,
    density: float,
    parameters: OscillationParameters | None = None,
    *,
    electron_fraction: float = 0.5,
    antineutrino: bool = False,
) -> NDArray[np.float64]:
    """Calculate all probabilities in constant-density matter."""

    amplitudes = matter_amplitude_matrix(
        energy=energy,
        baseline=baseline,
        density=density,
        parameters=parameters,
        electron_fraction=electron_fraction,
        antineutrino=antineutrino,
    )

    probabilities = np.abs(amplitudes) ** 2

    return np.clip(probabilities, 0.0, 1.0)


def matter_probability(
    initial: Flavor,
    final: Flavor,
    energy: float,
    baseline: float,
    density: float,
    parameters: OscillationParameters | None = None,
    *,
    electron_fraction: float = 0.5,
    antineutrino: bool = False,
) -> float:
    """Calculate one probability in constant-density matter."""

    initial_index = _flavor_index(initial)
    final_index = _flavor_index(final)

    probabilities = matter_probability_matrix(
        energy=energy,
        baseline=baseline,
        density=density,
        parameters=parameters,
        electron_fraction=electron_fraction,
        antineutrino=antineutrino,
    )

    return float(probabilities[final_index, initial_index])
