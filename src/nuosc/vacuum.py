"""Three-flavor neutrino oscillations in vacuum."""

from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from nuosc.parameters import OscillationParameters
from nuosc.pmns import pmns_matrix

type Flavor = Literal["e", "mu", "tau"]

FLAVOR_INDEX: Final[dict[str, int]] = {
    "e": 0,
    "mu": 1,
    "tau": 2,
}

# The transition-amplitude phase is
#
#     exp(-i * PHASE_FACTOR * dm2[eV^2] * L[km] / E[GeV]).
#
# This is twice the commonly used 1.2669 factor appearing inside
# sin^2(dm2 L / 4E) in oscillation probabilities.
PHASE_FACTOR: Final[float] = 2.533865


def _flavor_index(flavor: str) -> int:
    """Convert a flavor name to its matrix index."""

    try:
        return FLAVOR_INDEX[flavor]
    except KeyError as error:
        valid = ", ".join(FLAVOR_INDEX)
        raise ValueError(
            f"Unknown flavor {flavor!r}. Expected one of: {valid}"
        ) from error


def vacuum_amplitude_matrix(
    energy: float,
    baseline: float,
    parameters: OscillationParameters | None = None,
    *,
    antineutrino: bool = False,
) -> NDArray[np.complex128]:
    """Calculate the flavor-transition amplitude matrix in vacuum.

    Parameters
    ----------
    energy:
        Neutrino energy in GeV.
    baseline:
        Propagation distance in km.
    parameters:
        Oscillation parameters. Default parameters are used when omitted.
    antineutrino:
        If True, calculate the antineutrino amplitude matrix.

    Returns
    -------
    numpy.ndarray
        Complex matrix with shape ``(3, 3)``. Element ``[beta, alpha]``
        is the amplitude for a transition from flavor ``alpha`` to
        flavor ``beta``.
    """

    if energy <= 0.0:
        raise ValueError(f"energy must be positive, got {energy}")

    if baseline < 0.0:
        raise ValueError(f"baseline must be non-negative, got {baseline}")

    if parameters is None:
        parameters = OscillationParameters()

    mixing_matrix = pmns_matrix(
        parameters,
        antineutrino=antineutrino,
    )

    mass_squared = np.array(
        [
            0.0,
            parameters.dm21,
            parameters.dm31,
        ],
        dtype=np.float64,
    )

    phases = np.exp(-1j * PHASE_FACTOR * mass_squared * baseline / energy)

    return (mixing_matrix * phases) @ mixing_matrix.conjugate().T


def vacuum_probability_matrix(
    energy: float,
    baseline: float,
    parameters: OscillationParameters | None = None,
    *,
    antineutrino: bool = False,
) -> NDArray[np.float64]:
    """Calculate all three-flavor oscillation probabilities in vacuum.

    Element ``[beta, alpha]`` is the probability for a transition from
    flavor ``alpha`` to flavor ``beta``.
    """

    amplitudes = vacuum_amplitude_matrix(
        energy=energy,
        baseline=baseline,
        parameters=parameters,
        antineutrino=antineutrino,
    )

    probabilities = np.abs(amplitudes) ** 2

    # Remove tiny floating-point excursions outside the physical range.
    return np.clip(probabilities, 0.0, 1.0)


def vacuum_probability(
    initial: Flavor,
    final: Flavor,
    energy: float,
    baseline: float,
    parameters: OscillationParameters | None = None,
    *,
    antineutrino: bool = False,
) -> float:
    """Calculate one three-flavor oscillation probability in vacuum.

    Parameters
    ----------
    initial:
        Initial flavor: ``"e"``, ``"mu"``, or ``"tau"``.
    final:
        Final flavor: ``"e"``, ``"mu"``, or ``"tau"``.
    energy:
        Neutrino energy in GeV.
    baseline:
        Propagation distance in km.
    parameters:
        Oscillation parameters. Default parameters are used when omitted.
    antineutrino:
        If True, calculate the antineutrino probability.

    Returns
    -------
    float
        Oscillation probability between zero and one.
    """

    initial_index = _flavor_index(initial)
    final_index = _flavor_index(final)

    probabilities = vacuum_probability_matrix(
        energy=energy,
        baseline=baseline,
        parameters=parameters,
        antineutrino=antineutrino,
    )

    return float(probabilities[final_index, initial_index])
