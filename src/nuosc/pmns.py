"""PMNS mixing matrix."""

import numpy as np
from numpy.typing import NDArray

from nuosc.parameters import OscillationParameters

ComplexMatrix = NDArray[np.complex128]


def pmns_matrix(
    parameters: OscillationParameters,
    *,
    antineutrino: bool = False,
) -> ComplexMatrix:
    """Construct the three-flavor PMNS mixing matrix.

    The matrix follows the standard parameterization

        U = R23 U13 R12,

    where the complex CP phase is included in the 1-3 rotation.

    Parameters
    ----------
    parameters:
        Neutrino oscillation parameters.
    antineutrino:
        If True, return the complex-conjugated PMNS matrix.

    Returns
    -------
    numpy.ndarray
        A complex array with shape ``(3, 3)``.
    """

    theta12 = parameters.theta12
    theta13 = parameters.theta13
    theta23 = parameters.theta23
    delta_cp = parameters.delta_cp

    s12 = np.sin(theta12)
    c12 = np.cos(theta12)
    s13 = np.sin(theta13)
    c13 = np.cos(theta13)
    s23 = np.sin(theta23)
    c23 = np.cos(theta23)

    phase_positive = np.exp(1j * delta_cp)
    phase_negative = np.exp(-1j * delta_cp)

    matrix = np.array(
        [
            [
                c12 * c13,
                s12 * c13,
                s13 * phase_negative,
            ],
            [
                -s12 * c23 - c12 * s23 * s13 * phase_positive,
                c12 * c23 - s12 * s23 * s13 * phase_positive,
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * phase_positive,
                -c12 * s23 - s12 * c23 * s13 * phase_positive,
                c23 * c13,
            ],
        ],
        dtype=np.complex128,
    )

    if antineutrino:
        return matrix.conjugate()

    return matrix
