"""Oscillation parameter definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OscillationParameters:
    """Parameters for standard three-flavor neutrino oscillations.

    Mixing angles and the CP phase are expressed in radians.
    Mass-squared differences are expressed in eV^2.

    The default values are convenient reference values for development.
    They should not be interpreted as an official global-fit parameter set.
    """

    theta12: float = 0.5843
    theta13: float = 0.1489
    theta23: float = 0.8590
    delta_cp: float = -1.601

    dm21: float = 7.42e-5
    dm31: float = 2.517e-3

    def __post_init__(self) -> None:
        """Validate the parameter values."""

        angles = {
            "theta12": self.theta12,
            "theta13": self.theta13,
            "theta23": self.theta23,
        }

        for name, value in angles.items():
            if not 0.0 <= value <= 0.5 * 3.141592653589793:
                raise ValueError(
                    f"{name} must be between 0 and pi/2 radians, got {value}"
                )

        if self.dm21 <= 0.0:
            raise ValueError(f"dm21 must be positive, got {self.dm21}")
