"""Plot vacuum and constant-density matter oscillation probabilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nuosc import matter_probability, vacuum_probability


def calculate_probabilities(
    energies: np.ndarray,
    *,
    baseline: float,
    density: float,
    electron_fraction: float,
    antineutrino: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate vacuum and matter probabilities over an energy range."""

    vacuum_values = np.array(
        [
            vacuum_probability(
                initial="mu",
                final="e",
                energy=float(energy),
                baseline=baseline,
                antineutrino=antineutrino,
            )
            for energy in energies
        ]
    )

    matter_values = np.array(
        [
            matter_probability(
                initial="mu",
                final="e",
                energy=float(energy),
                baseline=baseline,
                density=density,
                electron_fraction=electron_fraction,
                antineutrino=antineutrino,
            )
            for energy in energies
        ]
    )

    return vacuum_values, matter_values


def main() -> None:
    """Plot the muon-to-electron appearance probability."""

    baseline = 100.
    density = 2.6
    electron_fraction = 0.5

    energies = np.linspace(0.1, 2.0, 500)

    vacuum_values, matter_values = calculate_probabilities(
        energies,
        baseline=baseline,
        density=density,
        electron_fraction=electron_fraction,
    )

    antineutrino_vacuum, antineutrino_matter = calculate_probabilities(
        energies,
        baseline=baseline,
        density=density,
        electron_fraction=electron_fraction,
        antineutrino=True,
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        energies,
        vacuum_values,
        label=r"Vacuum $\nu_\mu \rightarrow \nu_e$",
        color="tab:blue",
        linestyle="--",
    )
    axis.plot(
        energies,
        matter_values,
        label=r"Matter $\nu_\mu \rightarrow \nu_e$",
        color="tab:blue",
    )
    axis.plot(
        energies,
        antineutrino_vacuum,
        label=r"Vacuum $\bar{\nu}_\mu \rightarrow \bar{\nu}_e$",
        color="tab:orange",
        linestyle="--",
    )
    axis.plot(
        energies,
        antineutrino_matter,
        label=r"Matter $\bar{\nu}_\mu \rightarrow \bar{\nu}_e$",
        color="tab:orange",
    )

    axis.set_xlabel("Neutrino energy [GeV]")
    axis.set_ylabel("Oscillation probability")
    axis.set_title(
        rf"$L={baseline:.0f}$ km, "
        rf"$\rho={density:.1f}$ g/cm$^3$, "
        rf"$Y_e={electron_fraction:.1f}$"
    )
    axis.set_xlim(energies.min(), energies.max())
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    output_path = Path("oscillation_probabilities.png")
    figure.savefig(output_path, dpi=150)

    print(f"Saved plot to: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
