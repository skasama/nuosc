"""Basic example of three-flavor neutrino oscillation calculations."""

from nuosc import (
    matter_probability,
    vacuum_probability,
)


def print_probabilities(
    *,
    energy: float,
    baseline: float,
    density: float,
    electron_fraction: float,
    antineutrino: bool = False,
) -> None:
    """Print vacuum and matter probabilities for an initial muon neutrino."""

    particle_name = "antineutrino" if antineutrino else "neutrino"

    print(f"Particle: {particle_name}")
    print(f"Energy: {energy:.3f} GeV")
    print(f"Baseline: {baseline:.1f} km")
    print(f"Density: {density:.3f} g/cm^3")
    print(f"Electron fraction: {electron_fraction:.3f}")
    print()

    vacuum_sum = 0.0
    matter_sum = 0.0

    print("Transition probabilities")
    print("-" * 55)
    print(f"{'Channel':<15}{'Vacuum':>15}{'Matter':>15}")
    print("-" * 55)

    for final_flavor in ("e", "mu", "tau"):
        vacuum_value = vacuum_probability(
            initial="mu",
            final=final_flavor,
            energy=energy,
            baseline=baseline,
            antineutrino=antineutrino,
        )

        matter_value = matter_probability(
            initial="mu",
            final=final_flavor,
            energy=energy,
            baseline=baseline,
            density=density,
            electron_fraction=electron_fraction,
            antineutrino=antineutrino,
        )

        vacuum_sum += vacuum_value
        matter_sum += matter_value

        channel = f"mu -> {final_flavor}"

        print(f"{channel:<15}{vacuum_value:>15.8f}{matter_value:>15.8f}")

    print("-" * 55)
    print(f"{'Sum':<15}{vacuum_sum:>15.8f}{matter_sum:>15.8f}")


def main() -> None:
    """Run the basic oscillation example."""

    print_probabilities(
        energy=0.6,
        baseline=295.0,
        density=2.6,
        electron_fraction=0.5,
    )


if __name__ == "__main__":
    main()
