# nuosc
A Pythonic library for three-flavor neutrino oscillation calculations

## Vacuum oscillations

`nuosc` can calculate standard three-flavor neutrino oscillation
probabilities in vacuum.

A neutrino produced with flavor $\alpha$ is a superposition of the three
mass eigenstates:

```math
|\nu_\alpha\rangle
=
\sum_i U_{\alpha i}^{*} |\nu_i\rangle,
```

where $U$ is the PMNS mixing matrix.

During propagation, each mass eigenstate acquires a different phase. The
flavor-transition amplitude matrix is calculated as

```math
S(L,E)
=
U
\begin{pmatrix}
1 & 0 & 0 \\
0 & \exp\left(-i\frac{\Delta m_{21}^{2}L}{2E}\right) & 0 \\
0 & 0 & \exp\left(-i\frac{\Delta m_{31}^{2}L}{2E}\right)
\end{pmatrix}
U^\dagger.
```

The probability for a neutrino with initial flavor $\alpha$ to be detected
with flavor $\beta$ is

```math
P(\nu_\alpha \rightarrow \nu_\beta)
=
|S_{\beta\alpha}|^2.
```

Only the mass-squared differences are required because a phase common to all
three mass eigenstates does not affect the oscillation probabilities.

### Units

The vacuum calculation uses the following units:

- neutrino energy: GeV
- baseline: km
- mass-squared differences: eV$^2$
- mixing angles and the CP phase: radians

### Calculate one oscillation probability

For example, the following code calculates the
$\nu_\mu \rightarrow \nu_e$ appearance probability for a neutrino with an
energy of $1.0$ GeV after traveling $100$ km:

```python
from nuosc import vacuum_probability

probability = vacuum_probability(
    initial="mu",
    final="e",
    energy=1.0,
    baseline=100,
)

print(probability)
```

## Example

A basic example comparing vacuum and constant-density matter oscillations
is available in `examples/example.py`.

Run the example from the repository root:

```bash
uv run python examples/example.py
```

The example calculates the transition probabilities from an initial muon
neutrino to electron, muon, and tau neutrinos. It also checks that the three
probabilities sum to one.

### Plot oscillation probabilities

The energy dependence of the
$\nu_\mu \rightarrow \nu_e$ appearance probability can be plotted using:

```bash
uv run python examples/plot_probabilities.py
```

The example compares vacuum and constant-density matter probabilities for
both neutrinos and antineutrinos. The default settings are:

- baseline: 295 km
- matter density: 2.6 g/cm$^3$
- electron fraction: 0.5
- energy range: 0.1 to 2.0 GeV

The resulting figure is saved as:

```text
oscillation_probabilities.png
```
``