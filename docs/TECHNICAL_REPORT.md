# A tailored optimal-control engine for MQ-MAS excitation and conversion

## Executive summary

A custom CPU optimiser was built for the paired RF-control problem in MQ-MAS:
creation of an odd-order multiple-quantum coherence from thermal `Iz`, followed
by conversion of that coherence to the central transition.  The implementation
is tailored to static high-field quadrupolar spins under periodic MAS rather
than being a generic GRAPE program.

The main accelerations are:

1. the MAS-modulated diagonal drift is represented by a short rotor Fourier
   series and integrated analytically over every half time step;
2. periodic drift phases are precomputed once and recycled modulo the rotor
   period;
3. each RF rotation and its exact Cartesian Frechet derivatives are evaluated
   once per time step and B1 scale, then reused for every crystallite;
4. only a few state vectors are propagated, giving leading cost
   `O(M N d^2)` rather than density-matrix `O(M N d^3)`;
5. excitation and conversion are optimised jointly through their coherent
   pathway product, so crystallite-dependent phases may cancel between pulses.

On the reference AMD EPYC 9V74 virtual CPU, an objective plus all four control
gradients takes approximately 26-28 ms for 1,536 ensemble members.  The supplied
10 microsecond excitation and 10 microsecond conversion pairs were optimised in
about 16 s for two-site Rb2SO4 (`I=3/2`) and 19 s for AlPO4 (`I=5/2`).

## 1. What could be established from the linked 2026 paper

The linked article is:

V. Kaushik and N. Khaneja, *Control framework for broadband population transfer
in multilevel quantum ensembles with time-dependent dispersions*, Physical
Review A **113**, 062620 (2026), DOI 10.1103/zm4y-wbyb.

The APS full text required authorization in the environment used for this work,
so the following is an abstract-level summary rather than a section-by-section
reconstruction.

The paper addresses an ensemble of `n`-level systems whose nominal energy
levels and ensemble dispersion both vary in time.  Its stated strategy is to
choose the control so that the target transition lives in a retained effective
subspace dynamically decoupled from the remaining levels.  A phase design then
moves the residual ensemble dispersion into a global final-state phase, making
population transfer uniform across ensemble members.  Quadrupolar nuclei under
MAS are used as the physical example because rotor motion creates both explicit
time dependence and crystallite-dependent, time-varying dispersion.

Three ideas are particularly relevant to MQ-MAS:

- **Retain only the task-relevant subspace or matrix elements.**  For MQ-MAS the
  relevant quantity is not the full density operator but the selected pQ
  coherence and its conversion to CT coherence.
- **Treat phase dispersion as a design variable.**  Uniform transfer magnitude
  is insufficient if the crystallite contributions cancel after powder
  averaging.
- **Exploit the deterministic time dependence.**  MAS is periodic, so its drift
  should be represented and reused rather than recomputed as unrelated dense
  matrix exponentials at every time sample.

The first and third ideas directly determine the data structures in this code.
The second leads to the joint pair objective below.

## 2. MQ-MAS control problem

For a half-integer spin `I`, let `p` be an odd MQ order, normally `p=3`.  In the
Zeeman basis, define the selected pQ matrix element

```math
A_s^{\mathrm{ex}}
 = \langle +p/2|U_{e,s} I_z U_{e,s}^{\dagger}|-p/2\rangle .
```

A conversion propagator contributes to CT coherence through

```math
A_s^{\mathrm{conv}}
 = \langle +1/2|U_{c,s}|+p/2\rangle
   \langle -1/2|U_{c,s}|-p/2\rangle^* .
```

Here `s` labels crystallite orientation, site, offset, B1 scale, and any other
ensemble parameter.  The selected-pathway CT amplitude is therefore

```math
C = \sum_s w_s A_s^{\mathrm{ex}} A_s^{\mathrm{conv}} .
```

The optimisation maximises

```math
\Phi = |C|/I
```

(or equivalently its square).  This objective differs fundamentally from
maximising `sum |A_ex|` and `sum |A_conv|` separately: it rewards the phase
relation that survives the complete excitation-conversion pair.

This is motivated by quadrupolar-jolting-frame analyses of rotor-synchronous
satellite-transition pulses.  A single long pulse can create a strongly
orientation-dependent effective RF phase, but identical excitation and
conversion pulses can refocus the opposite phase factors.  Double-frequency
cosine irradiation can also make the two ST effective Hamiltonians conjugate,
although its intrinsically phase-convergent bandwidth is limited.  The joint
objective allows unrestricted Cartesian controls to discover the same type of
phase compensation without forcing the two waveforms to be identical.

The present objective corresponds to zero indirect evolution, or to
rotor-period-synchronised increments after known evolution phases have been
removed.  A complete sequence optimisation can multiply each member by its
known t1 pathway phase, or average over selected t1 increments.

## 3. High-field MAS drift approximation

The quadrupolar Hamiltonian convention used in the preprocessing code is

```math
\frac{H_Q}{h}
 = \frac{C_Q}{4I(2I-1)}
   \left[3I_{z'}^2-I(I+1)\mathbf 1
   +\eta_Q(I_{x'}^2-I_{y'}^2)\right].
```

At each rotor phase the code retains the diagonal first-order term and adds the
static second-order perturbative correction

```math
E_m^{(2)}(t)
 = \sum_{k\ne m}
   \frac{|\langle m|H_Q(t)/h|k\rangle|^2}
        {E_m^Z/h-E_k^Z/h},
\qquad E_m^Z/h=-\nu_0 m .
```

The common identity component is removed.  This gives a time-dependent drift
that is diagonal in the Zeeman basis,

```math
D_s(t)=2\pi\,\mathrm{diag}\{d_{s,m}(t)\} .
```

For MAS about the magic-angle rotor axis, each energy is represented as

```math
d_{s,m}(t)=\sum_{k=-K}^{K}d_{s,m}^{(k)}e^{ik\omega_r t} .
```

`K=2` is sufficient for the first-order term, and `K=4` for the instantaneous
second-order construction used here.  The coefficients are obtained from 16
uniform rotor phases.  Reconstruction errors in the supplied cases are near
floating-point precision.

This is an **instantaneous high-field effective-energy approximation**.  It is
not the exact time-dependent van Vleck/Floquet Hamiltonian and does not retain
nonsecular quadrupolar mixing during the pulse.  Its accuracy should improve as
`C_Q/nu_0` and `nu_r/nu_0` decrease, but it must be checked against a complete
Hamiltonian for the intended experiment.

## 4. Exact rotor integration plus Strang splitting

Because the drift is diagonal and given by a finite Fourier series, its phase
integral over any interval is analytical.  For a control interval
`[t_n,t_n+dt]`, define

```math
E_{s,n}^{\mathrm{pre}}
 = \exp\left[-i\int_{t_n}^{t_n+dt/2}D_s(t)dt\right],
```

```math
E_{s,n}^{\mathrm{post}}
 = \exp\left[-i\int_{t_n+dt/2}^{t_n+dt}D_s(t)dt\right].
```

Only the drift/RF noncommutation is approximated:

```math
U_{s,n}
 \approx E_{s,n}^{\mathrm{post}}
 R_{g,n}
 E_{s,n}^{\mathrm{pre}},
```

```math
R_{g,n}
 = \exp[-i b_g dt (u_{x,n}I_x+u_{y,n}I_y)] .
```

This is a second-order Strang step.  Unlike a naive sampled-drift Trotter
scheme, the MAS drift phase inside both half steps is integrated exactly within
the retained harmonic model.

The rotor phase table is periodic.  `Epre` and `Epost` are therefore calculated
once for every crystallite and fine-grid rotor index, then reused in every
objective and gradient call.  The RF matrices depend only on the control and B1
group, so they are shared by all crystallites.

## 5. Exact derivatives of the split propagator

The small RF Hamiltonian is diagonalised analytically/numerically in dimension
4 or 6.  For each Cartesian control, the derivative of the matrix exponential
is evaluated with the exact divided-difference Frechet derivative.  Thus the
computed gradient is exact for the split propagator; no small-control or
first-order GRAPE derivative is used.

Forward and backward vector trajectories give derivatives of `A_ex` and
`A_conv`.  The chain rule through the product `A_ex A_conv` and the coherent
ensemble sum supplies all four gradients:

```text
excitation ux, excitation uy, conversion ux, conversion uy.
```

For `M` ensemble members, `N` total fine time steps, and Hilbert dimension `d`,
the leading work is `O(M N d^2)`.  The implementation avoids Liouville space
and full density-matrix propagation.

## 6. Controls, constraints, and optional hardware model

Hardware pixels are complex Cartesian controls.  A smooth radial map

```math
u_n=u_{\max}\,\frac{\tanh|z_n|}{|z_n|}z_n
```

keeps every pixel below the RF ceiling while retaining unconstrained variables
for L-BFGS.  Optional penalties control RF power and complex-waveform
smoothness.

A hardware pixel can be repeated on a finer internal propagation grid.  The
fine-grid gradients are summed exactly back to the pixel variable.  The
supplied Rb pulse uses 0.2 microsecond pixels and 0.05 microsecond internal
steps; Al uses 0.2 and 0.1 microseconds.

The optional probe model is

```math
\dot y=-(\kappa+i\Delta)y+\kappa u,
```

with exact zero-order-hold propagation and an exact discrete adjoint.  This can
be placed inside the optimisation so the C++ core sees the delivered coil
field while the optimiser updates the commanded waveform.

Rotor-phase uncertainty can likewise be included by averaging the objective
and gradient over several excitation/conversion start indices.  This is useful
because the supplied Rb solution is noticeably more start-phase sensitive than
the Al solution.

## 7. Numerical demonstrations

These are proof-of-principle calculations, not experimentally calibrated pulse
claims.  Both use 10 microsecond excitation and 10 microsecond conversion
pulses, 0.2 microsecond command pixels, MAS at 12.5 kHz, and B1 scales
`0.9, 1.0, 1.1` with weights `0.25, 0.50, 0.25`.

### Rb2SO4, 87Rb, I=3/2, 3Q

Two equally weighted sites were included:

- `C_Q=2.72 MHz`, `eta=0.93`, offset `+2.660 kHz`;
- `C_Q=5.29 MHz`, `eta=0.12`, offset `-2.660 kHz`.

The field was represented by `nu_0=196.4 MHz`; maximum nutation frequency was
120 kHz.  Optimisation used 256 orientations per site and 0.05 microsecond
internal steps.

### AlPO4, 27Al, I=5/2, 3Q

The model used `C_Q=4.06 MHz`, `eta=0.33`, `nu_0=156.4 MHz`, and a maximum
nutation frequency of 100 kHz.  Optimisation used 512 orientations and 0.1
microsecond internal steps.

### Independent validation

| Quantity | Rb2SO4 | AlPO4 |
|---|---:|---:|
| Training coherent pathway amplitude | 0.22263 | 0.30110 |
| Held-out coherent pathway amplitude | 0.20023 | 0.28898 |
| Held-out incoherent amplitude | 0.22270 | 0.29899 |
| Held-out coherence factor | 0.8991 | 0.9665 |
| Best tested rectangular-pulse amplitude | 0.03626 | 0.08953 |
| Shaped / rectangular amplitude ratio | 5.52 | 3.23 |
| Fine-step change | +0.506% (0.05 vs 0.025 us) | +0.209% (0.10 vs 0.025 us) |
| Split vs direct-exponential subset | +0.604% | +0.233% |
| Optimisation time | 15.6 s | 19.3 s |
| Objective + gradient, 1,536 members | 26.0 ms | 28.2 ms |

The coherent pathway amplitude is `|sum w_s A_ex,s A_conv,s|/I`; it is not the
same quantity as a published experimental MQMAS efficiency.  The coherence
factor is the coherent amplitude divided by `sum w_s |A_ex,s A_conv,s|/I` and
therefore reports phase cancellation separately from transfer magnitude.

The direct-exponential check removes Strang splitting on a 96-member subset but
still uses the same high-field diagonal effective drift.  It does not validate
the omitted nonsecular quadrupolar dynamics.

Both optimisations reached their iteration limits while still changing slowly;
the waveforms are good local solutions, not proofs of global optimality.  Their
RF amplitudes spend much of the time close to the imposed ceiling and use phase
modulation to exploit the rotor-dependent multilevel dynamics.

## 8. What the linked paper suggests beyond this implementation

The paper's retained-subspace and global-phase ideas suggest several next
steps:

1. **Explicit reduced-subspace parameterisation.**  Rather than allowing an
   arbitrary RF waveform, construct a control ansatz whose transformed-frame
   Hamiltonian suppresses coupling outside the chosen pQ/CT pathway.  This may
   reduce the search dimension and improve powder generalisation.
2. **Target a prescribed global phase.**  The present objective is invariant to
   one global complex phase.  For phase cycling or concatenation with other
   sequence elements, a phase-sensitive real-part objective may be preferable.
3. **Include t1 evolution in the retained-pathway product.**  Known
   crystallite-dependent evolution phases can be incorporated before the
   coherent ensemble sum, and several t1 points can be optimised jointly.
4. **Use rotor-phase jitter as an ensemble coordinate.**  This directly attacks
   timing-dependent phase dispersion and t1 noise.
5. **Hybrid analytical/numerical seeds.**  QJF-derived single- or
   double-frequency long pulses, OCFASTER-like low-power pulses, and the
   paper's reduced-subspace construction can seed the unrestricted optimiser.

## 9. Limitations and recommended validation ladder

The fast model currently omits or simplifies:

- exact nonsecular, time-dependent quadrupolar evolution;
- quadrupolar mixing of the RF matrix elements;
- CSA anisotropy and relative EFG/CSA tensors;
- dipolar/J couplings and relaxation;
- explicit free t1 evolution, phase cycling, and coherence-selection gradients;
- measured probe transfer functions unless the optional resonator is enabled;
- rotor-frequency and start-phase distributions unless supplied as ensembles.

A sensible workflow is:

1. optimise quickly with the supplied Fourier/split engine;
2. validate with a finer internal time step;
3. validate with direct exponentials of the same effective Hamiltonian;
4. export the waveform to a trusted complete-Hamiltonian MAS simulator;
5. add probe response and timing uncertainty;
6. refine on a compressed exact model or on the user's larger machine.

The current fast runs are not too heavy for the reference machine.  A full
nonsecular exact-Hamiltonian optimisation over thousands of orientations,
several B1/offset samples, multiple rotor phases, and many t1 points would be
substantially heavier and is the point at which a workstation/GPU or the user's
machine becomes worthwhile.

## References

1. V. Kaushik and N. Khaneja, Physical Review A 113, 062620 (2026),
   DOI 10.1103/zm4y-wbyb.
2. I. Hung and Z. Gan, Journal of Magnetic Resonance 328, 106994 (2021),
   DOI 10.1016/j.jmr.2021.106994.
3. T. Vosegaard et al., Journal of the American Chemical Society 127,
   13768-13769 (2005), DOI 10.1021/ja054035g.
4. A. Medek, J. S. Harwood, and L. Frydman, Journal of the American Chemical
   Society 117, 12779-12787 (1995), DOI 10.1021/ja00156a015.
