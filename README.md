# Tailored MQ-MAS excitation/conversion optimiser

This package implements a fast joint optimiser for multiple-quantum excitation
and MQ-to-central-transition conversion pulses under magic-angle spinning.  It
currently supports half-integer spins `I=3/2` and `I=5/2`, with odd MQ orders
that exist in the corresponding Hilbert space.

The central design choice is to optimise the **coherent product** of the
excitation and conversion pathway amplitudes.  This allows crystallite-
dependent pulse phases to refocus between the two pulses instead of forcing
each pulse to have a uniform phase independently.

## Contents

- `cpp/mqmas_core.cpp` - OpenMP C++20 objective/gradient engine.
- `python/mqmas/model.py` - quadrupolar MAS drift preprocessing.
- `python/mqmas/optimizer.py` - pulse parameterisation and L-BFGS driver.
- `python/mqmas/probe.py` - optional first-order complex resonator and adjoint.
- `python/mqmas/validation.py` - direct matrix-exponential validation.
- `examples/` - reproducible Rb2SO4 and AlPO4 scripts.
- `results/` - pulse CSV files, validation JSON, and benchmarks.
- `docs/TECHNICAL_REPORT.md` - paper summary, derivation, results, and limits.

## Build

Requirements: CMake, a C++20 compiler with OpenMP, Python 3.10+, NumPy, SciPy,
Matplotlib, and pytest.

### Linux and macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
python -m pip install -r requirements.txt
```

### Windows (64-bit Visual Studio toolchain)

Install the **Desktop development with C++** workload (including CMake) from
Visual Studio Build Tools, then run these commands in PowerShell from the
repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
cmake -S . -B build -A x64
cmake --build build --config Release --parallel
$env:PYTHONPATH = "$PWD\python"
pytest -q
```

Do not copy `libmqmas_core.so` from Linux to Windows. Native build products are
not portable between operating systems or between 32- and 64-bit processes.
The loader can also be pointed at a nonstandard build using the
`MQMAS_CORE_LIBRARY` environment variable.

For reproducible CPU timings, avoid nested BLAS threading:

```bash
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/python"
```

## Tests

```bash
pytest -q
```

The tests check the rotor-harmonic drift reconstruction and compare the full
analytical control gradient, including the optional probe adjoint, against
centred finite differences.

## Quick examples

```bash
python examples/optimise_rb2so4.py --quick --output results/my_rb_quick
python examples/optimise_alpo4.py --quick --output results/my_al_quick
python examples/validate_supplied.py
```

Production-like settings matching the supplied pulse designs are selected with
`--production`.  They take tens of seconds on the reference 9-core CPU, but the
exact matrix-exponential validation is much slower.

## Pulse files

Each `*_pixels.csv` file contains one complex Cartesian control sample per
0.2 microsecond hardware pixel:

```text
time_us, ux_hz, uy_hz, amplitude_hz, phase_rad
```

`*_delivered.csv` contains the internally repeated (and, if enabled, probe-
filtered) waveform used by the propagator.

## Important scope

The fast model is a high-field effective model.  It keeps instantaneous
first- and second-order quadrupolar energies in the Zeeman basis and integrates
their MAS Fourier series analytically.  It does not propagate the complete
nonsecular time-dependent quadrupolar Hamiltonian.  Final experimental pulses
must therefore be checked in a trusted full-Hamiltonian simulator, preferably
including the measured probe response, timing uncertainty, CSA, and the actual
coherence-selection sequence.
# MQMAS_OC_Python
