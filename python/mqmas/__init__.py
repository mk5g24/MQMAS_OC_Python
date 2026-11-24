"""Tailored low-rank optimal control for static and MAS MQMAS pulses."""
from .model import (
    MASDriftModel,
    build_mas_drift,
    concatenate_drift_models,
    effective_zeeman_energies_hz,
    make_ensemble_mapping,
    quadrupolar_hamiltonian_hz,
    sobol_so3,
    spin_operators,
)
from .bindings import MQMASCore, PairMetrics

__all__ = [
    "MASDriftModel", "MQMASCore", "PairMetrics", "build_mas_drift",
    "concatenate_drift_models",
    "effective_zeeman_energies_hz", "make_ensemble_mapping",
    "quadrupolar_hamiltonian_hz", "sobol_so3", "spin_operators",
]
from .optimizer import PairOptimiser, PulseGrid, constant_seed, smooth_random_seed
from .probe import Resonator

__all__ += ["PairOptimiser", "PulseGrid", "Resonator", "constant_seed", "smooth_random_seed"]
from .validation import ExactMetrics, exact_pair_metrics
__all__ += ["ExactMetrics", "exact_pair_metrics"]
