"""Tailored low-rank optimal control for static and MAS MQMAS pulses.

The native C++ binding is loaded lazily.  This keeps pure-Python utilities such
as :mod:`mqmas.reporting` usable before the shared library has been built.
"""
from __future__ import annotations

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
from .probe import Resonator

__all__ = [
    "MASDriftModel",
    "build_mas_drift",
    "concatenate_drift_models",
    "effective_zeeman_energies_hz",
    "make_ensemble_mapping",
    "quadrupolar_hamiltonian_hz",
    "sobol_so3",
    "spin_operators",
    "Resonator",
    "MQMASCore",
    "PairMetrics",
    "PairOptimiser",
    "PulseGrid",
    "constant_seed",
    "smooth_random_seed",
    "ExactMetrics",
    "exact_pair_metrics",
    "CaseSpec",
    "ReportArtifacts",
    "ReportError",
    "ReportSettings",
    "generate_report",
    "load_report_config",
]


def __getattr__(name: str):
    if name in {"MQMASCore", "PairMetrics"}:
        from .bindings import MQMASCore, PairMetrics

        return {"MQMASCore": MQMASCore, "PairMetrics": PairMetrics}[name]
    if name in {"PairOptimiser", "PulseGrid", "constant_seed", "smooth_random_seed"}:
        from .optimizer import PairOptimiser, PulseGrid, constant_seed, smooth_random_seed

        return {
            "PairOptimiser": PairOptimiser,
            "PulseGrid": PulseGrid,
            "constant_seed": constant_seed,
            "smooth_random_seed": smooth_random_seed,
        }[name]
    if name in {"ExactMetrics", "exact_pair_metrics"}:
        from .validation import ExactMetrics, exact_pair_metrics

        return {"ExactMetrics": ExactMetrics, "exact_pair_metrics": exact_pair_metrics}[name]
    if name in {
        "CaseSpec",
        "ReportArtifacts",
        "ReportError",
        "ReportSettings",
        "generate_report",
        "load_report_config",
    }:
        from .reporting import (
            CaseSpec,
            ReportArtifacts,
            ReportError,
            ReportSettings,
            generate_report,
            load_report_config,
        )

        return {
            "CaseSpec": CaseSpec,
            "ReportArtifacts": ReportArtifacts,
            "ReportError": ReportError,
            "ReportSettings": ReportSettings,
            "generate_report": generate_report,
            "load_report_config": load_report_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
