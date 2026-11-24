"""Independent validation utilities for the MQMAS optimiser."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .model import MASDriftModel, spin_operators

ArrayC = NDArray[np.complex128]


@dataclass(frozen=True)
class ExactMetrics:
    coherent_amplitude: float
    incoherent_amplitude: float
    coherence_factor: float
    coherent_complex: complex
    elapsed_s: float
    ensemble_size: int
    substeps_per_control_step: int


def drift_energy_hz(drift: MASDriftModel, orientation: int, phase_rad: float) -> NDArray[np.float64]:
    phase = np.exp(1j * drift.fourier_k * phase_rad)
    e = np.einsum("k,kd->d", phase, drift.fourier_coeff_hz[orientation], optimize=True)
    return np.real(e)


def _unitary_hermitian(h_angular: ArrayC, dt_s: float) -> ArrayC:
    val, vec = np.linalg.eigh(h_angular)
    return (vec * np.exp(-1j * dt_s * val)) @ vec.conj().T


def exact_pair_metrics(
    drift: MASDriftModel,
    excitation: ArrayC,
    conversion: ArrayC,
    *,
    mq_order: int = 3,
    b1_scales: Iterable[float] = (1.0,),
    b1_weights: Iterable[float] | None = None,
    excitation_start_phase_rad: float = 0.0,
    conversion_start_phase_rad: float = 0.0,
    substeps: int = 4,
) -> ExactMetrics:
    """Validate with direct exponentiation of drift+RF at fine midpoint steps.

    This removes the RF/drift Strang splitting used by the fast core.  It still
    validates the same high-field effective Hamiltonian; it does not restore
    nonsecular quadrupolar terms omitted by that model.
    """
    if substeps < 1:
        raise ValueError("substeps must be positive")
    excitation = np.asarray(excitation, dtype=np.complex128)
    conversion = np.asarray(conversion, dtype=np.complex128)
    b1 = np.asarray(tuple(b1_scales), dtype=float)
    if b1_weights is None:
        bw = np.full(b1.size, 1.0 / b1.size)
    else:
        bw = np.asarray(tuple(b1_weights), dtype=float)
        bw = bw / bw.sum()
    spin = drift.spin
    d = drift.dimension
    ix, iy, iz = spin_operators(spin)
    mvals = spin - np.arange(d)
    mq_plus = int(np.flatnonzero(np.isclose(mvals, 0.5 * mq_order))[0])
    mq_minus = int(np.flatnonzero(np.isclose(mvals, -0.5 * mq_order))[0])
    ct_plus = int(np.flatnonzero(np.isclose(mvals, 0.5))[0])
    ct_minus = int(np.flatnonzero(np.isclose(mvals, -0.5))[0])
    omega_r = 2.0 * np.pi * drift.mas_hz
    fine_dt = drift.dt_s / substeps
    start = perf_counter()
    coherent = 0j
    incoherent = 0.0
    for o in range(drift.rotations.shape[0]):
        for ib, bs in enumerate(b1):
            ue = np.eye(d, dtype=np.complex128)
            for n, u in enumerate(excitation):
                for q in range(substeps):
                    tmid = (n + (q + 0.5) / substeps) * drift.dt_s
                    e = drift_energy_hz(drift, o, excitation_start_phase_rad + omega_r * tmid)
                    h = 2.0 * np.pi * np.diag(e) + bs * (u.real * ix + u.imag * iy)
                    ue = _unitary_hermitian(h, fine_dt) @ ue
            rho = ue @ iz @ ue.conj().T
            ae = rho[mq_plus, mq_minus]

            uc = np.eye(d, dtype=np.complex128)
            for n, u in enumerate(conversion):
                for q in range(substeps):
                    tmid = (n + (q + 0.5) / substeps) * drift.dt_s
                    e = drift_energy_hz(drift, o, conversion_start_phase_rad + omega_r * tmid)
                    h = 2.0 * np.pi * np.diag(e) + bs * (u.real * ix + u.imag * iy)
                    uc = _unitary_hermitian(h, fine_dt) @ uc
            ac = uc[ct_plus, mq_plus] * np.conj(uc[ct_minus, mq_minus])
            w = drift.powder_weights[o] * bw[ib]
            coherent += w * ae * ac
            incoherent += w * abs(ae * ac)
    elapsed = perf_counter() - start
    amp = abs(coherent) / spin
    inc = incoherent / spin
    return ExactMetrics(
        coherent_amplitude=float(amp),
        incoherent_amplitude=float(inc),
        coherence_factor=float(abs(coherent) / incoherent if incoherent else 0.0),
        coherent_complex=complex(coherent / spin),
        elapsed_s=elapsed,
        ensemble_size=drift.rotations.shape[0] * b1.size,
        substeps_per_control_step=substeps,
    )
