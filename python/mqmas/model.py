"""Physical model and MAS drift preprocessing for the tailored MQMAS optimiser.

The fast model uses a high-field, Zeeman-basis effective Hamiltonian.  At every
rotor phase, the quadrupolar Hamiltonian is treated by static perturbation
through second order in C_Q/nu_0.  Its diagonal energies are represented by a
finite rotor Fourier series and integrated analytically over each split step.

Frequencies supplied to this module are ordinary frequencies in Hz.  The C++
control core uses angular RF frequencies in rad/s.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

ArrayF = NDArray[np.float64]
ArrayC = NDArray[np.complex128]


def spin_operators(spin: float) -> tuple[ArrayC, ArrayC, ArrayC]:
    """Return Ix, Iy, Iz in the |I,m> basis ordered m=I,...,-I."""
    d = int(round(2 * spin + 1))
    if abs(d - (2 * spin + 1)) > 1e-10 or d < 2:
        raise ValueError("spin must be an integer or half-integer >= 1/2")
    m = spin - np.arange(d, dtype=float)
    iz = np.diag(m).astype(np.complex128)
    ip = np.zeros((d, d), dtype=np.complex128)
    for j in range(1, d):
        mj = m[j]
        ip[j - 1, j] = math.sqrt(spin * (spin + 1.0) - mj * (mj + 1.0))
    im = ip.conj().T
    ix = 0.5 * (ip + im)
    iy = (ip - im) / (2.0j)
    return ix, iy, iz


def _rz(a: float) -> ArrayF:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> ArrayF:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def euler_zyz(alpha: float, beta: float, gamma: float) -> ArrayF:
    """Active ZYZ rotation matrix."""
    return _rz(alpha) @ _ry(beta) @ _rz(gamma)


def sobol_so3(n: int, *, seed: int = 7, scramble: bool = True) -> tuple[ArrayF, ArrayF]:
    """Quasi-uniform rotations from the EFG PAS into the rotor frame.

    The returned shape is (n,3,3).  Equal weights implement Haar averaging.
    A Sobol net is used when n is a power of two; otherwise the first n points
    of a sufficiently large Sobol net are retained.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    m = int(math.ceil(math.log2(n)))
    sampler = qmc.Sobol(d=3, scramble=scramble, seed=seed)
    u = sampler.random_base2(m=m)[:n]
    alpha = 2.0 * np.pi * u[:, 0]
    beta = np.arccos(np.clip(1.0 - 2.0 * u[:, 1], -1.0, 1.0))
    gamma = 2.0 * np.pi * u[:, 2]
    rotations = np.empty((n, 3, 3), dtype=float)
    for j in range(n):
        rotations[j] = euler_zyz(float(alpha[j]), float(beta[j]), float(gamma[j]))
    return rotations, np.full(n, 1.0 / n, dtype=float)


def quadrupolar_hamiltonian_hz(
    spin: float,
    cq_hz: float,
    eta: float,
    r_lab_from_pas: ArrayF,
    ops: tuple[ArrayC, ArrayC, ArrayC] | None = None,
) -> ArrayC:
    r"""Return the full quadrupolar Hamiltonian H_Q/h in Hz.

    The convention is

        H_Q/h = C_Q/[4 I (2I-1)] *
                {3 I_z'^2 - I(I+1) 1 + eta (I_x'^2-I_y'^2)}.

    ``r_lab_from_pas`` maps PAS vector components into laboratory components;
    its columns are the PAS axes expressed in the laboratory frame.
    """
    if spin <= 0.5:
        raise ValueError("quadrupolar interaction requires I > 1/2")
    if not (0.0 <= eta <= 1.0 + 1e-12):
        raise ValueError("eta must lie in [0,1]")
    if ops is None:
        ops = spin_operators(spin)
    ix, iy, iz = ops
    lab_ops = (ix, iy, iz)
    pas_ops: list[ArrayC] = []
    for axis in range(3):
        op = sum(r_lab_from_pas[k, axis] * lab_ops[k] for k in range(3))
        pas_ops.append(np.asarray(op, dtype=np.complex128))
    ixp, iyp, izp = pas_ops
    d = ix.shape[0]
    ident = np.eye(d, dtype=np.complex128)
    prefactor = cq_hz / (4.0 * spin * (2.0 * spin - 1.0))
    hq = prefactor * (
        3.0 * (izp @ izp)
        - spin * (spin + 1.0) * ident
        + eta * ((ixp @ ixp) - (iyp @ iyp))
    )
    return 0.5 * (hq + hq.conj().T)


def effective_zeeman_energies_hz(
    spin: float,
    cq_hz: float,
    eta: float,
    nu0_hz: float,
    r_lab_from_pas: ArrayF,
    *,
    order: int = 2,
    offset_hz: float = 0.0,
    ops: tuple[ArrayC, ArrayC, ArrayC] | None = None,
) -> ArrayF:
    r"""Instantaneous high-field quadrupolar energies in the Zeeman basis.

    ``order=1`` retains the diagonal first-order quadrupolar Hamiltonian.
    ``order=2`` adds static second-order perturbation terms

        E_m^(2) = sum_{k != m} |H_Q,mk|^2/(E_m^Z-E_k^Z),

    with E_m^Z=-nu0*m.  This is an adiabatic high-field approximation to the
    time-dependent MAS Hamiltonian; corrections scale with both C_Q/nu0 and
    nu_r/nu0.  A common, time-dependent identity term is removed.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if nu0_hz <= 0.0:
        raise ValueError("nu0_hz must be positive")
    if ops is None:
        ops = spin_operators(spin)
    hq = quadrupolar_hamiltonian_hz(spin, cq_hz, eta, r_lab_from_pas, ops)
    d = hq.shape[0]
    m = spin - np.arange(d, dtype=float)
    energy = np.real(np.diag(hq)).copy() - offset_hz * m
    if order == 2:
        ez = -nu0_hz * m
        for i in range(d):
            denom = ez[i] - ez
            mask = np.arange(d) != i
            energy[i] += np.sum(np.abs(hq[i, mask]) ** 2 / denom[mask]).real
    energy -= np.mean(energy)
    return energy


@dataclass(frozen=True)
class MASDriftModel:
    spin: float
    cq_hz: float
    eta: float
    nu0_hz: float
    mas_hz: float
    dt_s: float
    nphase: int
    order: int
    rotations: ArrayF
    powder_weights: ArrayF
    fourier_k: NDArray[np.int64]
    fourier_coeff_hz: ArrayC  # [orientation, k, m]
    epre: ArrayC              # [orientation, rotor phase, m]
    epost: ArrayC
    reconstruction_rms_hz: float
    reconstruction_max_hz: float

    @property
    def dimension(self) -> int:
        return int(round(2 * self.spin + 1))

    @property
    def rotor_period_s(self) -> float:
        return 1.0 / self.mas_hz


def _integral_fourier(
    coeff: ArrayC,
    k: NDArray[np.int64],
    phi0: ArrayF,
    phi1: ArrayF,
    omega_r: float,
) -> ArrayC:
    """Integrate sum_k coeff[k] exp(i k omega_r t) from phi0 to phi1.

    coeff shape: (..., nk, d); phi arrays shape (np,).  Output (...,np,d)
    in units coeff * seconds.
    """
    lead = coeff.shape[:-2]
    nk, d = coeff.shape[-2:]
    out = np.zeros(lead + (phi0.size, d), dtype=np.complex128)
    for idx in range(nk):
        kk = int(k[idx])
        c = coeff[..., idx, :]
        if kk == 0:
            factor = (phi1 - phi0) / omega_r
        else:
            factor = (np.exp(1j * kk * phi1) - np.exp(1j * kk * phi0)) / (1j * kk * omega_r)
        out += c[..., None, :] * factor.reshape((1,) * len(lead) + (phi0.size, 1))
    return out


def build_mas_drift(
    *,
    spin: float,
    cq_hz: float,
    eta: float,
    nu0_hz: float,
    mas_hz: float,
    dt_s: float,
    n_orientations: int = 128,
    order: int = 2,
    offset_hz: float = 0.0,
    fourier_samples: int = 24,
    max_harmonic: int | None = None,
    seed: int = 7,
    scramble: bool = True,
    rotations: ArrayF | None = None,
    powder_weights: ArrayF | None = None,
) -> MASDriftModel:
    """Precompute exact integrated MAS drift phases for Strang splitting.

    ``dt_s`` is adjusted very slightly so an integer number of propagation
    steps fits in one rotor period.  For first-order energies harmonics through
    |k|=2 are exact; static second-order perturbative energies require |k|<=4.
    """
    if mas_hz <= 0.0 or dt_s <= 0.0:
        raise ValueError("mas_hz and dt_s must be positive")
    if fourier_samples < 2:
        raise ValueError("fourier_samples too small")
    if max_harmonic is None:
        max_harmonic = 2 if order == 1 else 4
    if fourier_samples < 2 * max_harmonic + 1:
        raise ValueError("fourier_samples cannot resolve requested harmonics")
    nphase = max(1, int(round(1.0 / (mas_hz * dt_s))))
    actual_dt = 1.0 / (mas_hz * nphase)
    if rotations is None:
        rotations, weights = sobol_so3(n_orientations, seed=seed, scramble=scramble)
    else:
        rotations = np.asarray(rotations, dtype=float)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape (n,3,3)")
        n_orientations = rotations.shape[0]
        weights = np.full(n_orientations, 1.0 / n_orientations)
    if powder_weights is not None:
        weights = np.asarray(powder_weights, dtype=float)
        if weights.shape != (n_orientations,) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("invalid powder_weights")
        weights = weights / weights.sum()

    ops = spin_operators(spin)
    d = int(round(2 * spin + 1))
    theta_m = math.acos(1.0 / math.sqrt(3.0))
    r_lab_rotor = _ry(theta_m)
    sample_phi = 2.0 * np.pi * np.arange(fourier_samples) / fourier_samples
    energies = np.empty((n_orientations, fourier_samples, d), dtype=float)
    for o in range(n_orientations):
        r0 = rotations[o]
        for j, phi in enumerate(sample_phi):
            r = r_lab_rotor @ _rz(float(phi)) @ r0
            energies[o, j] = effective_zeeman_energies_hz(
                spin, cq_hz, eta, nu0_hz, r,
                order=order, offset_hz=offset_hz, ops=ops,
            )

    full_coeff = np.fft.fft(energies, axis=1) / fourier_samples
    all_k = np.rint(np.fft.fftfreq(fourier_samples) * fourier_samples).astype(int)
    keep = np.abs(all_k) <= max_harmonic
    k = all_k[keep]
    coeff = full_coeff[:, keep, :]
    # Sort harmonics from negative to positive for reproducible files.
    order_idx = np.argsort(k)
    k = k[order_idx]
    coeff = coeff[:, order_idx, :]

    # Quantify the truncation against the directly sampled energies.
    reconstructed = np.zeros_like(energies, dtype=np.complex128)
    for q, kk in enumerate(k):
        reconstructed += coeff[:, q:q+1, :] * np.exp(1j * kk * sample_phi)[None, :, None]
    residual = np.real(reconstructed) - energies
    rms = float(np.sqrt(np.mean(residual * residual)))
    maxerr = float(np.max(np.abs(residual)))

    omega_r = 2.0 * np.pi * mas_hz
    phi = 2.0 * np.pi * np.arange(nphase) / nphase
    half = 0.5 * omega_r * actual_dt
    pre_int = _integral_fourier(coeff, k, phi, phi + half, omega_r)
    post_int = _integral_fourier(coeff, k, phi + half, phi + 2.0 * half, omega_r)
    # Numerical roundoff can leave tiny imaginary parts in integrals of a real function.
    pre_int = np.real(pre_int)
    post_int = np.real(post_int)
    epre = np.exp(-2.0j * np.pi * pre_int)
    epost = np.exp(-2.0j * np.pi * post_int)

    return MASDriftModel(
        spin=spin,
        cq_hz=cq_hz,
        eta=eta,
        nu0_hz=nu0_hz,
        mas_hz=mas_hz,
        dt_s=actual_dt,
        nphase=nphase,
        order=order,
        rotations=np.asarray(rotations),
        powder_weights=np.asarray(weights),
        fourier_k=k,
        fourier_coeff_hz=coeff,
        epre=np.ascontiguousarray(epre, dtype=np.complex128),
        epost=np.ascontiguousarray(epost, dtype=np.complex128),
        reconstruction_rms_hz=rms,
        reconstruction_max_hz=maxerr,
    )


def make_ensemble_mapping(
    drift: MASDriftModel,
    b1_scales: Iterable[float] = (1.0,),
    b1_weights: Iterable[float] | None = None,
) -> tuple[ArrayF, NDArray[np.int32], NDArray[np.int32], ArrayF]:
    """Return weights, drift indices, B1 indices and B1 scale vector."""
    b1 = np.asarray(tuple(b1_scales), dtype=float)
    if b1.ndim != 1 or b1.size == 0 or np.any(b1 <= 0):
        raise ValueError("b1_scales must be positive")
    if b1_weights is None:
        bw = np.full(b1.size, 1.0 / b1.size)
    else:
        bw = np.asarray(tuple(b1_weights), dtype=float)
        if bw.shape != b1.shape or np.any(bw < 0) or bw.sum() <= 0:
            raise ValueError("invalid b1_weights")
        bw = bw / bw.sum()
    no = drift.rotations.shape[0]
    didx = np.repeat(np.arange(no, dtype=np.int32), b1.size)
    bidx = np.tile(np.arange(b1.size, dtype=np.int32), no)
    weights = (drift.powder_weights[:, None] * bw[None, :]).reshape(-1)
    weights /= weights.sum()
    return np.ascontiguousarray(weights), didx, bidx, np.ascontiguousarray(b1)


def concatenate_drift_models(
    models: Iterable[MASDriftModel],
    model_weights: Iterable[float] | None = None,
) -> MASDriftModel:
    """Combine sites or parameter samples that share spin/MAS/time grids.

    The returned powder weights include the supplied site/parameter weights.
    Scalar ``cq_hz`` and ``eta`` metadata are set to NaN when models differ.
    """
    models = tuple(models)
    if not models:
        raise ValueError("at least one model is required")
    ref = models[0]
    for m in models[1:]:
        for name in ("spin", "nu0_hz", "mas_hz", "dt_s", "nphase", "order"):
            if not np.isclose(getattr(m, name), getattr(ref, name), rtol=1e-12, atol=1e-15):
                raise ValueError(f"incompatible drift models: {name}")
        if not np.array_equal(m.fourier_k, ref.fourier_k):
            raise ValueError("incompatible Fourier harmonics")
    if model_weights is None:
        mw = np.full(len(models), 1.0 / len(models))
    else:
        mw = np.asarray(tuple(model_weights), dtype=float)
        if mw.shape != (len(models),) or np.any(mw < 0) or mw.sum() <= 0:
            raise ValueError("invalid model_weights")
        mw = mw / mw.sum()
    weights = np.concatenate([w * m.powder_weights for w, m in zip(mw, models)])
    weights /= weights.sum()
    cq_equal = all(np.isclose(m.cq_hz, ref.cq_hz) for m in models)
    eta_equal = all(np.isclose(m.eta, ref.eta) for m in models)
    return MASDriftModel(
        spin=ref.spin,
        cq_hz=ref.cq_hz if cq_equal else float("nan"),
        eta=ref.eta if eta_equal else float("nan"),
        nu0_hz=ref.nu0_hz,
        mas_hz=ref.mas_hz,
        dt_s=ref.dt_s,
        nphase=ref.nphase,
        order=ref.order,
        rotations=np.concatenate([m.rotations for m in models], axis=0),
        powder_weights=weights,
        fourier_k=ref.fourier_k.copy(),
        fourier_coeff_hz=np.concatenate([m.fourier_coeff_hz for m in models], axis=0),
        epre=np.concatenate([m.epre for m in models], axis=0),
        epost=np.concatenate([m.epost for m in models], axis=0),
        reconstruction_rms_hz=float(np.sqrt(np.mean([m.reconstruction_rms_hz**2 for m in models]))),
        reconstruction_max_hz=max(m.reconstruction_max_hz for m in models),
    )
