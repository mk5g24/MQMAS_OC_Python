"""Joint excitation/conversion optimiser for MQMAS pulse pairs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult

from .bindings import MQMASCore, PairMetrics
from .probe import Resonator

ArrayF = NDArray[np.float64]
ArrayC = NDArray[np.complex128]


@dataclass(frozen=True)
class PulseGrid:
    excitation_duration_s: float
    conversion_duration_s: float
    pixel_dt_s: float
    internal_dt_s: float

    def __post_init__(self) -> None:
        ratio = self.pixel_dt_s / self.internal_dt_s
        if abs(ratio - round(ratio)) > 1e-10:
            raise ValueError("pixel_dt_s must be an integer multiple of internal_dt_s")
        for duration in (self.excitation_duration_s, self.conversion_duration_s):
            n = duration / self.pixel_dt_s
            if abs(n - round(n)) > 1e-10:
                raise ValueError("pulse durations must be integer multiples of pixel_dt_s")

    @property
    def repeat(self) -> int:
        return int(round(self.pixel_dt_s / self.internal_dt_s))

    @property
    def n_excitation_pixels(self) -> int:
        return int(round(self.excitation_duration_s / self.pixel_dt_s))

    @property
    def n_conversion_pixels(self) -> int:
        return int(round(self.conversion_duration_s / self.pixel_dt_s))


@dataclass
class OptimisationRecord:
    iteration: int
    objective: float
    coherent_amplitude: float
    incoherent_amplitude: float
    coherence_factor: float
    elapsed_s: float


def _radial_map(z: ArrayC, umax: float) -> tuple[ArrayC, ArrayF, ArrayF]:
    """Smoothly map unconstrained complex variables into |u| < umax.

    Returns u, g=tanh(r)/r, and h=(dg/dr)/r for the Jacobian
    du = umax [g I + h z z^T] dz in real Cartesian coordinates.
    """
    z = np.asarray(z, dtype=np.complex128)
    r = np.abs(z)
    g = np.empty_like(r)
    h = np.empty_like(r)
    small = r < 1e-5
    rs = r[small]
    g[small] = 1.0 - rs * rs / 3.0 + 2.0 * rs**4 / 15.0
    h[small] = -2.0 / 3.0 + 8.0 * rs * rs / 15.0
    rr = r[~small]
    th = np.tanh(rr)
    g[~small] = th / rr
    h[~small] = ((1.0 / np.cosh(rr) ** 2) * rr - th) / (rr**3)
    return umax * g * z, g, h


def _radial_pullback(z: ArrayC, grad_u: ArrayC, umax: float, g: ArrayF, h: ArrayF) -> ArrayC:
    dot = z.real * grad_u.real + z.imag * grad_u.imag
    return umax * (g * grad_u + h * z * dot)


def physical_to_raw(u: ArrayC, umax: float) -> ArrayC:
    u = np.asarray(u, dtype=np.complex128)
    ratio = np.clip(np.abs(u) / umax, 0.0, 1.0 - 1e-12)
    r = np.arctanh(ratio)
    phase = np.exp(1j * np.angle(u))
    return r * phase


def smooth_random_seed(n: int, umax: float, rng: np.random.Generator, amplitude: float = 0.7) -> ArrayC:
    phase_steps = rng.normal(scale=0.45, size=n)
    phase = np.cumsum(phase_steps)
    # Smooth amplitude with nonzero endpoints to avoid a flat zero-gradient seed.
    amp = amplitude * umax * (0.75 + 0.25 * np.cos(np.linspace(-np.pi, np.pi, n)))
    return amp * np.exp(1j * phase)


def constant_seed(n: int, umax: float, amplitude: float = 0.75, phase: float = 0.0) -> ArrayC:
    return np.full(n, amplitude * umax * np.exp(1j * phase), dtype=np.complex128)


class PairOptimiser:
    """Joint optimiser for an MQ excitation pulse and MQ->CT conversion pulse.

    Controls are specified in angular frequency (rad/s).  The C++ core returns
    the exact gradient of the split propagator.  Control pixels are repeated on
    a finer internal grid, and their gradients are summed exactly.
    """

    def __init__(
        self,
        core: MQMASCore,
        grid: PulseGrid,
        *,
        max_nutation_hz: float,
        excitation_phase_index: int = 0,
        conversion_phase_index: int = 0,
        rotor_phase_pairs: Iterable[tuple[int, int, float]] | None = None,
        power_penalty: float = 0.0,
        smoothness_penalty: float = 0.0,
        probe: Resonator | None = None,
        objective: str = "amplitude",
    ) -> None:
        if abs(core.drift.dt_s - grid.internal_dt_s) > 1e-12 * max(1.0, grid.internal_dt_s):
            raise ValueError("grid internal_dt_s must equal the drift model's actual dt_s")
        if objective not in ("amplitude", "squared"):
            raise ValueError("objective must be 'amplitude' or 'squared'")
        self.core = core
        self.grid = grid
        self.umax = 2.0 * np.pi * max_nutation_hz
        self.oe = excitation_phase_index
        self.oc = conversion_phase_index
        if rotor_phase_pairs is None:
            self.phase_pairs = ((self.oe, self.oc, 1.0),)
        else:
            pairs = tuple((int(a), int(b), float(w)) for a, b, w in rotor_phase_pairs)
            if not pairs or any(w < 0 for _, _, w in pairs) or sum(w for _, _, w in pairs) <= 0:
                raise ValueError("invalid rotor_phase_pairs")
            ws = sum(w for _, _, w in pairs)
            self.phase_pairs = tuple((a, b, w / ws) for a, b, w in pairs)
        self.power_penalty = float(power_penalty)
        self.smoothness_penalty = float(smoothness_penalty)
        self.probe = probe
        self.objective_kind = objective
        self.history: list[OptimisationRecord] = []
        self._start = 0.0
        self._last: tuple[float, PairMetrics] | None = None

    @property
    def variable_count(self) -> int:
        return 2 * (self.grid.n_excitation_pixels + self.grid.n_conversion_pixels)

    def pack_raw(self, ze: ArrayC, zc: ArrayC) -> ArrayF:
        return np.concatenate((ze.real, ze.imag, zc.real, zc.imag)).astype(float)

    def unpack_raw(self, x: ArrayF) -> tuple[ArrayC, ArrayC]:
        ne, nc = self.grid.n_excitation_pixels, self.grid.n_conversion_pixels
        x = np.asarray(x, dtype=float)
        if x.size != 2 * (ne + nc):
            raise ValueError("wrong variable count")
        a, b, c, d = np.split(x, [ne, 2 * ne, 2 * ne + nc])
        return a + 1j * b, c + 1j * d

    def _expand_and_filter(self, u: ArrayC) -> tuple[ArrayC, ArrayC | None]:
        expanded = np.repeat(u, self.grid.repeat)
        if self.probe is None:
            return expanded, None
        delivered, states = self.probe.forward(expanded, self.grid.internal_dt_s)
        return delivered, states

    def _collapse_gradient(self, g: ArrayC, states: ArrayC | None) -> ArrayC:
        if self.probe is not None:
            assert states is not None
            g = self.probe.adjoint(g, states, self.grid.internal_dt_s)
        return g.reshape(-1, self.grid.repeat).sum(axis=1)

    def _regularisation(self, ue: ArrayC, uc: ArrayC) -> tuple[float, ArrayC, ArrayC]:
        score_penalty = 0.0
        ge = np.zeros_like(ue)
        gc = np.zeros_like(uc)
        for u, g in ((ue, ge), (uc, gc)):
            if self.power_penalty:
                scale = self.power_penalty / max(1, u.size) / (self.umax**2)
                score_penalty += scale * float(np.vdot(u, u).real)
                g += 2.0 * scale * u
            if self.smoothness_penalty and u.size > 1:
                du = np.diff(u)
                scale = self.smoothness_penalty / (u.size - 1) / (self.umax**2)
                score_penalty += scale * float(np.vdot(du, du).real)
                g[:-1] += 2.0 * scale * (u[:-1] - u[1:])
                g[1:] += 2.0 * scale * (u[1:] - u[:-1])
        return score_penalty, ge, gc

    def objective_and_gradient(self, x: ArrayF) -> tuple[float, ArrayF]:
        ze, zc = self.unpack_raw(x)
        ue, ge_map, he_map = _radial_map(ze, self.umax)
        uc, gc_map, hc_map = _radial_map(zc, self.umax)
        de, se = self._expand_and_filter(ue)
        dc, sc = self._expand_and_filter(uc)
        metric_acc = np.zeros(10, dtype=float)
        gde = np.zeros(ue.size, dtype=np.complex128)
        gdc = np.zeros(uc.size, dtype=np.complex128)
        score = 0.0
        for oe, oc, pw in self.phase_pairs:
            metrics_i, grads = self.core.evaluate(
                de.real, de.imag, dc.real, dc.imag,
                excitation_phase_index=oe, conversion_phase_index=oc, gradient=True,
            )
            assert grads is not None
            metric_acc += pw * np.array(list(metrics_i.__dict__.values()), dtype=float)
            gex, gey, gcx, gcy = grads
            ge_i = self._collapse_gradient(gex + 1j * gey, se)
            gc_i = self._collapse_gradient(gcx + 1j * gcy, sc)
            if self.objective_kind == "amplitude":
                denom = max(2.0 * metrics_i.coherent_amplitude, 1e-14)
                score += pw * metrics_i.coherent_amplitude
                gde += pw * ge_i / denom
                gdc += pw * gc_i / denom
            else:
                score += pw * metrics_i.objective
                gde += pw * ge_i
                gdc += pw * gc_i
        metrics = PairMetrics.from_array(metric_acc)
        penalty, pe, pc = self._regularisation(ue, uc)
        score -= penalty
        gde -= pe
        gdc -= pc
        gze = _radial_pullback(ze, gde, self.umax, ge_map, he_map)
        gzc = _radial_pullback(zc, gdc, self.umax, gc_map, hc_map)
        gradient = self.pack_raw(gze, gzc)
        self._last = (score, metrics)
        return -score, -gradient

    def optimise(
        self,
        excitation_seed: ArrayC,
        conversion_seed: ArrayC,
        *,
        maxiter: int = 150,
        maxls: int = 30,
        gtol: float = 1e-7,
        callback: Callable[[OptimisationRecord], None] | None = None,
    ) -> OptimizeResult:
        if excitation_seed.shape != (self.grid.n_excitation_pixels,):
            raise ValueError("excitation seed length mismatch")
        if conversion_seed.shape != (self.grid.n_conversion_pixels,):
            raise ValueError("conversion seed length mismatch")
        x0 = self.pack_raw(physical_to_raw(excitation_seed, self.umax),
                           physical_to_raw(conversion_seed, self.umax))
        self.history.clear()
        self._start = time.perf_counter()
        iteration = 0

        def cb(xk: ArrayF) -> None:
            nonlocal iteration
            iteration += 1
            # SciPy usually calls the objective immediately before callback;
            # recalculate only if needed so the history is always reliable.
            f, _ = self.objective_and_gradient(xk)
            assert self._last is not None
            _, m = self._last
            rec = OptimisationRecord(
                iteration=iteration,
                objective=-f,
                coherent_amplitude=m.coherent_amplitude,
                incoherent_amplitude=m.incoherent_amplitude,
                coherence_factor=m.coherence_factor,
                elapsed_s=time.perf_counter() - self._start,
            )
            self.history.append(rec)
            if callback:
                callback(rec)

        result = minimize(
            fun=self.objective_and_gradient,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-4.5, 4.5)] * x0.size,
            callback=cb,
            options={"maxiter": maxiter, "maxls": maxls, "gtol": gtol,
                     "ftol": 1e-12, "maxcor": 20},
        )
        return result

    def physical_controls(self, x: ArrayF) -> tuple[ArrayC, ArrayC, ArrayC, ArrayC]:
        ze, zc = self.unpack_raw(x)
        ue, _, _ = _radial_map(ze, self.umax)
        uc, _, _ = _radial_map(zc, self.umax)
        de, _ = self._expand_and_filter(ue)
        dc, _ = self._expand_and_filter(uc)
        return ue, uc, de, dc

    def evaluate_result(self, x: ArrayF) -> PairMetrics:
        _, _, de, dc = self.physical_controls(x)
        acc = np.zeros(10, dtype=float)
        for oe, oc, pw in self.phase_pairs:
            m, _ = self.core.evaluate(de.real, de.imag, dc.real, dc.imag,
                                      excitation_phase_index=oe, conversion_phase_index=oc,
                                      gradient=False)
            acc += pw * np.array(list(m.__dict__.values()), dtype=float)
        return PairMetrics.from_array(acc)

    def save_result(self, directory: str | Path, result: OptimizeResult, metadata: dict | None = None) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        ue, uc, de, dc = self.physical_controls(result.x)
        m = self.evaluate_result(result.x)
        np.savetxt(out / "excitation_pixels.csv",
                   np.column_stack((np.arange(ue.size) * self.grid.pixel_dt_s * 1e6,
                                    ue.real / (2*np.pi), ue.imag / (2*np.pi),
                                    np.abs(ue)/(2*np.pi), np.angle(ue))),
                   delimiter=",", header="time_us,ux_hz,uy_hz,amplitude_hz,phase_rad", comments="")
        np.savetxt(out / "conversion_pixels.csv",
                   np.column_stack((np.arange(uc.size) * self.grid.pixel_dt_s * 1e6,
                                    uc.real / (2*np.pi), uc.imag / (2*np.pi),
                                    np.abs(uc)/(2*np.pi), np.angle(uc))),
                   delimiter=",", header="time_us,ux_hz,uy_hz,amplitude_hz,phase_rad", comments="")
        np.savetxt(out / "excitation_delivered.csv",
                   np.column_stack((np.arange(de.size) * self.grid.internal_dt_s * 1e6,
                                    de.real/(2*np.pi), de.imag/(2*np.pi))),
                   delimiter=",", header="time_us,ux_hz,uy_hz", comments="")
        np.savetxt(out / "conversion_delivered.csv",
                   np.column_stack((np.arange(dc.size) * self.grid.internal_dt_s * 1e6,
                                    dc.real/(2*np.pi), dc.imag/(2*np.pi))),
                   delimiter=",", header="time_us,ux_hz,uy_hz", comments="")
        payload = {
            "success": bool(result.success), "message": str(result.message),
            "nit": int(result.nit), "nfev": int(result.nfev),
            "metrics": asdict(m), "grid": asdict(self.grid),
            "max_nutation_hz": self.umax/(2*np.pi),
            "power_penalty": self.power_penalty,
            "smoothness_penalty": self.smoothness_penalty,
            "probe": None if self.probe is None else asdict(self.probe),
            "rotor_phase_pairs": [list(x) for x in self.phase_pairs],
            "history": [asdict(h) for h in self.history],
        }
        if metadata:
            payload["metadata"] = metadata
        (out / "result.json").write_text(json.dumps(payload, indent=2))
