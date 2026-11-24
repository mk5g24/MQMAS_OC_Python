from __future__ import annotations
import numpy as np
from mqmas import build_mas_drift, MQMASCore, PairOptimiser, PulseGrid, Resonator


def check(probe=None):
    drift = build_mas_drift(spin=1.5, cq_hz=1.5e6, eta=0.3, nu0_hz=196.4e6,
                            mas_hz=12_500, dt_s=0.2e-6, n_orientations=4,
                            fourier_samples=16)
    core = MQMASCore(drift, mq_order=3, b1_scales=(0.9, 1.0, 1.1))
    grid = PulseGrid(2.4e-6, 2.0e-6, 0.4e-6, drift.dt_s)
    opt = PairOptimiser(core, grid, max_nutation_hz=100e3, probe=probe,
                        power_penalty=1e-3, smoothness_penalty=1e-3)
    x = np.random.default_rng(2).normal(scale=0.5, size=opt.variable_count)
    f, g = opt.objective_and_gradient(x)
    for idx in (0, 3, x.size - 1):
        h = 1e-5
        xp, xm = x.copy(), x.copy()
        xp[idx] += h; xm[idx] -= h
        fp, _ = opt.objective_and_gradient(xp)
        fm, _ = opt.objective_and_gradient(xm)
        fd = (fp - fm) / (2 * h)
        assert np.isclose(fd, g[idx], rtol=2e-6, atol=2e-9), (idx, fd, g[idx])


def test_gradient_without_probe():
    check(None)


def test_gradient_with_probe():
    check(Resonator(q_loaded=150))
