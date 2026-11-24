#!/usr/bin/env python3
"""Optimise a joint 87Rb 3Q excitation/conversion pair for two-site Rb2SO4."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from mqmas import (MQMASCore, PairOptimiser, PulseGrid, build_mas_drift,
                   concatenate_drift_models, smooth_random_seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="small smoke-test calculation")
    mode.add_argument("--production", action="store_true", help="settings close to supplied result")
    ap.add_argument("--output", type=Path, default=Path("results/rb2so4_example"))
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()
    production = args.production
    nori = 256 if production else 64
    maxiter = 300 if production else 50
    internal_dt = 0.05e-6 if production else 0.1e-6

    common = dict(spin=1.5, nu0_hz=196.4e6, mas_hz=12.5e3,
                  dt_s=internal_dt, n_orientations=nori, order=2,
                  fourier_samples=16, max_harmonic=4, scramble=True)
    site1 = build_mas_drift(cq_hz=2.72e6, eta=0.93, offset_hz=+2660,
                            seed=args.seed + 1, **common)
    site2 = build_mas_drift(cq_hz=5.29e6, eta=0.12, offset_hz=-2660,
                            seed=args.seed + 2, **common)
    drift = concatenate_drift_models((site1, site2), (0.5, 0.5))
    core = MQMASCore(drift, mq_order=3, b1_scales=(0.9, 1.0, 1.1),
                     b1_weights=(0.25, 0.50, 0.25))
    grid = PulseGrid(80e-6, 10e-6, 0.2e-6, drift.dt_s)
    opt = PairOptimiser(core, grid, max_nutation_hz=100e3,
                        power_penalty=3e-5, smoothness_penalty=1e-4,
                        objective="amplitude")
    rng = np.random.default_rng(args.seed)
    ex0 = smooth_random_seed(grid.n_excitation_pixels, opt.umax, rng)
    cv0 = smooth_random_seed(grid.n_conversion_pixels, opt.umax, rng)
    t0 = time.perf_counter()
    res = opt.optimise(ex0, cv0, maxiter=maxiter)
    elapsed = time.perf_counter() - t0
    opt.save_result(args.output, res, metadata={
        "system": "Rb2SO4 two-site 87Rb", "orientations_per_site": nori,
        "elapsed_s": elapsed, "mode": "production" if production else "quick"})
    print(opt.evaluate_result(res.x))
    print(f"elapsed = {elapsed:.3f} s; saved to {args.output}")


if __name__ == "__main__":
    main()
