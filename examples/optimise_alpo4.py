#!/usr/bin/env python3
"""Optimise a joint 27Al 3Q excitation/conversion pair for AlPO4."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from mqmas import (MQMASCore, PairOptimiser, PulseGrid, build_mas_drift,
                   smooth_random_seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--production", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("results/alpo4_example"))
    ap.add_argument("--seed", type=int, default=81)
    args = ap.parse_args()
    production = args.production
    nori = 512 if production else 96
    maxiter = 350 if production else 50

    drift = build_mas_drift(
        spin=2.5, cq_hz=4.06e6, eta=0.33, nu0_hz=156.4e6,
        mas_hz=12.5e3, dt_s=0.1e-6, n_orientations=nori,
        order=2, fourier_samples=16, max_harmonic=4,
        seed=args.seed + 1, scramble=True)
    core = MQMASCore(drift, mq_order=3, b1_scales=(0.9, 1.0, 1.1),
                     b1_weights=(0.25, 0.50, 0.25))
    grid = PulseGrid(10e-6, 10e-6, 0.2e-6, drift.dt_s)
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
        "system": "AlPO4 27Al", "orientations": nori,
        "elapsed_s": elapsed, "mode": "production" if production else "quick"})
    print(opt.evaluate_result(res.x))
    print(f"elapsed = {elapsed:.3f} s; saved to {args.output}")


if __name__ == "__main__":
    main()
