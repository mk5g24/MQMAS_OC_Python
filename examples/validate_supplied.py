#!/usr/bin/env python3
"""Fast held-out validation of the supplied Rb2SO4 and AlPO4 pulse pairs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from mqmas import MQMASCore, build_mas_drift, concatenate_drift_models

ROOT = Path(__file__).resolve().parents[1]


def load_delivered(folder: Path, name: str) -> np.ndarray:
    a = np.genfromtxt(folder / f"{name}_delivered.csv", delimiter=",", names=True)
    return 2 * np.pi * (a["ux_hz"] + 1j * a["uy_hz"])


def evaluate(core: MQMASCore, ex: np.ndarray, cv: np.ndarray) -> None:
    m, _ = core.evaluate(ex.real, ex.imag, cv.real, cv.imag, gradient=False)
    print(m)


def main() -> None:
    rb_dir = ROOT / "results/rb2so4_10us_dt05"
    kw = dict(spin=1.5, nu0_hz=196.4e6, mas_hz=12.5e3,
              dt_s=0.05e-6, n_orientations=128, order=2,
              fourier_samples=16, max_harmonic=4, scramble=True)
    r1 = build_mas_drift(cq_hz=2.72e6, eta=0.93, offset_hz=+2660, seed=9301, **kw)
    r2 = build_mas_drift(cq_hz=5.29e6, eta=0.12, offset_hz=-2660, seed=9302, **kw)
    rb = concatenate_drift_models((r1, r2))
    rb_core = MQMASCore(rb, mq_order=3, b1_scales=(0.9, 1.0, 1.1),
                        b1_weights=(0.25, 0.50, 0.25))
    print("Rb2SO4:")
    evaluate(rb_core, load_delivered(rb_dir, "excitation"),
             load_delivered(rb_dir, "conversion"))

    al_dir = ROOT / "results/alpo4_10us_dense"
    al = build_mas_drift(spin=2.5, cq_hz=4.06e6, eta=0.33,
                         nu0_hz=156.4e6, mas_hz=12.5e3, dt_s=0.1e-6,
                         n_orientations=256, order=2, fourier_samples=16,
                         max_harmonic=4, seed=9401, scramble=True)
    al_core = MQMASCore(al, mq_order=3, b1_scales=(0.9, 1.0, 1.1),
                        b1_weights=(0.25, 0.50, 0.25))
    print("AlPO4:")
    evaluate(al_core, load_delivered(al_dir, "excitation"),
             load_delivered(al_dir, "conversion"))


if __name__ == "__main__":
    main()
