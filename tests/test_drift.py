from __future__ import annotations
import numpy as np
from mqmas import build_mas_drift


def test_harmonic_reconstruction():
    d = build_mas_drift(spin=2.5, cq_hz=4e6, eta=0.4, nu0_hz=156.4e6,
                        mas_hz=12_500, dt_s=0.2e-6, n_orientations=8,
                        order=2, fourier_samples=16)
    assert d.reconstruction_rms_hz < 1e-5
    assert d.reconstruction_max_hz < 1e-4
    assert np.allclose(np.abs(d.epre), 1.0)
    assert np.allclose(np.abs(d.epost), 1.0)
