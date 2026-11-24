"""Simple complex-envelope resonator model and exact discrete adjoint.

The loaded resonator is represented by
    dy/dt = -(kappa + i*delta) y + kappa u,
where u is the commanded complex RF envelope and y the coil field.  The field
returned for each sample is the exact interval average under zero-order hold.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]


@dataclass(frozen=True)
class Resonator:
    f0_hz: float = 150e6
    q_loaded: float = 150.0
    detuning_hz: float = 0.0
    gain: float = 1.0

    @property
    def kappa_rad_s(self) -> float:
        return np.pi * self.f0_hz / self.q_loaded

    @property
    def ringdown_s(self) -> float:
        return 1.0 / self.kappa_rad_s

    def coefficients(self, dt_s: float) -> tuple[complex, complex, complex, complex]:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        kappa = self.kappa_rad_s
        lam = kappa + 2.0j * np.pi * self.detuning_hz
        a = np.exp(-lam * dt_s)
        b = kappa * (1.0 - a) / lam
        av = (1.0 - a) / (lam * dt_s)
        bv = (kappa / lam) * (1.0 - av)
        return complex(a), complex(b), complex(av), complex(bv)

    def forward(self, command: ArrayC, dt_s: float, initial: complex = 0j) -> tuple[ArrayC, ArrayC]:
        command = np.asarray(command, dtype=np.complex128)
        if command.ndim != 1:
            raise ValueError("command must be 1D")
        a, b, av, bv = self.coefficients(dt_s)
        states = np.empty(command.size + 1, dtype=np.complex128)
        field = np.empty(command.size, dtype=np.complex128)
        states[0] = initial
        for n, u in enumerate(command):
            field[n] = self.gain * (av * states[n] + bv * u)
            states[n + 1] = a * states[n] + b * u
        return field, states

    def adjoint(self, field_gradient: ArrayC, states: ArrayC, dt_s: float) -> ArrayC:
        """Map dJ/d(field_x,field_y) to dJ/d(command_x,command_y).

        Complex gradients use dJ = Re[conj(g) dz].
        """
        g = np.asarray(field_gradient, dtype=np.complex128)
        states = np.asarray(states, dtype=np.complex128)
        if states.shape != (g.size + 1,):
            raise ValueError("states shape mismatch")
        a, b, av, bv = self.coefficients(dt_s)
        lam_next = 0j
        gu = np.empty_like(g)
        for n in range(g.size - 1, -1, -1):
            gv = self.gain * g[n]
            gu[n] = np.conj(bv) * gv + np.conj(b) * lam_next
            lam_next = np.conj(av) * gv + np.conj(a) * lam_next
        return gu
