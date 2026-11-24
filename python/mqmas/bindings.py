"""ctypes interface to the C++ low-rank MQMAS propagator/gradient core."""
from __future__ import annotations

from dataclasses import dataclass
import ctypes as ct
import os
import platform
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .model import MASDriftModel, make_ensemble_mapping

ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class PairMetrics:
    objective: float
    coherent_real: float
    coherent_imag: float
    coherent_amplitude: float
    incoherent_amplitude: float
    coherence_factor: float
    excitation_coherent: float
    conversion_coherent: float
    excitation_magnitude: float
    conversion_magnitude: float

    @classmethod
    def from_array(cls, a: ArrayF) -> "PairMetrics":
        return cls(*map(float, a[:10]))


class MQMASCore:
    """C++ context for a fixed MAS drift and ensemble."""

    def __init__(
        self,
        drift: MASDriftModel,
        *,
        mq_order: int = 3,
        b1_scales: Iterable[float] = (1.0,),
        b1_weights: Iterable[float] | None = None,
        library: str | Path | None = None,
    ) -> None:
        self.drift = drift
        self.dimension = drift.dimension
        if self.dimension not in (4, 6):
            raise ValueError("compiled core currently supports I=3/2 and I=5/2")
        root = Path(__file__).resolve().parents[2]
        library_path = self._resolve_library(root, library)
        try:
            # The exported C API uses the cdecl calling convention, so CDLL is
            # correct on Windows as well as on Unix-like systems.
            self.lib = ct.CDLL(str(library_path))
        except OSError as exc:
            python_bits = ct.sizeof(ct.c_void_p) * 8
            raise OSError(
                f"Failed to load MQMAS native library: {library_path}\n"
                f"Platform: {sys.platform}, machine: {platform.machine()}, "
                f"Python: {python_bits}-bit.\n"
                "Rebuild the library on this machine and make sure the DLL/"
                "shared-library architecture matches Python. On Windows use "
                "`cmake -S . -B build -A x64` followed by "
                "`cmake --build build --config Release --parallel`."
            ) from exc
        self._configure_api()
        weights, didx, bidx, b1 = make_ensemble_mapping(drift, b1_scales, b1_weights)
        self.weights, self.didx, self.bidx, self.b1 = weights, didx, bidx, b1
        epre = np.ascontiguousarray(drift.epre).view(np.float64)
        epost = np.ascontiguousarray(drift.epost).view(np.float64)
        create = getattr(self.lib, f"mqmas_create_d{self.dimension}")
        self.handle = create(
            drift.rotations.shape[0], drift.nphase, weights.size, b1.size,
            drift.dt_s, int(mq_order),
            epre.ctypes.data_as(ct.POINTER(ct.c_double)),
            epost.ctypes.data_as(ct.POINTER(ct.c_double)),
            weights.ctypes.data_as(ct.POINTER(ct.c_double)),
            didx.ctypes.data_as(ct.POINTER(ct.c_int)),
            bidx.ctypes.data_as(ct.POINTER(ct.c_int)),
            b1.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        if not self.handle:
            raise RuntimeError("C++ context creation failed")
        self.mq_order = mq_order

    @staticmethod
    def _resolve_library(
        root: Path, library: str | Path | None
    ) -> Path:
        """Resolve only a native library for the current operating system."""
        if library is not None:
            path = Path(library).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"native library does not exist: {path}")
            return path

        env_library = os.environ.get("MQMAS_CORE_LIBRARY")
        if env_library:
            path = Path(env_library).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    "MQMAS_CORE_LIBRARY points to a missing file: " f"{path}"
                )
            return path

        if sys.platform == "win32":
            names = ("mqmas_core.dll",)
        elif sys.platform == "darwin":
            names = ("libmqmas_core.dylib",)
        else:
            names = ("libmqmas_core.so",)

        build = root / "build"
        search_dirs = (
            build,
            build / "Release",
            build / "RelWithDebInfo",
            build / "Debug",
            build / "MinSizeRel",
        )
        candidates = [directory / name for directory in search_dirs for name in names]
        for path in candidates:
            if path.is_file():
                return path.resolve()

        checked = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not find the MQMAS native library for this platform. "
            "Build it with CMake first. Checked:\n  " + checked
        )

    def _configure_api(self) -> None:
        pdouble = ct.POINTER(ct.c_double)
        pint = ct.POINTER(ct.c_int)
        for d in (4, 6):
            create = getattr(self.lib, f"mqmas_create_d{d}")
            create.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_int,
                               pdouble, pdouble, pdouble, pint, pint, pdouble]
            create.restype = ct.c_void_p
            destroy = getattr(self.lib, f"mqmas_destroy_d{d}")
            destroy.argtypes = [ct.c_void_p]
            destroy.restype = None
            evaluate = getattr(self.lib, f"mqmas_eval_pair_d{d}")
            evaluate.argtypes = [ct.c_void_p,
                                 pdouble, pdouble, ct.c_int, ct.c_int,
                                 pdouble, pdouble, ct.c_int, ct.c_int,
                                 pdouble, pdouble, pdouble, pdouble, pdouble]
            evaluate.restype = ct.c_int

    def close(self) -> None:
        if getattr(self, "handle", None):
            getattr(self.lib, f"mqmas_destroy_d{self.dimension}")(self.handle)
            self.handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _control(a: ArrayF) -> ArrayF:
        a = np.ascontiguousarray(a, dtype=np.float64)
        if a.ndim != 1 or not np.all(np.isfinite(a)):
            raise ValueError("control arrays must be finite 1D float arrays")
        return a

    def evaluate(
        self,
        ex: ArrayF,
        ey: ArrayF,
        cx: ArrayF,
        cy: ArrayF,
        *,
        excitation_phase_index: int = 0,
        conversion_phase_index: int = 0,
        gradient: bool = True,
    ) -> tuple[PairMetrics, tuple[ArrayF, ArrayF, ArrayF, ArrayF] | None]:
        ex, ey, cx, cy = map(self._control, (ex, ey, cx, cy))
        if ex.shape != ey.shape or cx.shape != cy.shape:
            raise ValueError("x and y controls must have equal lengths")
        metrics = np.zeros(10, dtype=np.float64)
        if gradient:
            grads = tuple(np.zeros_like(a) for a in (ex, ey, cx, cy))
            ptrs = [g.ctypes.data_as(ct.POINTER(ct.c_double)) for g in grads]
        else:
            grads = None
            ptrs = [ct.POINTER(ct.c_double)()] * 4
        evaluate = getattr(self.lib, f"mqmas_eval_pair_d{self.dimension}")
        rc = evaluate(
            self.handle,
            ex.ctypes.data_as(ct.POINTER(ct.c_double)),
            ey.ctypes.data_as(ct.POINTER(ct.c_double)), ex.size, excitation_phase_index,
            cx.ctypes.data_as(ct.POINTER(ct.c_double)),
            cy.ctypes.data_as(ct.POINTER(ct.c_double)), cx.size, conversion_phase_index,
            metrics.ctypes.data_as(ct.POINTER(ct.c_double)), *ptrs,
        )
        if rc != 0:
            raise RuntimeError(f"C++ evaluation failed with status {rc}")
        return PairMetrics.from_array(metrics), grads
