"""Configurable result summaries and plots for MQMAS optimal-control runs.

The module is intentionally independent of the native C++ backend.  It reads the
JSON/CSV files written by the optimiser and can therefore be used on any machine
that has NumPy and Matplotlib installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DT_KEY_RE = re.compile(r"^split_dt_(?P<value>[0-9]+(?:[p.][0-9]+)?)us$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MISSING = object()


class ReportError(RuntimeError):
    """Raised when report inputs are invalid or incomplete in strict mode."""


@dataclass
class CaseSpec:
    """Files and labels belonging to one optimisation case."""

    case_id: str
    label: str
    folder: Path
    result_file: Path | None = None
    excitation_file: Path | None = None
    conversion_file: Path | None = None
    excitation_delivered_file: Path | None = None
    conversion_delivered_file: Path | None = None
    validation_file: Path | None = None
    validation_key: str | None = None
    baseline_file: Path | None = None
    baseline_key: str | None = None
    baseline_metric: str = "metrics.coherent_amplitude"
    rotor_phase_file: Path | None = None
    rotor_phase_key: str | None = None
    rotor_phase_x: str = "shift_us"
    rotor_phase_y: str = "amplitude"
    benchmark_file: Path | None = None
    benchmark_key: str | None = None
    history_x: str = "iteration"
    history_y: str = "coherent_amplitude"
    plot_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSettings:
    """Global output and plotting settings."""

    root: Path
    summary_dir: Path
    plots_dir: Path
    summary_json: str = "summary.json"
    summary_csv: str = "summary.csv"
    summary_markdown: str = "summary.md"
    manifest_json: str = "report_manifest.json"
    dpi: int = 180
    plot_formats: tuple[str, ...] = ("png",)
    strict: bool = False
    make_pulse_plots: bool = True
    make_convergence_plot: bool = True
    make_baseline_plot: bool = True
    make_time_step_plot: bool = True
    make_rotor_phase_plot: bool = True
    markdown_columns: tuple[str, ...] = (
        "system",
        "training_amplitude",
        "validation_amplitude",
        "validation_incoherent_amplitude",
        "coherence_factor",
        "baseline_amplitude",
        "shaped_to_baseline_ratio",
        "time_step_relative_error",
        "split_vs_direct_subset_error",
        "optimisation_s",
        "objective_gradient_s",
    )


@dataclass
class ReportArtifacts:
    """Paths and warnings returned by :func:`generate_report`."""

    summary_rows: list[dict[str, Any]]
    files: list[Path]
    warnings: list[str]


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip()).strip("._")
    return cleaned or "case"


def _resolve(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_inside(folder: Path, value: str | Path | None, default: str | None) -> Path | None:
    if value is None:
        if default is None:
            return None
        return (folder / default).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = folder / path
    return path.resolve()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"Invalid JSON in {path}: {exc}") from exc


def _nested_get(data: Any, path: str | None, default: Any = _MISSING) -> Any:
    if path in (None, ""):
        return data
    current = data
    for token in str(path).split("."):
        if isinstance(current, Mapping) and token in current:
            current = current[token]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                current = current[int(token)]
                continue
            except (ValueError, IndexError):
                pass
        if default is not _MISSING:
            return default
        raise KeyError(path)
    return current


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dt_from_key(key: str) -> float | None:
    match = _DT_KEY_RE.match(key)
    if not match:
        return None
    return float(match.group("value").replace("p", "."))


def extract_time_step_series(validation: Mapping[str, Any]) -> list[tuple[float, str, Mapping[str, Any]]]:
    """Return ``(dt_us, key, metrics)`` triples sorted from fine to coarse."""

    series: list[tuple[float, str, Mapping[str, Any]]] = []
    for key, value in validation.items():
        dt_us = _dt_from_key(str(key))
        if dt_us is not None and isinstance(value, Mapping):
            series.append((dt_us, str(key), value))
    series.sort(key=lambda item: item[0])
    return series


def _select_validation_metrics(
    validation: Mapping[str, Any], key: str | None
) -> tuple[Mapping[str, Any] | None, float | None, list[tuple[float, str, Mapping[str, Any]]]]:
    series = extract_time_step_series(validation)
    if key:
        selected = validation.get(key)
        if not isinstance(selected, Mapping):
            raise ReportError(f"Validation key {key!r} is missing or is not an object")
        return selected, _dt_from_key(key), series
    if series:
        dt_us, _, selected = series[0]
        return selected, dt_us, series
    metrics = validation.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics, None, series
    if "coherent_amplitude" in validation:
        return validation, None, series
    return None, None, series


def _compute_split_exact_error(validation: Mapping[str, Any]) -> float | None:
    direct = _nested_get(validation, "exact_subset_direct.coherent_amplitude", None)
    split = _nested_get(validation, "exact_subset_split.coherent_amplitude", None)
    direct_f = _as_float(direct)
    split_f = _as_float(split)
    if direct_f is None or split_f is None or direct_f == 0.0:
        return _as_float(validation.get("relative_split_vs_exact_subset"))
    return abs(split_f - direct_f) / abs(direct_f)


def _read_benchmark_file(path: Path) -> dict[str, dict[str, Any]]:
    """Read benchmark records from JSON, CSV, or ``key name value ...`` text."""

    if not path.exists():
        raise ReportError(f"Benchmark file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = _read_json(path)
        if isinstance(data, Mapping):
            return {
                str(key): dict(value) if isinstance(value, Mapping) else {"value": value}
                for key, value in data.items()
            }
        raise ReportError(f"Benchmark JSON must contain an object: {path}")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            records: dict[str, dict[str, Any]] = {}
            for index, row in enumerate(reader):
                key = row.get("id") or row.get("case") or row.get("key") or str(index)
                records[str(key)] = dict(row)
            return records

    records: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = shlex.split(line)
        if not tokens:
            continue
        key = tokens[0]
        record: dict[str, Any] = {}
        if (len(tokens) - 1) % 2 != 0:
            raise ReportError(
                f"Malformed benchmark line {line_number} in {path}; expected key/value pairs"
            )
        for name, value in zip(tokens[1::2], tokens[2::2]):
            try:
                number = float(value)
                record[name] = int(number) if number.is_integer() else number
            except ValueError:
                record[name] = value
        records[key] = record
    return records


def _column_map(names: Sequence[str]) -> dict[str, str]:
    return {str(name).lower(): str(name) for name in names}


def _find_column(names: Mapping[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        if alias.lower() in names:
            return names[alias.lower()]
    return None


def _scaled_column(
    array: np.ndarray,
    names: Mapping[str, str],
    aliases_and_scales: Sequence[tuple[str, float]],
) -> np.ndarray | None:
    for alias, scale in aliases_and_scales:
        column = _find_column(names, [alias])
        if column is not None:
            return np.asarray(array[column], dtype=float) * scale
    return None


def read_pulse_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a pulse CSV using common column-name aliases.

    The returned arrays always use microseconds for time, hertz for Cartesian RF
    and amplitude, and radians for phase.  Amplitude and phase are calculated
    from ``ux``/``uy`` if they are absent from the file.
    """

    if not path.exists():
        raise ReportError(f"Pulse CSV not found: {path}")
    try:
        raw = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    except Exception as exc:  # pragma: no cover - NumPy error text varies
        raise ReportError(f"Could not read pulse CSV {path}: {exc}") from exc
    if raw.dtype.names is None:
        raise ReportError(f"Pulse CSV has no named columns: {path}")
    raw = np.atleast_1d(raw)
    names = _column_map(raw.dtype.names)

    time_us = _scaled_column(
        raw,
        names,
        (
            ("time_us", 1.0),
            ("t_us", 1.0),
            ("time_microseconds", 1.0),
            ("time_s", 1e6),
            ("t_s", 1e6),
        ),
    )
    ux_hz = _scaled_column(
        raw,
        names,
        (("ux_hz", 1.0), ("u_x_hz", 1.0), ("x_hz", 1.0), ("ux_khz", 1e3), ("ux_mhz", 1e6)),
    )
    uy_hz = _scaled_column(
        raw,
        names,
        (("uy_hz", 1.0), ("u_y_hz", 1.0), ("y_hz", 1.0), ("uy_khz", 1e3), ("uy_mhz", 1e6)),
    )
    if time_us is None or ux_hz is None or uy_hz is None:
        raise ReportError(
            f"Pulse CSV {path} must contain time and Cartesian RF columns. "
            "Recognised examples: time_us, ux_hz, uy_hz."
        )

    amplitude_hz = _scaled_column(
        raw,
        names,
        (("amplitude_hz", 1.0), ("amplitude_khz", 1e3), ("amplitude_mhz", 1e6), ("amp_hz", 1.0)),
    )
    if amplitude_hz is None:
        amplitude_hz = np.hypot(ux_hz, uy_hz)

    phase = _scaled_column(raw, names, (("phase_rad", 1.0), ("phase", 1.0), ("phase_deg", np.pi / 180.0)))
    if phase is None:
        phase = np.arctan2(uy_hz, ux_hz)

    if not (len(time_us) == len(ux_hz) == len(uy_hz) == len(amplitude_hz) == len(phase)):
        raise ReportError(f"Pulse CSV columns have inconsistent lengths: {path}")
    return {
        "time_us": time_us,
        "ux_hz": ux_hz,
        "uy_hz": uy_hz,
        "amplitude_hz": amplitude_hz,
        "phase_rad": phase,
    }


def _save_figure(
    fig: plt.Figure,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    outputs: list[Path] = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        extension = extension.lower().lstrip(".")
        output = stem.with_suffix(f".{extension}")
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension in {"png", "jpg", "jpeg", "tif", "tiff", "webp"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(output, **save_kwargs)
        outputs.append(output)
    plt.close(fig)
    return outputs


def _plot_pulse_pair(
    case: CaseSpec,
    settings: ReportSettings,
) -> list[Path]:
    if case.excitation_file is None or case.conversion_file is None:
        return []
    excitation = read_pulse_csv(case.excitation_file)
    conversion = read_pulse_csv(case.conversion_file)
    delivered: list[dict[str, np.ndarray] | None] = [None, None]
    if case.excitation_delivered_file and case.excitation_delivered_file.exists():
        delivered[0] = read_pulse_csv(case.excitation_delivered_file)
    if case.conversion_delivered_file and case.conversion_delivered_file.exists():
        delivered[1] = read_pulse_csv(case.conversion_delivered_file)

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 8.0), sharex="col", squeeze=False)
    for column, (pulse, delivered_pulse, pulse_name) in enumerate(
        zip((excitation, conversion), delivered, ("Excitation", "Conversion"))
    ):
        axes[0, column].plot(pulse["time_us"], pulse["ux_hz"] / 1e3, label="ux command")
        axes[0, column].plot(pulse["time_us"], pulse["uy_hz"] / 1e3, label="uy command")
        if delivered_pulse is not None:
            axes[0, column].plot(
                delivered_pulse["time_us"], delivered_pulse["ux_hz"] / 1e3,
                linestyle="--", label="ux delivered"
            )
            axes[0, column].plot(
                delivered_pulse["time_us"], delivered_pulse["uy_hz"] / 1e3,
                linestyle="--", label="uy delivered"
            )
        axes[0, column].set_title(pulse_name)
        axes[0, column].set_ylabel("Cartesian RF (kHz)")
        axes[0, column].grid(True, alpha=0.25)
        axes[0, column].legend(loc="best")

        axes[1, column].plot(pulse["time_us"], pulse["amplitude_hz"] / 1e3, label="command")
        max_amplitude = float(np.max(pulse["amplitude_hz"]))
        if delivered_pulse is not None:
            axes[1, column].plot(
                delivered_pulse["time_us"], delivered_pulse["amplitude_hz"] / 1e3,
                linestyle="--", label="delivered"
            )
            max_amplitude = max(max_amplitude, float(np.max(delivered_pulse["amplitude_hz"])))
            axes[1, column].legend(loc="best")
        axes[1, column].set_ylabel("Amplitude (kHz)")
        axes[1, column].set_ylim(0.0, 1.06 * max(max_amplitude / 1e3, 1e-12))
        axes[1, column].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[1, column].grid(True, alpha=0.25)

        axes[2, column].plot(pulse["time_us"], np.unwrap(pulse["phase_rad"]), label="command")
        if delivered_pulse is not None:
            axes[2, column].plot(
                delivered_pulse["time_us"], np.unwrap(delivered_pulse["phase_rad"]),
                linestyle="--", label="delivered"
            )
            axes[2, column].legend(loc="best")
        axes[2, column].set_xlabel("Time (microseconds)")
        axes[2, column].set_ylabel("Unwrapped phase (rad)")
        axes[2, column].grid(True, alpha=0.25)

    fig.suptitle(case.label)
    fig.tight_layout()
    filename = case.plot_name or f"{_safe_name(case.case_id)}_pulse_pair"
    stem = settings.plots_dir / Path(filename).stem
    return _save_figure(fig, stem, settings.plot_formats, settings.dpi)


def _normalise_case_config(root: Path, raw: Mapping[str, Any]) -> CaseSpec:
    case_id = str(raw.get("id") or raw.get("case_id") or Path(str(raw.get("folder", "case"))).name)
    label = str(raw.get("label") or raw.get("name") or case_id)
    folder = _resolve(root, raw.get("folder", "."))
    assert folder is not None

    baseline = raw.get("baseline")
    baseline_file: Path | None = None
    baseline_key: str | None = None
    baseline_metric = "metrics.coherent_amplitude"
    if isinstance(baseline, str):
        baseline_file = _resolve(root, baseline)
    elif isinstance(baseline, Mapping):
        baseline_file = _resolve(root, baseline.get("file"))
        baseline_key = baseline.get("key")
        baseline_metric = str(baseline.get("metric", baseline_metric))

    rotor_phase = raw.get("rotor_phase")
    rotor_phase_file: Path | None = None
    rotor_phase_key: str | None = None
    rotor_x = "shift_us"
    rotor_y = "amplitude"
    if isinstance(rotor_phase, str):
        rotor_phase_file = _resolve(root, rotor_phase)
    elif isinstance(rotor_phase, Mapping):
        rotor_phase_file = _resolve(root, rotor_phase.get("file"))
        rotor_phase_key = rotor_phase.get("key")
        rotor_x = str(rotor_phase.get("x", rotor_x))
        rotor_y = str(rotor_phase.get("y", rotor_y))

    benchmark = raw.get("benchmark")
    benchmark_file: Path | None = None
    benchmark_key: str | None = raw.get("benchmark_key")
    if isinstance(benchmark, str):
        benchmark_file = _resolve(root, benchmark)
    elif isinstance(benchmark, Mapping):
        benchmark_file = _resolve(root, benchmark.get("file"))
        benchmark_key = benchmark.get("key", benchmark_key)

    validation = raw.get("validation")
    validation_file: Path | None = None
    validation_key: str | None = raw.get("validation_key")
    if isinstance(validation, str):
        validation_file = _resolve(root, validation)
    elif isinstance(validation, Mapping):
        validation_file = _resolve(root, validation.get("file"))
        validation_key = validation.get("key", validation_key)

    return CaseSpec(
        case_id=case_id,
        label=label,
        folder=folder,
        result_file=_resolve_inside(folder, raw.get("result", raw.get("result_file")), "result.json"),
        excitation_file=_resolve_inside(folder, raw.get("excitation", raw.get("excitation_file")), "excitation_pixels.csv"),
        conversion_file=_resolve_inside(folder, raw.get("conversion", raw.get("conversion_file")), "conversion_pixels.csv"),
        excitation_delivered_file=_resolve_inside(
            folder, raw.get("excitation_delivered", raw.get("excitation_delivered_file")), None
        ),
        conversion_delivered_file=_resolve_inside(
            folder, raw.get("conversion_delivered", raw.get("conversion_delivered_file")), None
        ),
        validation_file=validation_file,
        validation_key=validation_key,
        baseline_file=baseline_file,
        baseline_key=baseline_key,
        baseline_metric=baseline_metric,
        rotor_phase_file=rotor_phase_file,
        rotor_phase_key=rotor_phase_key,
        rotor_phase_x=rotor_x,
        rotor_phase_y=rotor_y,
        benchmark_file=benchmark_file,
        benchmark_key=benchmark_key,
        history_x=str(raw.get("history_x", "iteration")),
        history_y=str(raw.get("history_y", "coherent_amplitude")),
        plot_name=raw.get("plot_name"),
        metadata=dict(raw.get("metadata", {})),
    )


def load_report_config(path: Path) -> tuple[ReportSettings, list[CaseSpec]]:
    """Load a JSON report configuration.

    All paths are resolved relative to ``root``.  A relative ``root`` is itself
    resolved relative to the configuration file.
    """

    path = path.expanduser().resolve()
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        raise ReportError(f"Top-level configuration must be an object: {path}")
    root = _resolve(path.parent, raw.get("root", "."))
    assert root is not None
    output_dir = _resolve(root, raw.get("output_dir"))
    if output_dir is not None:
        default_summary_dir = output_dir
        default_plots_dir = output_dir / "plots"
    else:
        default_summary_dir = root / "results"
        default_plots_dir = root / "plots"
    summary_dir = _resolve(root, raw.get("summary_dir")) or default_summary_dir
    plots_dir = _resolve(root, raw.get("plots_dir")) or default_plots_dir

    plots = raw.get("plots", {})
    if not isinstance(plots, Mapping):
        plots = {}
    summary = raw.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}

    formats_raw = plots.get("formats", raw.get("plot_formats", ["png"]))
    if isinstance(formats_raw, str):
        formats = (formats_raw,)
    else:
        formats = tuple(str(item) for item in formats_raw)
    if not formats:
        formats = ("png",)

    settings = ReportSettings(
        root=root,
        summary_dir=summary_dir,
        plots_dir=plots_dir,
        summary_json=str(summary.get("json", "summary.json")),
        summary_csv=str(summary.get("csv", "summary.csv")),
        summary_markdown=str(summary.get("markdown", "summary.md")),
        manifest_json=str(summary.get("manifest", "report_manifest.json")),
        dpi=int(plots.get("dpi", raw.get("dpi", 180))),
        plot_formats=formats,
        strict=bool(raw.get("strict", False)),
        make_pulse_plots=bool(plots.get("pulse_pairs", True)),
        make_convergence_plot=bool(plots.get("convergence", True)),
        make_baseline_plot=bool(plots.get("validation_vs_baseline", True)),
        make_time_step_plot=bool(plots.get("time_step_convergence", True)),
        make_rotor_phase_plot=bool(plots.get("rotor_phase_sensitivity", True)),
        markdown_columns=tuple(summary.get("markdown_columns", ReportSettings.markdown_columns)),
    )

    global_benchmark = raw.get("benchmark")
    cases_raw = raw.get("cases", [])
    if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, (str, bytes, bytearray)):
        raise ReportError("Configuration key 'cases' must be a list")
    cases: list[CaseSpec] = []
    for item in cases_raw:
        if not isinstance(item, Mapping):
            raise ReportError("Each case configuration must be an object")
        item_copy = dict(item)
        if "benchmark" not in item_copy and global_benchmark is not None:
            item_copy["benchmark"] = global_benchmark
        cases.append(_normalise_case_config(root, item_copy))
    if not cases:
        raise ReportError(f"No cases were defined in {path}")
    return settings, cases


def _case_warning(warnings: list[str], case: CaseSpec, message: str) -> None:
    warnings.append(f"[{case.case_id}] {message}")


def _load_optional_json(
    path: Path | None,
    case: CaseSpec,
    settings: ReportSettings,
    warnings: list[str],
    description: str,
) -> Any | None:
    if path is None:
        return None
    if not path.exists():
        message = f"{description} file not found: {path}"
        if settings.strict:
            raise ReportError(f"[{case.case_id}] {message}")
        _case_warning(warnings, case, message)
        return None
    return _read_json(path)


def _extract_baseline(case: CaseSpec, data: Any) -> float | None:
    if data is None:
        return None
    selected = _nested_get(data, case.baseline_key, None) if case.baseline_key else data
    if selected is None:
        return None
    value = _nested_get(selected, case.baseline_metric, None)
    if value is None and isinstance(selected, Mapping):
        value = selected.get("coherent_amplitude")
        if value is None and isinstance(selected.get("metrics"), Mapping):
            value = selected["metrics"].get("coherent_amplitude")
    return _as_float(value)


def _load_benchmark_for_case(
    case: CaseSpec,
    cache: dict[Path, dict[str, dict[str, Any]]],
    settings: ReportSettings,
    warnings: list[str],
) -> dict[str, Any] | None:
    if case.benchmark_file is None:
        return None
    try:
        if case.benchmark_file not in cache:
            cache[case.benchmark_file] = _read_benchmark_file(case.benchmark_file)
        records = cache[case.benchmark_file]
    except ReportError as exc:
        if settings.strict:
            raise
        _case_warning(warnings, case, str(exc))
        return None
    key = case.benchmark_key or case.case_id
    record = records.get(key)
    if record is None:
        message = f"Benchmark key {key!r} was not found in {case.benchmark_file}"
        if settings.strict:
            raise ReportError(f"[{case.case_id}] {message}")
        _case_warning(warnings, case, message)
    return record


def _build_summary_row(
    case: CaseSpec,
    result: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    baseline_amplitude: float | None,
    benchmark: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[tuple[float, str, Mapping[str, Any]]]]:
    row: dict[str, Any] = {
        "id": case.case_id,
        "system": case.metadata.get("system", case.label),
    }
    row.update(case.metadata)
    series: list[tuple[float, str, Mapping[str, Any]]] = []

    if result is not None:
        metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), Mapping) else {}
        grid = result.get("grid", {}) if isinstance(result.get("grid"), Mapping) else {}
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), Mapping) else {}
        row.update(
            {
                "success": result.get("success"),
                "iterations": result.get("nit"),
                "function_evaluations": result.get("nfev"),
                "training_amplitude": _as_float(metrics.get("coherent_amplitude")),
                "training_incoherent_amplitude": _as_float(metrics.get("incoherent_amplitude")),
                "training_coherence_factor": _as_float(metrics.get("coherence_factor")),
                "excitation_us": _as_float(grid.get("excitation_duration_s")),
                "conversion_us": _as_float(grid.get("conversion_duration_s")),
                "pixel_dt_us": _as_float(grid.get("pixel_dt_s")),
                "internal_dt_us": _as_float(grid.get("internal_dt_s")),
                "max_nutation_khz": _as_float(result.get("max_nutation_hz")),
                "optimisation_s": _as_float(metadata.get("training_elapsed_s")),
            }
        )
        for field_name in ("excitation_us", "conversion_us", "pixel_dt_us", "internal_dt_us"):
            if row[field_name] is not None:
                row[field_name] *= 1e6
        if row["max_nutation_khz"] is not None:
            row["max_nutation_khz"] /= 1e3
        for key in ("training_orientations", "training_orientations_per_site", "training_sites", "seed_index"):
            if key in metadata and key not in row:
                row[key] = metadata[key]

    if validation is not None:
        selected, selected_dt_us, series = _select_validation_metrics(validation, case.validation_key)
        if selected is not None:
            row["validation_dt_us"] = selected_dt_us
            row["validation_amplitude"] = _as_float(selected.get("coherent_amplitude"))
            row["validation_incoherent_amplitude"] = _as_float(selected.get("incoherent_amplitude"))
            row["coherence_factor"] = _as_float(selected.get("coherence_factor"))
        row["validation_orientations"] = validation.get("dense_powder_orientations")
        row["validation_ensemble_members"] = validation.get("ensemble_members")
        row["split_vs_direct_subset_error"] = _compute_split_exact_error(validation)
        if len(series) >= 2:
            fine_dt, _, fine_metrics = series[0]
            coarse_dt, _, coarse_metrics = series[-1]
            fine = _as_float(fine_metrics.get("coherent_amplitude"))
            coarse = _as_float(coarse_metrics.get("coherent_amplitude"))
            row["time_step_fine_us"] = fine_dt
            row["time_step_coarse_us"] = coarse_dt
            if fine is not None and coarse is not None and fine != 0.0:
                row["time_step_relative_error"] = abs(coarse - fine) / abs(fine)

    row["baseline_amplitude"] = baseline_amplitude
    validation_amplitude = _as_float(row.get("validation_amplitude"))
    if validation_amplitude is None:
        validation_amplitude = _as_float(row.get("training_amplitude"))
    if baseline_amplitude not in (None, 0.0) and validation_amplitude is not None:
        row["shaped_to_baseline_ratio"] = validation_amplitude / baseline_amplitude

    if benchmark is not None:
        row["objective_gradient_s"] = _as_float(
            benchmark.get("objective_gradient_s", benchmark.get("grad"))
        )
        row["preprocessing_s"] = _as_float(benchmark.get("preprocessing_s", benchmark.get("prep")))
        row["benchmark_ensemble_members"] = benchmark.get("ensemble_members", benchmark.get("ens"))
        row["benchmark_steps"] = benchmark.get("steps")

    return row, series


def _ordered_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "id", "system", "spin", "mq_order", "success", "iterations", "function_evaluations",
        "excitation_us", "conversion_us", "pixel_dt_us", "internal_dt_us", "max_nutation_khz",
        "training_orientations", "training_orientations_per_site", "training_sites",
        "validation_orientations", "validation_ensemble_members",
        "training_amplitude", "training_incoherent_amplitude", "training_coherence_factor",
        "validation_dt_us", "validation_amplitude", "validation_incoherent_amplitude", "coherence_factor",
        "baseline_amplitude", "shaped_to_baseline_ratio",
        "time_step_coarse_us", "time_step_fine_us", "time_step_relative_error",
        "split_vs_direct_subset_error", "optimisation_s", "objective_gradient_s", "preprocessing_s",
        "benchmark_ensemble_members", "benchmark_steps", "seed_index",
    ]
    present = {key for row in rows for key in row}
    ordered = [key for key in preferred if key in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _format_markdown(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        magnitude = abs(value)
        if magnitude < 1e-4 or magnitude >= 1e4:
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _write_summaries(
    rows: list[dict[str, Any]], settings: ReportSettings
) -> list[Path]:
    settings.summary_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    json_path = settings.summary_dir / settings.summary_json
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=_json_compatible) + "\n",
        encoding="utf-8",
    )
    outputs.append(json_path)

    csv_path = settings.summary_dir / settings.summary_csv
    fieldnames = _ordered_fieldnames(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(csv_path)

    markdown_path = settings.summary_dir / settings.summary_markdown
    columns = [column for column in settings.markdown_columns if any(column in row for row in rows)]
    if not columns:
        columns = fieldnames
    lines = ["# MQMAS optimal-control result summary", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_format_markdown(row.get(column)) for column in columns) + " |")
    else:
        lines.append("No cases were available.")
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    outputs.append(markdown_path)
    return outputs


def _plot_convergence(
    case_data: Sequence[tuple[CaseSpec, Mapping[str, Any] | None]], settings: ReportSettings
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    plotted = 0
    for case, result in case_data:
        if result is None:
            continue
        history = result.get("history")
        if not isinstance(history, Sequence):
            continue
        x_values: list[float] = []
        y_values: list[float] = []
        for index, item in enumerate(history):
            if not isinstance(item, Mapping):
                continue
            x = _as_float(item.get(case.history_x, index + 1))
            y = _as_float(item.get(case.history_y))
            if x is not None and y is not None:
                x_values.append(x)
                y_values.append(y)
        if x_values:
            axis.plot(x_values, y_values, label=case.label)
            plotted += 1
    if not plotted:
        plt.close(fig)
        return []
    axis.set_xlabel("Optimiser iteration")
    axis.set_ylabel("Training coherent pathway amplitude")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return _save_figure(fig, settings.plots_dir / "convergence", settings.plot_formats, settings.dpi)


def _plot_validation_vs_baseline(
    rows: Sequence[Mapping[str, Any]], settings: ReportSettings
) -> list[Path]:
    usable = [
        row for row in rows
        if _as_float(row.get("baseline_amplitude")) is not None
        and (_as_float(row.get("validation_amplitude")) is not None or _as_float(row.get("training_amplitude")) is not None)
    ]
    if not usable:
        return []
    labels = [str(row.get("system", row.get("id", "case"))) for row in usable]
    baseline = [_as_float(row.get("baseline_amplitude")) or 0.0 for row in usable]
    shaped = [
        _as_float(row.get("validation_amplitude"))
        if _as_float(row.get("validation_amplitude")) is not None
        else (_as_float(row.get("training_amplitude")) or 0.0)
        for row in usable
    ]
    x_positions = np.arange(len(usable))
    width = min(0.36, 0.8 / max(2, len(usable))) if len(usable) > 4 else 0.36
    fig, axis = plt.subplots(figsize=(max(7.4, 1.4 * len(usable) + 3.0), 4.5))
    axis.bar(x_positions - width / 2.0, baseline, width, label="Baseline")
    axis.bar(x_positions + width / 2.0, shaped, width, label="Optimised shaped pair")
    axis.set_xticks(x_positions, labels, rotation=20 if len(usable) > 3 else 0, ha="right" if len(usable) > 3 else "center")
    axis.set_ylabel("Coherent pathway amplitude")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return _save_figure(
        fig, settings.plots_dir / "validation_vs_baseline", settings.plot_formats, settings.dpi
    )


def _plot_time_step_convergence(
    series_by_case: Sequence[tuple[CaseSpec, list[tuple[float, str, Mapping[str, Any]]]]],
    settings: ReportSettings,
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    plotted = 0
    for case, series in series_by_case:
        points = [
            (dt_us, _as_float(metrics.get("coherent_amplitude")))
            for dt_us, _, metrics in series
        ]
        points = [(dt, value) for dt, value in points if value is not None]
        if len(points) < 2:
            continue
        points.sort(reverse=True)
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=case.label)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return []
    axis.invert_xaxis()
    axis.set_xlabel("Internal step (microseconds)")
    axis.set_ylabel("Coherent pathway amplitude")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return _save_figure(
        fig, settings.plots_dir / "time_step_convergence", settings.plot_formats, settings.dpi
    )


def _plot_rotor_phase(
    rotor_data: Sequence[tuple[CaseSpec, Mapping[str, Any] | None]], settings: ReportSettings
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    plotted = 0
    for case, data in rotor_data:
        if data is None:
            continue
        selected = _nested_get(data, case.rotor_phase_key, None) if case.rotor_phase_key else data
        if not isinstance(selected, Mapping):
            continue
        x_values = selected.get(case.rotor_phase_x)
        y_values = selected.get(case.rotor_phase_y)
        if not isinstance(x_values, Sequence) or not isinstance(y_values, Sequence):
            continue
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        if x_array.size == 0 or x_array.size != y_array.size:
            continue
        axis.plot(x_array, y_array, label=case.label)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return []
    axis.set_xlabel("Conversion start-time shift (microseconds)")
    axis.set_ylabel("Coherent pathway amplitude")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return _save_figure(
        fig, settings.plots_dir / "rotor_phase_sensitivity", settings.plot_formats, settings.dpi
    )


def generate_report(settings: ReportSettings, cases: Sequence[CaseSpec]) -> ReportArtifacts:
    """Generate summary files and all applicable plots for ``cases``."""

    settings.summary_dir.mkdir(parents=True, exist_ok=True)
    settings.plots_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    files: list[Path] = []
    rows: list[dict[str, Any]] = []
    case_results: list[tuple[CaseSpec, Mapping[str, Any] | None]] = []
    series_by_case: list[tuple[CaseSpec, list[tuple[float, str, Mapping[str, Any]]]]] = []
    rotor_data: list[tuple[CaseSpec, Mapping[str, Any] | None]] = []
    benchmark_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    for case in cases:
        result_raw = _load_optional_json(case.result_file, case, settings, warnings, "result")
        result = result_raw if isinstance(result_raw, Mapping) else None
        if result_raw is not None and result is None:
            raise ReportError(f"[{case.case_id}] Result JSON must contain an object")

        validation_raw = _load_optional_json(case.validation_file, case, settings, warnings, "validation")
        validation = validation_raw if isinstance(validation_raw, Mapping) else None
        if validation_raw is not None and validation is None:
            raise ReportError(f"[{case.case_id}] Validation JSON must contain an object")

        baseline_raw = _load_optional_json(case.baseline_file, case, settings, warnings, "baseline")
        baseline_amplitude = _extract_baseline(case, baseline_raw)
        if case.baseline_file is not None and baseline_amplitude is None:
            _case_warning(
                warnings,
                case,
                f"Could not extract baseline metric {case.baseline_metric!r} from {case.baseline_file}",
            )

        benchmark = _load_benchmark_for_case(case, benchmark_cache, settings, warnings)
        row, series = _build_summary_row(case, result, validation, baseline_amplitude, benchmark)
        rows.append(row)
        case_results.append((case, result))
        series_by_case.append((case, series))

        rotor_raw = _load_optional_json(case.rotor_phase_file, case, settings, warnings, "rotor-phase")
        rotor_data.append((case, rotor_raw if isinstance(rotor_raw, Mapping) else None))

        if settings.make_pulse_plots:
            required = [case.excitation_file, case.conversion_file]
            missing = [path for path in required if path is None or not path.exists()]
            if missing:
                message = "Pulse-pair plot skipped; missing " + ", ".join(str(path) for path in missing)
                if settings.strict:
                    raise ReportError(f"[{case.case_id}] {message}")
                _case_warning(warnings, case, message)
            else:
                files.extend(_plot_pulse_pair(case, settings))

    files.extend(_write_summaries(rows, settings))
    if settings.make_convergence_plot:
        files.extend(_plot_convergence(case_results, settings))
    if settings.make_baseline_plot:
        files.extend(_plot_validation_vs_baseline(rows, settings))
    if settings.make_time_step_plot:
        files.extend(_plot_time_step_convergence(series_by_case, settings))
    if settings.make_rotor_phase_plot:
        files.extend(_plot_rotor_phase(rotor_data, settings))

    manifest_path = settings.summary_dir / settings.manifest_json
    manifest = {
        "root": str(settings.root),
        "summary_dir": str(settings.summary_dir),
        "plots_dir": str(settings.plots_dir),
        "cases": [case.case_id for case in cases],
        "files": [str(path) for path in files],
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append(manifest_path)
    return ReportArtifacts(summary_rows=rows, files=files, warnings=warnings)


def _parse_id_path(value: str) -> tuple[str, str]:
    if "=" in value:
        case_id, path = value.split("=", 1)
        if not case_id or not path:
            raise argparse.ArgumentTypeError("Expected CASE_ID=PATH")
        return case_id, path
    path = value
    return Path(path).name, path


def _parse_id_value(items: Sequence[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ReportError(f"Expected CASE_ID=VALUE, got {item!r}")
        case_id, value = item.split("=", 1)
        result[case_id] = value
    return result


def _parse_file_key(value: str) -> tuple[str, str | None]:
    if "#" in value:
        file_name, key = value.rsplit("#", 1)
        return file_name, key or None
    return value, None


def _discover_cases(root: Path) -> list[CaseSpec]:
    results_dir = root / "results"
    cases: list[CaseSpec] = []
    for result_path in sorted(results_dir.glob("*/result.json")):
        folder = result_path.parent.resolve()
        case_id = folder.name
        cases.append(
            CaseSpec(
                case_id=case_id,
                label=case_id,
                folder=folder,
                result_file=result_path.resolve(),
                excitation_file=(folder / "excitation_pixels.csv").resolve(),
                conversion_file=(folder / "conversion_pixels.csv").resolve(),
            )
        )
    return cases


def _settings_and_cases_from_args(args: argparse.Namespace) -> tuple[ReportSettings, list[CaseSpec]]:
    if args.config:
        settings, cases = load_report_config(args.config)
        if args.output_dir:
            output = args.output_dir.expanduser().resolve()
            settings.summary_dir = output
            settings.plots_dir = output / "plots"
        if args.dpi is not None:
            settings.dpi = args.dpi
        if args.plot_format:
            settings.plot_formats = tuple(args.plot_format)
        if args.strict:
            settings.strict = True
        if args.no_plots:
            settings.make_pulse_plots = False
            settings.make_convergence_plot = False
            settings.make_baseline_plot = False
            settings.make_time_step_plot = False
            settings.make_rotor_phase_plot = False
        return settings, cases

    root = args.root.expanduser().resolve()
    case_entries = [_parse_id_path(item) for item in args.case]
    labels = _parse_id_value(args.label)
    result_files = _parse_id_value(args.result_file)
    excitation_files = _parse_id_value(args.excitation_file)
    conversion_files = _parse_id_value(args.conversion_file)
    validation_files = _parse_id_value(args.validation_file)
    baseline_files = _parse_id_value(args.baseline_file)
    rotor_files = _parse_id_value(args.rotor_phase_file)
    benchmark_keys = _parse_id_value(args.benchmark_key)

    cases: list[CaseSpec] = []
    for case_id, folder_text in case_entries:
        folder = _resolve(root, folder_text)
        assert folder is not None
        baseline_file: Path | None = None
        baseline_key: str | None = None
        if case_id in baseline_files:
            file_text, baseline_key = _parse_file_key(baseline_files[case_id])
            baseline_file = _resolve(root, file_text)
        rotor_file: Path | None = None
        rotor_key: str | None = None
        if case_id in rotor_files:
            file_text, rotor_key = _parse_file_key(rotor_files[case_id])
            rotor_file = _resolve(root, file_text)
        cases.append(
            CaseSpec(
                case_id=case_id,
                label=labels.get(case_id, case_id),
                folder=folder,
                result_file=_resolve_inside(folder, result_files.get(case_id), "result.json"),
                excitation_file=_resolve_inside(folder, excitation_files.get(case_id), "excitation_pixels.csv"),
                conversion_file=_resolve_inside(folder, conversion_files.get(case_id), "conversion_pixels.csv"),
                validation_file=_resolve(root, validation_files.get(case_id)),
                baseline_file=baseline_file,
                baseline_key=baseline_key,
                rotor_phase_file=rotor_file,
                rotor_phase_key=rotor_key,
                benchmark_file=_resolve(root, args.benchmark_file),
                benchmark_key=benchmark_keys.get(case_id, case_id),
            )
        )
    if not cases:
        cases = _discover_cases(root)
    if not cases:
        raise ReportError(
            "No cases were supplied and no results/*/result.json files were discovered. "
            "Use --config or one or more --case CASE_ID=FOLDER arguments."
        )

    output = args.output_dir.expanduser().resolve() if args.output_dir else root / "generated_report"
    settings = ReportSettings(
        root=root,
        summary_dir=output,
        plots_dir=output / "plots",
        dpi=args.dpi or 180,
        plot_formats=tuple(args.plot_format or ["png"]),
        strict=args.strict,
        make_pulse_plots=not args.no_plots,
        make_convergence_plot=not args.no_plots,
        make_baseline_plot=not args.no_plots,
        make_time_step_plot=not args.no_plots,
        make_rotor_phase_plot=not args.no_plots,
    )
    return settings, cases


def example_config() -> dict[str, Any]:
    """Return an editable configuration matching the supplied example data."""

    return {
        "root": "..",
        "summary_dir": "results",
        "plots_dir": "plots",
        "summary": {
            "json": "summary.json",
            "csv": "summary.csv",
            "markdown": "summary.md",
            "manifest": "report_manifest.json",
        },
        "plots": {
            "formats": ["png"],
            "dpi": 180,
            "pulse_pairs": True,
            "convergence": True,
            "validation_vs_baseline": True,
            "time_step_convergence": True,
            "rotor_phase_sensitivity": True,
        },
        "benchmark": {"file": "results/benchmark.txt"},
        "cases": [
            {
                "id": "rb",
                "label": "Rb2SO4 87Rb",
                "folder": "results/rb2so4_10us_dt05",
                "result": "result.json",
                "excitation": "excitation_pixels.csv",
                "conversion": "conversion_pixels.csv",
                "validation": "results/rb_final_validation.json",
                "baseline": {
                    "file": "results/hard_pulse_final_validation.json",
                    "key": "rb",
                },
                "rotor_phase": {
                    "file": "results/rotor_phase_sensitivity.json",
                    "key": "rb",
                },
                "benchmark_key": "rb",
                "plot_name": "rb2so4_pulse_pair.png",
                "metadata": {
                    "system": "Rb2SO4 87Rb",
                    "spin": "3/2",
                    "mq_order": 3,
                    "training_orientations": 512,
                },
            },
            {
                "id": "al",
                "label": "AlPO4 27Al",
                "folder": "results/alpo4_10us_dense",
                "result": "result.json",
                "excitation": "excitation_pixels.csv",
                "conversion": "conversion_pixels.csv",
                "validation": "results/al_final_validation.json",
                "baseline": {
                    "file": "results/hard_pulse_final_validation.json",
                    "key": "al",
                },
                "rotor_phase": {
                    "file": "results/rotor_phase_sensitivity.json",
                    "key": "al",
                },
                "benchmark_key": "al",
                "plot_name": "alpo4_pulse_pair.png",
                "metadata": {
                    "system": "AlPO4 27Al",
                    "spin": "5/2",
                    "mq_order": 3,
                    "training_orientations": 512,
                },
            },
        ],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate configurable MQMAS result summaries and plots."
    )
    parser.add_argument("--config", type=Path, help="JSON report configuration")
    parser.add_argument("--write-example-config", type=Path, help="Write an editable example configuration and exit")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root used for direct CLI paths")
    parser.add_argument(
        "--case", action="append", default=[], metavar="CASE_ID=FOLDER",
        help="Add a result folder; may be repeated. CASE_ID= is optional.",
    )
    parser.add_argument("--label", action="append", metavar="CASE_ID=LABEL", help="Human-readable case label")
    parser.add_argument("--result-file", action="append", metavar="CASE_ID=FILE", help="Result JSON filename/path")
    parser.add_argument("--excitation-file", action="append", metavar="CASE_ID=FILE", help="Excitation pulse CSV filename/path")
    parser.add_argument("--conversion-file", action="append", metavar="CASE_ID=FILE", help="Conversion pulse CSV filename/path")
    parser.add_argument("--validation-file", action="append", metavar="CASE_ID=FILE", help="Validation JSON path")
    parser.add_argument(
        "--baseline-file", action="append", metavar="CASE_ID=FILE[#KEY]",
        help="Baseline JSON and optional object key",
    )
    parser.add_argument(
        "--rotor-phase-file", action="append", metavar="CASE_ID=FILE[#KEY]",
        help="Rotor-phase JSON and optional object key",
    )
    parser.add_argument("--benchmark-file", help="Shared benchmark JSON/CSV/text file")
    parser.add_argument("--benchmark-key", action="append", metavar="CASE_ID=KEY", help="Record key in the benchmark file")
    parser.add_argument("--output-dir", type=Path, help="Output directory; plots go into OUTPUT/plots")
    parser.add_argument("--plot-format", action="append", help="Plot extension, e.g. png or pdf; may be repeated")
    parser.add_argument("--dpi", type=int, help="Raster plot resolution")
    parser.add_argument("--strict", action="store_true", help="Treat missing optional inputs as errors")
    parser.add_argument("--no-plots", action="store_true", help="Generate summaries only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.write_example_config:
        output = args.write_example_config.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(example_config(), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote example configuration: {output}")
        return 0
    try:
        settings, cases = _settings_and_cases_from_args(args)
        artifacts = generate_report(settings, cases)
    except ReportError as exc:
        parser.error(str(exc))
        return 2

    print(f"Generated summaries for {len(artifacts.summary_rows)} case(s).")
    for path in artifacts.files:
        print(path)
    if artifacts.warnings:
        print("Warnings:", file=sys.stderr)
        for warning in artifacts.warnings:
            print(f"  - {warning}", file=sys.stderr)
    return 0


__all__ = [
    "CaseSpec",
    "ReportArtifacts",
    "ReportError",
    "ReportSettings",
    "example_config",
    "extract_time_step_series",
    "generate_report",
    "load_report_config",
    "main",
    "read_pulse_csv",
]
