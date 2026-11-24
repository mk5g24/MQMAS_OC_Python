# Configurable plots and result summaries

The reporting code is split into:

- `python/mqmas/reporting.py`: reusable Python API;
- `scripts/make_plots_and_summary.py`: command-line entry point;
- `scripts/report_config.json`: editable configuration for the supplied Rb and Al results.

The reporting code does not require the native C++ library. It only needs NumPy
and Matplotlib, so plots can still be regenerated before or independently of a
native build.

## Recommended configuration-file workflow

Edit `scripts/report_config.json` and run:

```bash
python scripts/make_plots_and_summary.py --config scripts/report_config.json
```

All paths are resolved relative to the configuration's `root`. In the supplied
configuration the file lives under `scripts/`, so `root` is `..`.

A case can use any folder and filenames:

```json
{
  "id": "my_case",
  "label": "My sample, 3Q",
  "folder": "results/run_2026_11_24",
  "result": "optimisation_output.json",
  "excitation": "excitation_shape.csv",
  "conversion": "conversion_shape.csv",
  "validation": "results/run_2026_11_24_validation.json",
  "baseline": {
    "file": "results/baselines.json",
    "key": "my_case",
    "metric": "metrics.coherent_amplitude"
  },
  "rotor_phase": {
    "file": "results/rotor_phase_scan.json",
    "key": "my_case"
  },
  "benchmark_key": "my_case",
  "metadata": {
    "system": "My sample",
    "spin": "3/2",
    "mq_order": 3
  }
}
```

The default names inside each case folder are:

```text
result.json
excitation_pixels.csv
conversion_pixels.csv
```

They only need to be specified when the files use different names.

Create a fresh editable configuration with:

```bash
python scripts/make_plots_and_summary.py \
  --write-example-config scripts/my_report_config.json
```

## Direct command-line workflow

For result folders that use the default filenames:

```bash
python scripts/make_plots_and_summary.py \
  --root . \
  --case rb=results/rb2so4_10us_dt05 \
  --case al=results/alpo4_10us_dense \
  --label "rb=Rb2SO4 87Rb" \
  --label "al=AlPO4 27Al" \
  --validation-file rb=results/rb_final_validation.json \
  --validation-file al=results/al_final_validation.json \
  --baseline-file "rb=results/hard_pulse_final_validation.json#rb" \
  --baseline-file "al=results/hard_pulse_final_validation.json#al" \
  --rotor-phase-file "rb=results/rotor_phase_sensitivity.json#rb" \
  --rotor-phase-file "al=results/rotor_phase_sensitivity.json#al" \
  --benchmark-file results/benchmark.txt \
  --benchmark-key rb=rb \
  --benchmark-key al=al \
  --output-dir generated_report
```

Quote values containing `#` in POSIX shells, because an unquoted `#` starts a
shell comment.

Override nonstandard filenames with:

```bash
--result-file my_case=optimisation_output.json
--excitation-file my_case=excitation_shape.csv
--conversion-file my_case=conversion_shape.csv
```

With no `--config` or `--case`, the script auto-discovers
`results/*/result.json`. Optional validation, baseline, timing, and rotor-phase
files still need to be supplied explicitly or through a configuration.

## Outputs

The generator writes:

- `summary.json`: complete machine-readable rows;
- `summary.csv`: the union of all available summary columns;
- `summary.md`: a compact table for reports or GitHub;
- `report_manifest.json`: generated paths and nonfatal warnings;
- one pulse-pair plot per case;
- a combined convergence plot when histories exist;
- a shaped-versus-baseline plot when baselines exist;
- a time-step convergence plot when at least two `split_dt_*us` records exist;
- a rotor-phase plot when rotor-phase data are supplied.

Missing optional files cause the corresponding metric or plot to be omitted.
Use `--strict` to convert such omissions into errors.

## Automatic validation parsing

Validation entries are discovered from keys such as:

```text
split_dt_0p1us
split_dt_0p05us
split_dt_0p025us
```

The smallest time step is used for the reported validation amplitude unless a
case sets an explicit validation key. The relative time-step error is computed
between the coarsest and finest available steps. Split-versus-direct error is
read from `relative_split_vs_exact_subset` or calculated from
`exact_subset_split.coherent_amplitude` and
`exact_subset_direct.coherent_amplitude`.

## Accepted pulse CSV columns

The canonical columns are:

```text
time_us,ux_hz,uy_hz,amplitude_hz,phase_rad
```

Several aliases are accepted, including `time_s`, `t_us`, `ux_khz`, `uy_khz`,
`phase_deg`, and others. If amplitude or phase is absent, it is reconstructed
from the Cartesian components.

## Python API

```python
from pathlib import Path
from mqmas.reporting import load_report_config, generate_report

settings, cases = load_report_config(Path("scripts/report_config.json"))
artifacts = generate_report(settings, cases)

print(artifacts.summary_rows)
print(artifacts.files)
print(artifacts.warnings)
```
