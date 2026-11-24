from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mqmas.reporting import extract_time_step_series, generate_report, load_report_config


def _write_pulse(path: Path) -> None:
    path.write_text(
        "t_s,ux_khz,uy_khz\n"
        "0.0,10.0,0.0\n"
        "1e-6,0.0,10.0\n",
        encoding="utf-8",
    )


def test_time_step_key_discovery() -> None:
    validation = {
        "split_dt_0p1us": {"coherent_amplitude": 0.20},
        "split_dt_0.025us": {"coherent_amplitude": 0.19},
        "unrelated": {},
    }
    series = extract_time_step_series(validation)
    assert [point[0] for point in series] == [0.025, 0.1]


def test_configurable_files_and_outputs(tmp_path: Path) -> None:
    run_folder = tmp_path / "arbitrary_run_folder"
    run_folder.mkdir()
    result_file = run_folder / "optimisation_output.json"
    excitation_file = run_folder / "my_excitation.csv"
    conversion_file = run_folder / "my_conversion.csv"
    validation_file = tmp_path / "held_out_metrics.json"
    baseline_file = tmp_path / "reference_pulses.json"
    rotor_file = tmp_path / "timing_scan.json"
    benchmark_file = tmp_path / "timings.txt"

    result_file.write_text(
        json.dumps(
            {
                "success": True,
                "nit": 12,
                "nfev": 15,
                "metrics": {
                    "coherent_amplitude": 0.25,
                    "incoherent_amplitude": 0.30,
                    "coherence_factor": 0.8333333333,
                },
                "grid": {
                    "excitation_duration_s": 8e-6,
                    "conversion_duration_s": 7e-6,
                    "pixel_dt_s": 0.2e-6,
                    "internal_dt_s": 0.05e-6,
                },
                "max_nutation_hz": 90e3,
                "history": [
                    {"iteration": 1, "coherent_amplitude": 0.1},
                    {"iteration": 2, "coherent_amplitude": 0.25},
                ],
                "metadata": {"training_elapsed_s": 3.5},
            }
        ),
        encoding="utf-8",
    )
    _write_pulse(excitation_file)
    _write_pulse(conversion_file)
    validation_file.write_text(
        json.dumps(
            {
                "dense_powder_orientations": 128,
                "ensemble_members": 384,
                "split_dt_0p1us": {
                    "coherent_amplitude": 0.241,
                    "incoherent_amplitude": 0.29,
                    "coherence_factor": 0.831,
                },
                "split_dt_0p025us": {
                    "coherent_amplitude": 0.240,
                    "incoherent_amplitude": 0.288,
                    "coherence_factor": 0.8333333333,
                },
                "exact_subset_split": {"coherent_amplitude": 0.201},
                "exact_subset_direct": {"coherent_amplitude": 0.200},
            }
        ),
        encoding="utf-8",
    )
    baseline_file.write_text(
        json.dumps({"reference": {"metrics": {"coherent_amplitude": 0.06}}}),
        encoding="utf-8",
    )
    rotor_file.write_text(
        json.dumps({"scan": {"shift_us": [-0.1, 0.0, 0.1], "amplitude": [0.20, 0.24, 0.21]}}),
        encoding="utf-8",
    )
    benchmark_file.write_text("case_x prep 1.2 grad 0.03 ens 384 steps 300\n", encoding="utf-8")

    config_file = tmp_path / "report_config.json"
    config_file.write_text(
        json.dumps(
            {
                "root": ".",
                "output_dir": "generated",
                "benchmark": {"file": "timings.txt"},
                "cases": [
                    {
                        "id": "case_x",
                        "label": "Custom case",
                        "folder": "arbitrary_run_folder",
                        "result": "optimisation_output.json",
                        "excitation": "my_excitation.csv",
                        "conversion": "my_conversion.csv",
                        "validation": "held_out_metrics.json",
                        "baseline": {"file": "reference_pulses.json", "key": "reference"},
                        "rotor_phase": {"file": "timing_scan.json", "key": "scan"},
                        "benchmark_key": "case_x",
                        "metadata": {"spin": "3/2", "mq_order": 3},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    settings, cases = load_report_config(config_file)
    artifacts = generate_report(settings, cases)
    assert not artifacts.warnings
    assert len(artifacts.summary_rows) == 1
    row = artifacts.summary_rows[0]
    assert row["system"] == "Custom case"
    assert np.isclose(row["validation_amplitude"], 0.240)
    assert np.isclose(row["baseline_amplitude"], 0.06)
    assert np.isclose(row["shaped_to_baseline_ratio"], 4.0)
    assert np.isclose(row["time_step_relative_error"], abs(0.241 - 0.240) / 0.240)
    assert np.isclose(row["split_vs_direct_subset_error"], 0.005)
    assert np.isclose(row["objective_gradient_s"], 0.03)

    output = tmp_path / "generated"
    assert (output / "summary.json").exists()
    assert (output / "summary.csv").exists()
    assert (output / "summary.md").exists()
    assert (output / "plots" / "case_x_pulse_pair.png").exists()
    assert (output / "plots" / "convergence.png").exists()
    assert (output / "plots" / "validation_vs_baseline.png").exists()
    assert (output / "plots" / "time_step_convergence.png").exists()
    assert (output / "plots" / "rotor_phase_sensitivity.png").exists()
