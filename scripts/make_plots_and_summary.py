#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_pulse(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True)


def pair_plot(folder: Path, title: str, output: Path):
    ex = read_pulse(folder / "excitation_pixels.csv")
    cv = read_pulse(folder / "conversion_pixels.csv")
    fig, ax = plt.subplots(3, 2, figsize=(10.0, 8.0), sharex="col")
    for j, (a, name) in enumerate(((ex, "Excitation"), (cv, "Conversion"))):
        ax[0, j].plot(a["time_us"], a["ux_hz"] / 1e3, label="ux")
        ax[0, j].plot(a["time_us"], a["uy_hz"] / 1e3, label="uy")
        ax[0, j].set_title(name)
        ax[0, j].set_ylabel("Cartesian RF (kHz)")
        ax[0, j].grid(True, alpha=0.25)
        ax[0, j].legend(loc="best")
        ax[1, j].plot(a["time_us"], a["amplitude_hz"] / 1e3)
        ax[1, j].set_ylabel("Amplitude (kHz)")
        ax[1, j].set_ylim(0.0, 1.06 * float(np.max(a["amplitude_hz"]) / 1e3))
        ax[1, j].ticklabel_format(axis="y", style="plain", useOffset=False)
        ax[1, j].grid(True, alpha=0.25)
        phase = np.unwrap(a["phase_rad"])
        ax[2, j].plot(a["time_us"], phase)
        ax[2, j].set_xlabel("Time (microseconds)")
        ax[2, j].set_ylabel("Unwrapped phase (rad)")
        ax[2, j].grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    rb_result = read_json(RESULTS / "rb2so4_10us_dt05/result.json")
    al_result = read_json(RESULTS / "alpo4_10us_dense/result.json")
    rb_val = read_json(RESULTS / "rb_final_validation.json")
    al_val = read_json(RESULTS / "al_final_validation.json")
    hard = read_json(RESULTS / "hard_pulse_final_validation.json")

    rb_amp = rb_val["split_dt_0p025us"]["coherent_amplitude"]
    al_amp = al_val["split_dt_0p025us"]["coherent_amplitude"]
    rb_hard = hard["rb"]["metrics"]["coherent_amplitude"]
    al_hard = hard["al"]["metrics"]["coherent_amplitude"]

    rows = [
        {
            "system": "Rb2SO4 87Rb",
            "spin": "3/2", "mq_order": 3,
            "excitation_us": 10.0, "conversion_us": 10.0,
            "pixel_dt_us": 0.2, "internal_dt_us": 0.05,
            "max_nutation_khz": 120.0,
            "training_orientations": 512,
            "validation_orientations": rb_val["dense_powder_orientations"],
            "validation_ensemble_members": rb_val["ensemble_members"],
            "training_amplitude": rb_result["metrics"]["coherent_amplitude"],
            "validation_amplitude": rb_amp,
            "validation_incoherent_amplitude": rb_val["split_dt_0p025us"]["incoherent_amplitude"],
            "coherence_factor": rb_val["split_dt_0p025us"]["coherence_factor"],
            "rectangular_baseline_amplitude": rb_hard,
            "shaped_to_rectangular_ratio": rb_amp / rb_hard,
            "time_step_relative_error": rb_val["relative_dt_error_0p05_vs_0p025"],
            "split_vs_direct_subset_error": rb_val["relative_split_vs_exact_subset"],
            "optimisation_s": rb_result["metadata"]["training_elapsed_s"],
            "objective_gradient_s": 0.02597538060001625,
        },
        {
            "system": "AlPO4 27Al",
            "spin": "5/2", "mq_order": 3,
            "excitation_us": 10.0, "conversion_us": 10.0,
            "pixel_dt_us": 0.2, "internal_dt_us": 0.10,
            "max_nutation_khz": 100.0,
            "training_orientations": 512,
            "validation_orientations": al_val["dense_powder_orientations"],
            "validation_ensemble_members": al_val["ensemble_members"],
            "training_amplitude": al_result["metrics"]["coherent_amplitude"],
            "validation_amplitude": al_amp,
            "validation_incoherent_amplitude": al_val["split_dt_0p025us"]["incoherent_amplitude"],
            "coherence_factor": al_val["split_dt_0p025us"]["coherence_factor"],
            "rectangular_baseline_amplitude": al_hard,
            "shaped_to_rectangular_ratio": al_amp / al_hard,
            "time_step_relative_error": al_val["relative_dt_error_0p1_vs_0p025"],
            "split_vs_direct_subset_error": al_val["relative_split_vs_exact_subset"],
            "optimisation_s": al_result["metadata"]["training_elapsed_s"],
            "objective_gradient_s": 0.028157924199990702,
        },
    ]
    (RESULTS / "summary.json").write_text(json.dumps(rows, indent=2))
    with (RESULTS / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    pair_plot(RESULTS / "rb2so4_10us_dt05",
              "Rb2SO4: 10 microsecond 3Q excitation and conversion",
              PLOTS / "rb2so4_pulse_pair.png")
    pair_plot(RESULTS / "alpo4_10us_dense",
              "AlPO4: 10 microsecond 3Q excitation and conversion",
              PLOTS / "alpo4_pulse_pair.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot([x["iteration"] for x in rb_result["history"]],
            [x["coherent_amplitude"] for x in rb_result["history"]], label="Rb2SO4")
    ax.plot([x["iteration"] for x in al_result["history"]],
            [x["coherent_amplitude"] for x in al_result["history"]], label="AlPO4")
    ax.set_xlabel("L-BFGS iteration")
    ax.set_ylabel("Training coherent pathway amplitude")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "convergence.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    x = np.arange(2); width = 0.36
    ax.bar(x - width/2, [rb_hard, al_hard], width, label="Best tested rectangular pair")
    ax.bar(x + width/2, [rb_amp, al_amp], width, label="Optimised shaped pair")
    ax.set_xticks(x, ["Rb2SO4", "AlPO4"])
    ax.set_ylabel("Held-out coherent pathway amplitude")
    ax.grid(True, axis="y", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "validation_vs_rectangular.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot([0.05, 0.025], [rb_val["split_dt_0p05us"]["coherent_amplitude"], rb_amp], marker="o", label="Rb2SO4")
    ax.plot([0.10, 0.05, 0.025], [al_val["split_dt_0p1us"]["coherent_amplitude"], al_val["split_dt_0p05us"]["coherent_amplitude"], al_amp], marker="o", label="AlPO4")
    ax.invert_xaxis(); ax.set_xlabel("Internal step (microseconds)")
    ax.set_ylabel("Held-out coherent pathway amplitude")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "time_step_convergence.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    rp = read_json(RESULTS / "rotor_phase_sensitivity.json")
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(rp["rb"]["shift_us"], rp["rb"]["amplitude"], label="Rb2SO4")
    ax.plot(rp["al"]["shift_us"], rp["al"]["amplitude"], label="AlPO4")
    ax.set_xlabel("Conversion start-time shift (microseconds)")
    ax.set_ylabel("Coherent pathway amplitude")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "rotor_phase_sensitivity.png", dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
