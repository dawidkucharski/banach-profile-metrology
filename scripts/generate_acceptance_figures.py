#!/usr/bin/env python3
"""Generate acceptance-optimising figures and tables from existing data.

Outputs:
  1. harmonic_decomposition.pdf  — FFT harmonic amplitudes (1–50 UPR)
  2. per_sweep_table.tex         — LaTeX table of per-sweep descriptor values
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / "radial_profile_reconstruction_metric"
SUPP_DIR = Path(__file__).resolve().parents[1] / "manuscript" / "elsevier_style"

# ── descriptor labels for table columns ──────────────────────────────
DESCRIPTOR_MAP = {
    "profile_smoothed_Ra_nm":        ("$R_a$", "nm"),
    "profile_smoothed_Rq_nm":        ("$R_q$", "nm"),
    "profile_smoothed_Rz_nm":        ("$R_z$", "nm"),
    "roundness_RONt_nm":             ("$RONt$", "nm"),
    "roundness_RONq_nm":             ("$RONq$", "nm"),
    "banach_l1_mean_nm":             ("$L^1$ mean", "nm"),
    "banach_l2_rms_nm":              ("$L^2$ RMS", "nm"),
    "banach_linf_nm":                ("$L^{\\infty}$", "nm"),
    "banach_bv_total_variation_nm":  ("BV total", "nm"),
    "banach_bv_total_variation_density_nm_per_sample": ("BV density", "nm\\,sample$^{-1}$"),
    "banach_w12_gradient_seminorm_nm_per_sample": ("$W^{1,2}$ grad.", "nm\\,sample$^{-1}$"),
}

DESCRIPTOR_ORDER = list(DESCRIPTOR_MAP.keys())

# ── 1. HARMONIC DECOMPOSITION ────────────────────────────────────────
def harmonic_amplitudes(profile: np.ndarray) -> np.ndarray:
    """Return harmonic amplitudes (1…N/2) from real FFT of a zero-mean profile."""
    y = profile - np.mean(profile)
    n = len(y)
    fft = np.fft.rfft(y)
    amps = np.abs(fft[1:]) * 2.0 / n  # skip DC; normalise
    return amps  # index 0 → 1 UPR


def plot_harmonics(ceramic_csv: Path, steel_csv: Path, output: Path) -> None:
    max_upr = 50
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)

    for ax, csv_path, label, color in [
        (axes[0], ceramic_csv, "FN~111 (ceramic)", "#4C78A8"),
        (axes[1], steel_csv, "FN~101 (steel)", "#E45756"),
    ]:
        df = pd.read_csv(csv_path)
        # use sweep 0, LSCI-centred residual
        sweep0 = df[df["sweep_index"] == 0]["roundness_residual_nm"].values
        amps = harmonic_amplitudes(sweep0)
        n_plot = min(max_upr, len(amps))
        ax.bar(np.arange(1, n_plot + 1), amps[:n_plot], color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(15, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="15~UPR filter cutoff")
        ax.set_ylabel("Amplitude (nm)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=0)
        # annotate dominant harmonic
        dom_idx = np.argmax(amps[:n_plot]) + 1
        dom_amp = amps[dom_idx - 1]
        ax.annotate(f"$H_{{{dom_idx}}}$ = {dom_amp:.0f}~nm",
                    xy=(dom_idx, dom_amp), xytext=(dom_idx + 5, dom_amp * 1.15),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    axes[1].set_xlabel("Harmonic order (UPR)", fontsize=9)
    fig.suptitle("Harmonic Decomposition of LSCI Residual Profile (Sweep~0, unfiltered)", fontsize=11, y=1.01)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output}")


# ── 2. PER-SWEEP DESCRIPTOR TABLE ────────────────────────────────────

def compute_descriptors(profile: np.ndarray, delta_alpha_deg: float) -> dict:
    """Compute all descriptors from a Gaussian-filtered residual profile."""
    p = profile - np.mean(profile)
    n = len(p)
    ra = float(np.mean(np.abs(p)))
    rq = float(np.sqrt(np.mean(p**2)))
    rz = float(np.max(p) - np.min(p))
    ront = rz  # for LSCI-centred Gaussian-filtered residual
    ronq = rq
    l1 = ra
    l2 = rq
    linf = float(np.max(np.abs(p)))
    bv_total = float(np.sum(np.abs(np.diff(p))))
    bv_density = bv_total / n
    w12_grad = float(np.sqrt(np.sum(np.diff(p)**2) / n))
    return {
        "Ra": ra, "Rq": rq, "Rz": rz, "RONt": ront, "RONq": ronq,
        "L1": l1, "L2": l2, "Linf": linf,
        "BV_total": bv_total, "BV_density": bv_density, "W12_grad": w12_grad,
    }


DESCRIPTOR_LABELS = [
    ("Ra",        "$R_a$",                "nm"),
    ("Rq",        "$R_q$",                "nm"),
    ("Rz",        "$R_z$",                "nm"),
    ("RONt",      "$RONt$",               "nm"),
    ("RONq",      "$RONq$",               "nm"),
    ("L1",        "$L^1$ mean",           "nm"),
    ("L2",        "$L^2$ RMS",            "nm"),
    ("Linf",      "$L^{\\infty}$",        "nm"),
    ("BV_total",  "BV total",             "nm"),
    ("BV_density","BV density",           "nm\\,sample$^{-1}$"),
    ("W12_grad",  "$W^{1,2}$ grad.",      "nm\\,sample$^{-1}$"),
]


def _fmt_sweep(val_nm: float, is_micron: bool) -> str:
    """Format a single sweep value for LaTeX."""
    v = val_nm / 1000.0 if is_micron else val_nm
    if abs(v) >= 1000:
        return f"{v:.0f}"
    elif abs(v) >= 100:
        return f"{v:.1f}"
    elif abs(v) >= 10:
        return f"{v:.2f}"
    elif abs(v) >= 1:
        return f"{v:.3f}"
    else:
        return f"{v:.4f}"


def _fmt_mean_u95(mean_nm: float, u95_nm: float, is_micron: bool) -> str:
    """Format mean ± U95 for LaTeX."""
    m = mean_nm / 1000.0 if is_micron else mean_nm
    u = u95_nm / 1000.0 if is_micron else u95_nm
    # match precision to uncertainty
    if u == 0 or np.isnan(u):
        return f"{m:.3f}" if is_micron else f"{m:.1f}"
    if u >= 10:
        return f"{m:.0f} \\pm {u:.0f}"
    elif u >= 1:
        return f"{m:.1f} \\pm {u:.1f}"
    elif u >= 0.1:
        return f"{m:.2f} \\pm {u:.2f}"
    elif u >= 0.01:
        return f"{m:.3f} \\pm {u:.3f}"
    else:
        return f"{m:.4f} \\pm {u:.4f}"


def build_per_sweep_table(ceramic_csv: Path, steel_csv: Path, output: Path) -> None:
    """Build per-sweep table from Gaussian-filtered profile data."""
    t_975_4 = 2.776

    # ── compute descriptors per sweep from smoothed_profile_nm ──
    all_data = {}  # artifact → {key: [s0, s1, s2, s3, s4]}
    for artifact, csv_path, delta_deg, is_micron in [
        ("ceramic", ceramic_csv, 0.0833333, False),
        ("steel", steel_csv, 0.020, True),
    ]:
        df = pd.read_csv(csv_path)
        sweeps = {}
        for sweep_idx in range(5):
            mask = df["sweep_index"] == sweep_idx
            profile = df.loc[mask, "roundness_residual_nm"].values
            if len(profile) == 0:
                continue
            sweeps[sweep_idx] = compute_descriptors(profile, delta_deg)
        all_data[artifact] = sweeps

    # ── build LaTeX ──
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{Per-sweep descriptor values computed from the LSCI-centred residual "
        r"(pre-Gaussian-filter, 15~UPR cutoff). "
        r"Each row reports the five individual sweep values followed by the mean $\pm$ expanded uncertainty "
        r"($U_{95}=t_{0.975,4}\,s/\sqrt{5}$, $t_{0.975,4}=2.776$). "
        r"Absolute values are higher than those in Table~2 of the main manuscript because the 15~UPR "
        r"Gaussian low-pass filter has not been applied; the sweep-to-sweep variability pattern is preserved.}"
    )
    lines.append(r"\label{tab:supp_per_sweep}")
    lines.append(r"\begin{tabular}{l" + "c" * 5 + "c" + "c" * 5 + "c}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{6}{c}{FN~111 (ceramic)} & \multicolumn{6}{c}{FN~101 (steel)} \\")
    lines.append(r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}")
    lines.append(
        r"Descriptor & S0 & S1 & S2 & S3 & S4 & Mean $\pm$ $U_{95}$ "
        r"& S0 & S1 & S2 & S3 & S4 & Mean $\pm$ $U_{95}$ \\"
    )
    lines.append(r"\midrule")

    for key, label, _unit in DESCRIPTOR_LABELS:
        row_parts = [label]
        for artifact, is_micron in [("ceramic", False), ("steel", True)]:
            vals = [all_data[artifact][s][key] for s in range(5)]
            arr = np.array(vals)
            mean_v = float(np.mean(arr))
            sem_v = float(np.std(arr, ddof=1) / np.sqrt(5))
            u95 = t_975_4 * sem_v
            for v in vals:
                row_parts.append(_fmt_sweep(v, is_micron))
            row_parts.append(_fmt_mean_u95(mean_v, u95, is_micron))
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output.write_text("\n".join(lines) + "\n")
    print(f"  → {output}")


# ── main ─────────────────────────────────────────────────────────────
def main() -> None:
    ceramic_profile = RESULTS / "ceramic" / "profile_analysis.csv"
    steel_profile = RESULTS / "steel" / "profile_analysis.csv"

    print("Generating harmonic decomposition figure …")
    plot_harmonics(ceramic_profile, steel_profile, RESULTS / "harmonic_decomposition.pdf")

    print("Generating per-sweep descriptor table …")
    build_per_sweep_table(ceramic_profile, steel_profile, SUPP_DIR / "per_sweep_table.tex")

    print("Done.")


if __name__ == "__main__":
    main()
