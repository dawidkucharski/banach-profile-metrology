"""Validate steel reconstruction before committing to manuscript."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path("results/radial_profile_reconstruction_metric/steel")
ok = True

def check(label, condition, detail=""):
    global ok
    status = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"  [{status}] {label} {detail}")

# 1. Phase continuity
phase = pd.read_csv(ROOT / "phase_sequence.csv")
vals = phase["epsilon_unwrapped"].to_numpy(float)
diff = np.abs(np.diff(vals))
valid_n = int(np.isfinite(vals).sum())
max_jump = float(np.nanmax(diff))
p99_jump = float(np.nanpercentile(diff, 99))

check("Phase valid frames", valid_n == len(phase),
      f"{valid_n}/{len(phase)}")
check("Phase continuity (max diff < 0.5 fringe)", max_jump < 0.5,
      f"max_diff={max_jump:.4f} fringe")
check("Phase continuity (p99 diff < 0.3 fringe)", p99_jump < 0.3,
      f"p99_diff={p99_jump:.4f} fringe")
print(f"  Phase range: {np.nanmax(vals)-np.nanmin(vals):.1f} fringe")

# 2. Metrics sanity
metrics = pd.read_csv(ROOT / "metrics.csv").iloc[0]
step_depth = float(metrics["step_depth_nm"])
ra = float(metrics["profile_smoothed_Ra_nm"])
rq = float(metrics["profile_smoothed_Rq_nm"])
rz = float(metrics["profile_smoothed_Rz_nm"])
check("Step depth > 100 nm", step_depth > 100, f"{step_depth:.1f} nm")
check("Step depth < 100 µm", step_depth < 100_000, f"{step_depth:.1f} nm")
check("Residual Ra << step depth", ra < step_depth * 0.8,
      f"Ra={ra:.1f} nm vs step={step_depth:.1f} nm")

# 3. Sweep repeatability
sweep = pd.read_csv(ROOT / "sweep_metric_uncertainty.csv")
if "step_depth_nm" in sweep.columns and len(sweep) >= 3:
    sd_vals = sweep["step_depth_nm"].dropna().to_numpy(float)
    if len(sd_vals) >= 3:
        sd_mean = float(np.mean(sd_vals))
        sd_std = float(np.std(sd_vals, ddof=1))
        check("Step depth sweep repeatability (std/mean < 10%)",
              sd_std / sd_mean < 0.10,
              f"mean={sd_mean:.0f} ± {sd_std:.0f} nm ({sd_std/sd_mean*100:.1f}%)")

# 4. Profile structure
profile = pd.read_csv(ROOT / "profile_analysis.csv")
if "step_reference_nm" in profile.columns:
    ref = profile["step_reference_nm"].to_numpy(float)
    ref_range = float(np.nanmax(ref) - np.nanmin(ref))
    check("Two-plateau reference has step", ref_range > step_depth * 0.5,
          f"plateau range={ref_range:.1f} nm, step_depth={step_depth:.1f} nm")

# 5. Per-sweep step check
if "sweep_index" in profile.columns and "step_reference_nm" in profile.columns:
    print("\n  Per-sweep step depth:")
    for si, grp in profile.groupby("sweep_index", sort=True):
        r = grp["step_reference_nm"].to_numpy(float)
        sd = float(np.nanmax(r) - np.nanmin(r)) if np.any(np.isfinite(r)) else np.nan
        print(f"    Sweep {int(si)}: step={sd:.1f} nm")

print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED — review before manuscript'}")

# Return exit code
sys.exit(0 if ok else 1)
