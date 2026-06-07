# Banach-Space Norm Descriptors for Interferometric Profile Metrology

Reproducibility package for *"Banach-Space Norm Descriptors for Interferometric Profile Metrology: Beyond Amplitude Averaging"* (submitted).

## Contents

| Path | Description |
|:---|:---|
| `scripts/reconstruct_radial_profile_sequence.py` | Main pipeline: raw interferogram → EFM phase decoding → unwrapping → nanometre profile → LSCI + Gaussian filtering → ISO & Banach descriptors |
| `src/btri/` | Python package: radial EFM, phase unwrapping, filtering, descriptor computation |
| `results/radial_profile_reconstruction_metric/` | All generated outputs — CSVs, JSONs, NPYs, vector PDF figures |
| `configs/` | Pipeline configuration files |
| `pyproject.toml` | Python project metadata and dependencies |

## Results structure

```
results/radial_profile_reconstruction_metric/
├── ceramic/
│   ├── profile_analysis.csv          # Nanometre profiles, reference models, residuals
│   ├── phase_sequence.csv            # Wrapped & unwrapped excess-fraction sequences
│   ├── metrics.csv / metrics.json    # Full-sequence scalar descriptors
│   ├── sweep_metric_uncertainty.csv  # Per-sweep descriptor values (n=5)
│   ├── *.pdf                         # Vector-graphics figures
│   └── *.npy                         # Binary phase/height arrays
├── steel/
│   └── (same structure)
├── sample_comparison_metrics.csv
├── descriptor_correlation_heatmap.pdf
├── profile_psd_comparison.pdf
├── cohens_d_artefact_separation.pdf
├── flick_flat_detection.pdf
├── flick_flat_measurement.pdf
└── lp_convergence.pdf
```

## Reproduce the analysis

### Re-run the full pipeline (requires raw PNG archives)

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/reconstruct_radial_profile_sequence.py \
  --dataset ceramic=ceramic --dataset steel=steel \
  --output-dir results/radial_profile_reconstruction_metric \
  --frame-limit 0 --frame-selection first --downsample 1 \
  --max-radius-px 240 --min-radius-px 5 \
  --ceramic-min-radius-px 50 --steel-min-radius-px 5 \
  --peak-spline-smoothing-factor 0.03 \
  --peak-prominence-fraction 0.04
```

### Regenerate figures from CSVs

All analytical figures can be regenerated from the CSV files without re-running the EFM pipeline.

### Environment

- Python 3.9+
- `numpy`, `scipy`, `pandas`, `matplotlib`
- `python -m venv .venv && source .venv/bin/activate && pip install numpy scipy pandas matplotlib`

## Data availability

Raw interferometric PNG archives (FN~111: 24,000 frames; FN~101: 100,000 frames, ~4 GB total) are deposited on Zenodo. DOI will be added upon publication.

## Licence

Provided for academic reproducibility. Contact the author for usage terms.

## Citation

Please cite the accompanying manuscript (reference to be added upon publication).
