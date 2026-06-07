"""
Automated spectral statistics and quantum-chaotic analogy analysis for BTRI Stage 5 outputs (final version).
"""
import numpy as np
from pathlib import Path
from src.btri import spectral

INPUTS = [
    {
        "npy": "results/r_btri_stage5_15000/spherical/spherical_mean_surface.npy",
        "outdir": "results/r_btri_stage5_15000/spectral_spherical",
        "label": "Ceramic (spherical)"
    },
    {
        "npy": "results/r_btri_stage5_15000/step/step_mean_surface.npy",
        "outdir": "results/r_btri_stage5_15000/spectral_step",
        "label": "Steel (step)"
    }
]

def main():
    for entry in INPUTS:
        arr = np.load(entry["npy"])
        outdir = Path(entry["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        label = entry["label"]
        spectrum = spectral.compute_2d_fft(arr)
        radial = spectral.radial_psd(spectrum)
        peaks = spectral.extract_spectral_peaks(spectrum)
        spacings = spectral.nearest_neighbor_spacings(peaks)
        unfolded = spectral.unfold_spectrum(spacings)
        spectral.plot_spectrum_map(spectrum, outdir / "fft_map.png")
        spectral.plot_radial_psd(radial, outdir / "radial_psd.png")
        if len(unfolded) > 0:
            spectral.plot_spacing_histogram(unfolded, outdir / "spacing_hist.png", label)
        print(f"[OK] Spectral analysis complete for {label}.")

if __name__ == "__main__":
    main()
