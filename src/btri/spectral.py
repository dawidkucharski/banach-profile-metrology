"""Spectral and quantum-chaotic analysis for BTRI residual fields."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist
from scipy.stats import norm


def compute_2d_fft(residual: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Compute the 2D FFT magnitude spectrum of a residual field."""
    if mask is not None:
        residual = np.where(mask, residual, 0)
    spectrum = np.abs(fftshift(fft2(residual)))
    return spectrum


def radial_psd(spectrum: np.ndarray) -> np.ndarray:
    """Compute the radial power spectral density (PSD) from a 2D spectrum."""
    y, x = np.indices(spectrum.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile


def extract_spectral_peaks(spectrum: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Extract dominant spectral peaks above a relative threshold."""
    flat = spectrum.ravel()
    peaks, _ = find_peaks(flat, height=threshold * np.max(flat))
    return flat[peaks]


def nearest_neighbor_spacings(peaks: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor spacings between sorted spectral peaks."""
    sorted_peaks = np.sort(peaks)
    spacings = np.diff(sorted_peaks)
    return spacings


def unfold_spectrum(spacings: np.ndarray) -> np.ndarray:
    """Normalize spacings (unfolding) for spacing statistics."""
    mean_spacing = np.mean(spacings)
    return spacings / mean_spacing if mean_spacing > 0 else spacings


def wigner_dyson_pdf(s: np.ndarray) -> np.ndarray:
    """Wigner-Dyson (GUE) spacing distribution."""
    return (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)


def poisson_pdf(s: np.ndarray) -> np.ndarray:
    """Poisson (exponential) spacing distribution."""
    return np.exp(-s)


def plot_spacing_histogram(spacings, output_path, label, bins=40):
    plt.figure(figsize=(6, 4))
    plt.hist(spacings, bins=bins, density=True, alpha=0.7, label=label)
    s = np.linspace(0, np.max(spacings), 100)
    plt.plot(s, poisson_pdf(s), 'k--', label='Poisson')
    plt.plot(s, wigner_dyson_pdf(s), 'r-', label='Wigner-Dyson (GUE)')
    plt.xlabel('Unfolded spacing')
    plt.ylabel('Probability density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_radial_psd(radial_psd, output_path):
    plt.figure(figsize=(6, 4))
    plt.plot(radial_psd)
    plt.xlabel('Radial frequency bin')
    plt.ylabel('Mean spectral power')
    plt.title('Radial Power Spectral Density')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_spectrum_map(spectrum, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log1p(spectrum), cmap='viridis', origin='lower')
    plt.colorbar(label='log(1 + |FFT|)')
    plt.title('2D FFT Magnitude Spectrum')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Additional functions for CDFs, autocorrelation, and Riemann-zeta-inspired reference can be added as needed.
