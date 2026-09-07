"""Additive white Gaussian noise channel for Synthetic Dataset V1.

SNR is defined as total in-band signal power over total noise power across the
full sampled bandwidth: ``10*log10(mean(|s|^2) / mean(|n|^2))``. The noise is
drawn, not scaled to hit the target exactly, so the realized SNR is reported
alongside the requested one and both are stored in ground truth.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseReport:
    noise_type: str
    target_snr_db: float | None
    signal_power: float
    target_noise_power: float
    realized_noise_power: float
    realized_snr_db: float | None
    snr_definition: str = "total_band_signal_power_over_noise_power"


def add_awgn(
    iq: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, NoiseReport]:
    """Add complex circularly-symmetric AWGN at the requested SNR."""

    clean = np.asarray(iq, dtype=np.complex64)
    signal_power = float(np.mean(np.abs(clean.astype(np.complex128)) ** 2))
    if snr_db is None:
        return clean, NoiseReport("none", None, signal_power, 0.0, 0.0, None)
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    sigma = np.sqrt(target_noise_power / 2.0)
    noise = rng.normal(0.0, sigma, clean.size) + 1j * rng.normal(0.0, sigma, clean.size)
    noisy = (clean.astype(np.complex128) + noise).astype(np.complex64)
    realized_noise_power = float(np.mean(np.abs(noisy.astype(np.complex128) - clean.astype(np.complex128)) ** 2))
    realized_snr_db = float(10 * np.log10(signal_power / realized_noise_power)) if realized_noise_power > 0 else None
    return noisy, NoiseReport(
        "awgn",
        float(snr_db),
        signal_power,
        float(target_noise_power),
        realized_noise_power,
        realized_snr_db,
    )
