"""Blind estimates for carrier frequency, bandwidth, SNR, and symbol rate."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, welch

from radiofry.contracts import UnifiedSignalContainer


@dataclass(frozen=True)
class ParameterEstimate:
    occupied_bandwidth_hz: float | None
    snr_db: float | None
    symbol_rate_hz: float | None
    method: str = "welch_nth_power"
    carrier_frequency_hz: float | None = None
    symbol_rate_confidence: float | None = None


def _occupied_band(psd: np.ndarray, frequencies: np.ndarray, fraction: float) -> tuple[float, float]:
    power = np.maximum(psd, 0)
    total = float(np.sum(power))
    if total <= 0:
        return float(frequencies[0]), float(frequencies[-1])
    cumulative = np.cumsum(power) / total
    lower = frequencies[np.searchsorted(cumulative, (1 - fraction) / 2)]
    upper = frequencies[np.searchsorted(cumulative, 1 - (1 - fraction) / 2)]
    return float(lower), float(upper)


def _select_symbol_rate(
    rates: np.ndarray,
    spectrum: np.ndarray,
    peaks: np.ndarray,
    prominences: np.ndarray,
) -> tuple[float | None, float | None]:
    """Prefer a supported fundamental over an isolated harmonic peak."""

    if peaks.size == 0:
        return None, None
    peak_values = spectrum[peaks]
    scores = []
    for peak, peak_value in zip(peaks, peak_values):
        frequency = rates[peak]
        if frequency <= 0:
            scores.append(-np.inf)
            continue
        harmonic_values = [
            float(np.interp(harmonic * frequency, rates, spectrum, left=0.0, right=0.0))
            for harmonic in (2, 3, 4)
        ]
        harmonic_support = sum(value for value in harmonic_values if value >= peak_value * 0.05)
        scores.append(float(peak_value + 0.25 * harmonic_support))
    selected = int(np.argmax(scores))
    ordered_scores = np.sort(np.asarray(scores))
    separation = (ordered_scores[-1] - ordered_scores[-2]) / (ordered_scores[-1] + 1e-12) if len(ordered_scores) > 1 else 1.0
    prominence = float(prominences[selected])
    noise_floor = float(np.median(spectrum[1:]))
    confidence = float(np.clip((prominence / (prominence + noise_floor + 1e-12)) * (0.5 + 0.5 * separation), 0.0, 1.0))
    return float(rates[peaks[selected]]), confidence


def estimate_parameters(
    signal: UnifiedSignalContainer,
    *,
    occupied_fraction: float = 0.99,
    nperseg: int = 1024,
) -> ParameterEstimate:
    """Estimate carrier, occupied bandwidth, SNR, and a symbol-rate spectral line."""

    if signal.sample_rate is None or signal.iq.size < 8:
        return ParameterEstimate(None, None, None)
    segment_length = min(nperseg, signal.iq.size)
    frequencies, psd = welch(signal.iq, fs=signal.sample_rate, nperseg=segment_length, return_onesided=False)
    order = np.argsort(frequencies)
    frequencies, psd = frequencies[order], np.real(psd[order])
    lower, upper = _occupied_band(psd, frequencies, occupied_fraction)
    in_band = (frequencies >= lower) & (frequencies <= upper)
    out_band = ~in_band
    signal_power = float(np.mean(psd[in_band])) if np.any(in_band) else 0.0
    noise_power = float(np.median(psd[out_band])) if np.any(out_band) else 0.0
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 and signal_power > 0 else None
    band_power = np.maximum(psd[in_band], 0.0)
    band_frequencies = frequencies[in_band]
    centroid = float(np.sum(band_frequencies * band_power) / np.sum(band_power)) if np.sum(band_power) > 0 else None
    metadata_center = signal.metadata.get("center_frequency_hz", signal.metadata.get("carrier_frequency_hz"))
    carrier_frequency = float(metadata_center) if metadata_center is not None else centroid
    parameter_method = "hardware_center_frequency+welch_nth_power" if metadata_center is not None else "welch_psd_centroid+nth_power"

    centered = signal.iq - np.mean(signal.iq)
    powered = np.abs(centered) ** 2
    spectrum = np.abs(np.fft.rfft(powered - np.mean(powered)))
    rates = np.fft.rfftfreq(powered.size, d=1 / signal.sample_rate)
    spectrum[0] = 0
    peaks, properties = find_peaks(spectrum, prominence=max(float(np.median(spectrum)) * 2.0, 1e-12))
    symbol_rate, symbol_rate_confidence = _select_symbol_rate(
        rates,
        spectrum,
        peaks,
        properties.get("prominences", np.array([], dtype=float)),
    )
    return ParameterEstimate(abs(upper - lower), snr_db, symbol_rate, parameter_method, carrier_frequency, symbol_rate_confidence)
