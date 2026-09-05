"""Blind estimates for bandwidth, SNR, and symbol rate."""

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


def _occupied_band(psd: np.ndarray, frequencies: np.ndarray, fraction: float) -> tuple[float, float]:
    power = np.maximum(psd, 0)
    total = float(np.sum(power))
    if total <= 0:
        return float(frequencies[0]), float(frequencies[-1])
    cumulative = np.cumsum(power) / total
    lower = frequencies[np.searchsorted(cumulative, (1 - fraction) / 2)]
    upper = frequencies[np.searchsorted(cumulative, 1 - (1 - fraction) / 2)]
    return float(lower), float(upper)


def estimate_parameters(
    signal: UnifiedSignalContainer,
    *,
    occupied_fraction: float = 0.99,
    nperseg: int = 1024,
) -> ParameterEstimate:
    """Estimate occupied bandwidth/SNR and a symbol-rate spectral line."""

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

    centered = signal.iq - np.mean(signal.iq)
    powered = np.abs(centered) ** 2
    spectrum = np.abs(np.fft.rfft(powered - np.mean(powered)))
    rates = np.fft.rfftfreq(powered.size, d=1 / signal.sample_rate)
    spectrum[0] = 0
    peaks, _ = find_peaks(spectrum)
    symbol_rate = float(rates[peaks[np.argmax(spectrum[peaks])]]) if peaks.size else None
    return ParameterEstimate(abs(upper - lower), snr_db, symbol_rate)
