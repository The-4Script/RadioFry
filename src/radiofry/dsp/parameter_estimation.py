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
    symbol_rate_feature: str | None = None
    symbol_rate_source: str = "estimated"


def _phase_second_difference(samples: np.ndarray) -> np.ndarray:
    frequency = np.diff(np.unwrap(np.angle(samples)), prepend=0.0)
    return np.diff(frequency, prepend=0.0) ** 2


# Pre-FFT nonlinearities searched for a symbol-rate spectral line. Envelope power is
# the classical squaring method and carries a timing tone only when pulses overlap;
# full-duty rectangular pulses put an exact null at the symbol rate, and constant-
# modulus constellations flatten the feature entirely. The transition and phase-step
# features cover those cases. See BANK.md Entries 003-004 for the measured comparison.
SYMBOL_RATE_FEATURES = {
    "envelope_power": lambda samples: np.abs(samples) ** 2,
    "transition_power": lambda samples: np.abs(np.diff(samples, prepend=samples[0])) ** 2,
    "phase_second_difference": _phase_second_difference,
}


# Reported occupied bandwidth (BANK.md Entry 041).
#
# For a rectangular-pulse linear modulation the main lobe spans +/-Rs, so BW/Rs ~ 2 is
# the physical target. Measured over 240 captures (6 modulations x samples-per-symbol
# 4/8/16/32 x SNR 20..0 dB x 2 seeds), the median BW/Rs by oversampling factor was:
#
#   fraction   sps4   sps8  sps16  sps32
#     0.99     3.65   7.35  14.73  29.00   <- tracks the sample rate, not the signal
#     0.90     2.41   3.23   4.86   6.75
#     0.85     1.43   2.11   2.71   2.92   <- closest to 2 and roughly sps-independent
#     0.80     1.18   1.29   1.36   1.41   <- inside the main lobe
#
# 0.85 is the tightest fraction that still brackets the main lobe. It is deliberately
# NOT used for the noise-floor band; see `estimate_parameters`.
BANDWIDTH_FRACTION = 0.85


# Cyclostationary symbol-rate estimation (BANK.md Entry 038).
#
# A signal built from symbols at rate Rs is cyclostationary with cycle frequencies at
# k*Rs, so the cyclic autocorrelation
#
#     R^alpha(tau) = (1/N) sum_n p(n) p*(n-tau) exp(-j 2 pi alpha n / Fs)
#
# is non-zero at alpha = Rs. Computing it as an FFT of the lag product gives every alpha
# at once, so the whole scan costs a handful of FFTs. `p` is the instantaneous frequency,
# which for FSK is a PAM-like waveform at the symbol rate.
#
# For a linearly modulated signal the alpha = Rs term is proportional to the pulse-
# spectrum overlap sum_f G(f) G*(f - Rs), so it exists only when the pulse has excess
# bandwidth. A rectangular CPFSK frequency pulse has it; a Gaussian GFSK pulse
# deliberately does not, and measurement confirms the feature is absent there. The
# estimate is therefore gated on its own peak-to-background ratio, and the existing
# adaptive features stay in charge whenever the cyclic evidence is weak.
#
# Measured over 120 FSK captures (2 modulations x sps 4/8/16/32 x SNR 20/15/10 x 5 seeds):
# a gate of 6.0 gained 4 correct estimates with ZERO regressions, while lower gates gained
# more but regressed 4-8 captures.
# Lowest symbol rate the peak search will consider, as fs / MAX_SAMPLES_PER_SYMBOL
# (BANK.md Entry 042).
#
# A burst - any capture that is not transmitting for its whole duration - puts a strong
# line at the burst envelope's own fundamental, near fs/N, together with a dense harmonic
# series. `_select_symbol_rate` rewards harmonic support, so that series outscored the
# real symbol-rate line and the estimate collapsed to ~24 Hz at fs = 200 kHz: at 75% duty
# the reported rate was 24 Hz against a true 25,000 Hz. The symbol-rate line itself was
# still present at full strength; only the selection was wrong.
#
# 256 is deliberately generous: the project generates, trains on and demodulates
# oversampling factors 4-32, so this excludes nothing that has ever been validated. It
# does exclude genuinely narrowband captures at a very high sample rate (sps > 256), which
# are outside the evaluated envelope; revisit this bound if such captures become real.
MAX_SAMPLES_PER_SYMBOL = 256

CYCLIC_LAGS = (1, 2, 3, 4, 6, 8)
CYCLIC_PEAK_RATIO_MIN = 6.0
_CYCLIC_ALPHA_MIN_HZ = 500.0


def _cyclic_symbol_rate(iq: np.ndarray, sample_rate: float) -> tuple[float | None, float]:
    """Blind cycle-frequency estimate and its peak-to-background ratio."""

    samples = np.asarray(iq, dtype=np.complex64)
    if samples.size < 64 or sample_rate is None or sample_rate <= 0:
        return None, 0.0
    frequency = np.diff(np.unwrap(np.angle(samples)), prepend=0.0).astype(np.float64)
    frequency = frequency - frequency.mean()
    profile = None
    for lag in CYCLIC_LAGS:
        if frequency.size <= lag:
            continue
        product = frequency[lag:] * frequency[:-lag]
        product = product - product.mean()
        magnitude = np.abs(np.fft.fft(product, n=frequency.size))
        profile = magnitude if profile is None else profile + magnitude
    if profile is None:
        return None, 0.0
    alphas = np.fft.fftfreq(frequency.size, d=1 / sample_rate)
    band = (alphas >= _CYCLIC_ALPHA_MIN_HZ) & (alphas <= sample_rate / 4)
    if not np.any(band):
        return None, 0.0
    candidates, strengths = alphas[band], profile[band]
    index = int(np.argmax(strengths))
    background = float(np.median(strengths))
    ratio = float(strengths[index] / (background + 1e-12))
    return float(candidates[index]), ratio


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


def _symbol_rate_candidate(powered: np.ndarray, sample_rate: float) -> tuple[float | None, float | None]:
    """Score one pre-FFT feature through the existing peak search and selector."""

    spectrum = np.abs(np.fft.rfft(powered - np.mean(powered)))
    rates = np.fft.rfftfreq(powered.size, d=1 / sample_rate)
    spectrum[0] = 0
    # Suppress the burst-envelope region before peak selection; see
    # MAX_SAMPLES_PER_SYMBOL. Without this a duty-cycled capture locks onto its own
    # envelope period instead of its symbol clock.
    spectrum[rates < sample_rate / MAX_SAMPLES_PER_SYMBOL] = 0
    peaks, properties = find_peaks(spectrum, prominence=max(float(np.median(spectrum)) * 2.0, 1e-12))
    return _select_symbol_rate(
        rates,
        spectrum,
        peaks,
        properties.get("prominences", np.array([], dtype=float)),
    )


def estimate_parameters(
    signal: UnifiedSignalContainer,
    *,
    occupied_fraction: float = 0.99,
    bandwidth_fraction: float = BANDWIDTH_FRACTION,
    nperseg: int = 1024,
) -> ParameterEstimate:
    """Estimate carrier, occupied bandwidth, SNR, and a symbol-rate spectral line.

    Two power fractions, because one band cannot serve both purposes:

    * `occupied_fraction` (0.99) bounds the band that is treated as *signal* when the
      noise floor is measured from everything outside it. It has to be generous, or the
      "out of band" region still contains signal and the noise floor is overestimated.
    * `bandwidth_fraction` (0.85) is what gets *reported* as the occupied bandwidth. It
      has to be tight, because at finite SNR the noise carries enough of the total power
      that a 0.99 criterion returns very nearly the whole sampled band whatever the
      signal is.

    Using 0.99 for both is what made the reported bandwidth ~0.9 x the sample rate for
    every capture regardless of its symbol rate (BANK.md Entry 041).
    """

    if signal.sample_rate is None or signal.iq.size < 8:
        return ParameterEstimate(None, None, None)
    segment_length = min(nperseg, signal.iq.size)
    frequencies, psd = welch(signal.iq, fs=signal.sample_rate, nperseg=segment_length, return_onesided=False)
    order = np.argsort(frequencies)
    frequencies, psd = frequencies[order], np.real(psd[order])
    lower, upper = _occupied_band(psd, frequencies, occupied_fraction)
    # Reported separately and more tightly - see the docstring.
    band_lower, band_upper = _occupied_band(psd, frequencies, bandwidth_fraction)
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
    symbol_rate, symbol_rate_confidence, symbol_rate_feature = None, None, None
    for name, feature in SYMBOL_RATE_FEATURES.items():
        rate, confidence = _symbol_rate_candidate(feature(centered), signal.sample_rate)
        if rate is None or confidence is None:
            continue
        if symbol_rate_confidence is None or confidence > symbol_rate_confidence:
            symbol_rate, symbol_rate_confidence, symbol_rate_feature = rate, confidence, name
    # Entry 038: a strong cyclic feature beats the transition-based features, which lose
    # the symbol-rate line above samples-per-symbol 8. Gated so a weak feature changes
    # nothing.
    cyclic_rate, cyclic_ratio = _cyclic_symbol_rate(signal.iq, signal.sample_rate)
    if cyclic_rate is not None and cyclic_ratio >= CYCLIC_PEAK_RATIO_MIN:
        symbol_rate = cyclic_rate
        symbol_rate_feature = "cyclic_autocorrelation"
        symbol_rate_confidence = float(np.clip(cyclic_ratio / (cyclic_ratio + 10.0), 0.0, 1.0))

    return ParameterEstimate(
        abs(band_upper - band_lower),
        snr_db,
        symbol_rate,
        parameter_method,
        carrier_frequency,
        symbol_rate_confidence,
        symbol_rate_feature,
    )
