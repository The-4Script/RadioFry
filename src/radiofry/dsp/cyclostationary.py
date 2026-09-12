"""Lightweight non-ML modulation-family cross-checks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassicalFamilyEstimate:
    family: str
    confidence: float
    evidence: dict[str, float]


CLASSICAL_THRESHOLDS = {
    "amplitude_cv_psk": 0.15,
    "amplitude_cv_qam": 0.2,
    "frequency_cv_fsk": 0.8,
    "fourth_power_psk": 0.2,
    # BPSK/QPSK retain a strong fourth-power carrier line even when phase
    # transitions make instantaneous-frequency CV look FSK-like.
    "fourth_power_psk_strong": 0.8,
}

# Positive family-level analog evidence (BANK.md Entry 027).
#
# `frequency_cv = std(dphi) / mean(|dphi|)` measures how *impulsive* the instantaneous
# frequency is. A digital signal jumps at symbol boundaries and sits still between them,
# so its phase increments are heavy-tailed and std >> mean|.|. An analog signal's
# instantaneous frequency moves smoothly, so the two are comparable.
#
# Measured over 800 digital controls (8 modulations x samples-per-symbol 4/8/16/32 x
# SNR 0-20 dB x 5 seeds) the lowest value seen anywhere was 1.032 (BFSK at sps=4).
# 0.9 leaves ~13% margin below that and produced 0/800 false positives.
ANALOG_FREQUENCY_CV_MAX = 0.9
_ANALOG_CONFIDENCE_FLOOR = 0.5
_ANALOG_CONFIDENCE_SPAN = 0.4


def _envelope_flatness(amplitude: np.ndarray) -> float:
    """Spectral flatness of the envelope: geometric mean over arithmetic mean.

    A constant-envelope waveform carries no message in `|s|`, so its envelope spectrum
    is noise-like and flat (-> 1). An amplitude-modulated waveform puts the message's
    discrete tones there, making it peaky (-> 0). Recorded as evidence only; it does not
    take part in the family decision (BANK.md Entry 032).
    """

    centred = amplitude - np.mean(amplitude)
    if not np.any(centred):
        return 1.0
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size))) ** 2
    spectrum = spectrum / (spectrum.sum() + 1e-30)
    geometric = float(np.exp(np.mean(np.log(spectrum + 1e-20))))
    return float(geometric / (float(np.mean(spectrum)) + 1e-20))


def estimate_modulation_family(iq: np.ndarray) -> ClassicalFamilyEstimate:
    """Classify a waveform coarsely using envelope and instantaneous phase statistics."""

    samples = np.asarray(iq, dtype=np.complex64)
    if samples.size < 4:
        return ClassicalFamilyEstimate("unknown", 0.0, {})
    amplitude = np.abs(samples)
    phase = np.unwrap(np.angle(samples))
    frequency = np.diff(phase)
    amplitude_cv = float(np.std(amplitude) / (np.mean(amplitude) + 1e-12))
    frequency_cv = float(np.std(frequency) / (np.mean(np.abs(frequency)) + 1e-12))
    fourth_power_line = float(np.abs(np.mean(np.exp(4j * phase))))
    evidence = {
        "amplitude_cv": amplitude_cv,
        "frequency_cv": frequency_cv,
        "fourth_power_line": fourth_power_line,
        "envelope_flatness": _envelope_flatness(amplitude),
    }
    if (
        amplitude_cv < CLASSICAL_THRESHOLDS["amplitude_cv_psk"]
        and fourth_power_line >= CLASSICAL_THRESHOLDS["fourth_power_psk_strong"]
    ):
        family, confidence = "PSK-like", min(1.0, 0.6 + fourth_power_line / 2)
    elif amplitude_cv < CLASSICAL_THRESHOLDS["amplitude_cv_psk"] and frequency_cv > CLASSICAL_THRESHOLDS["frequency_cv_fsk"]:
        family, confidence = "FSK-like", min(1.0, 0.55 + frequency_cv / 4)
    elif amplitude_cv < CLASSICAL_THRESHOLDS["amplitude_cv_qam"] and fourth_power_line > CLASSICAL_THRESHOLDS["fourth_power_psk"]:
        family, confidence = "PSK-like", min(1.0, 0.5 + fourth_power_line / 2)
    elif (
        frequency_cv < ANALOG_FREQUENCY_CV_MAX
        and fourth_power_line < CLASSICAL_THRESHOLDS["fourth_power_psk"]
    ):
        # Positive analog evidence, not a leftover bucket: the instantaneous frequency
        # is smooth (no symbol-transition impulses) and there is no PSK carrier line.
        # Deliberately family-level - it says "analog", never which analog scheme.
        # Confidence scales with how far below the threshold the evidence sits.
        margin = 1.0 - frequency_cv / ANALOG_FREQUENCY_CV_MAX
        family = "analog-like"
        confidence = _ANALOG_CONFIDENCE_FLOOR + _ANALOG_CONFIDENCE_SPAN * margin
    elif amplitude_cv >= CLASSICAL_THRESHOLDS["amplitude_cv_qam"]:
        family, confidence = "QAM-like", min(1.0, 0.45 + amplitude_cv / 2)
    else:
        # Nothing matched. "unknown" rather than "analog-like": analog is now a
        # positive verdict and must not be handed out by elimination.
        family, confidence = "unknown", 0.2
    return ClassicalFamilyEstimate(family, confidence, evidence)
