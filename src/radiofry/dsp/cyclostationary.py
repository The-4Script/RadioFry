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
}


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
    }
    if amplitude_cv < CLASSICAL_THRESHOLDS["amplitude_cv_psk"] and frequency_cv > CLASSICAL_THRESHOLDS["frequency_cv_fsk"]:
        family, confidence = "FSK-like", min(1.0, 0.55 + frequency_cv / 4)
    elif amplitude_cv < CLASSICAL_THRESHOLDS["amplitude_cv_qam"] and fourth_power_line > CLASSICAL_THRESHOLDS["fourth_power_psk"]:
        family, confidence = "PSK-like", min(1.0, 0.5 + fourth_power_line / 2)
    elif amplitude_cv >= CLASSICAL_THRESHOLDS["amplitude_cv_qam"]:
        family, confidence = "QAM-like", min(1.0, 0.45 + amplitude_cv / 2)
    else:
        family, confidence = "analog-like", 0.45
    return ClassicalFamilyEstimate(family, confidence, evidence)
