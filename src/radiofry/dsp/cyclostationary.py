"""Lightweight non-ML modulation-family cross-checks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassicalFamilyEstimate:
    family: str
    confidence: float
    evidence: dict[str, float]


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
    if amplitude_cv < 0.15 and frequency_cv > 0.8:
        family, confidence = "FSK-like", min(1.0, 0.55 + frequency_cv / 4)
    elif amplitude_cv < 0.2 and fourth_power_line > 0.2:
        family, confidence = "PSK-like", min(1.0, 0.5 + fourth_power_line / 2)
    elif amplitude_cv >= 0.2:
        family, confidence = "QAM-like", min(1.0, 0.45 + amplitude_cv / 2)
    else:
        family, confidence = "analog-like", 0.45
    return ClassicalFamilyEstimate(family, confidence, evidence)
