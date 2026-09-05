import numpy as np

from radiofry.dsp.cyclostationary import estimate_modulation_family
from radiofry.dsp.parameter_estimation import estimate_parameters
from radiofry.fusion.confidence_fusion import fuse_modulation
from radiofry.contracts import UnifiedSignalContainer


def test_parameter_estimation_returns_bandwidth_and_symbol_rate() -> None:
    sample_rate = 8_000
    time = np.arange(8_000) / sample_rate
    signal = UnifiedSignalContainer(np.exp(2j * np.pi * 500 * time), sample_rate)

    estimate = estimate_parameters(signal)

    assert estimate.occupied_bandwidth_hz is not None
    assert estimate.symbol_rate_hz is not None


def test_fusion_rejects_low_confidence_and_penalizes_disagreement() -> None:
    result = fuse_modulation("QPSK", 0.3, "FSK-like")

    assert result.label == "Unclassified"
    assert result.review_recommended
    assert result.trust_score == 0.15


def test_classical_estimator_returns_a_structured_result() -> None:
    samples = np.exp(1j * np.linspace(0, 20, 128))

    result = estimate_modulation_family(samples)

    assert result.family
    assert 0 <= result.confidence <= 1
    assert set(result.evidence) == {"amplitude_cv", "frequency_cv", "fourth_power_line"}
