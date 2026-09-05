import numpy as np
import pytest

from radiofry.dsp.cyclostationary import estimate_modulation_family
from radiofry.dsp.parameter_estimation import _select_symbol_rate, estimate_parameters
from radiofry.fusion.confidence_fusion import fuse_modulation
from radiofry.contracts import UnifiedSignalContainer
from radiofry.training.evaluate_classical import evaluate_classical_samples
from radiofry.models.modulation_metrics import expected_accuracy_at_snr


def test_parameter_estimation_returns_bandwidth_and_symbol_rate() -> None:
    sample_rate = 8_000
    time = np.arange(8_000) / sample_rate
    signal = UnifiedSignalContainer(np.exp(2j * np.pi * 500 * time), sample_rate)

    estimate = estimate_parameters(signal)

    assert estimate.occupied_bandwidth_hz is not None
    assert estimate.symbol_rate_hz is not None
    assert estimate.carrier_frequency_hz == pytest.approx(500.0, abs=8_000 / 1024)
    assert estimate.method == "welch_psd_centroid+nth_power"


def test_parameter_estimation_prefers_reported_center_frequency() -> None:
    sample_rate = 8_000
    time = np.arange(8_000) / sample_rate
    signal = UnifiedSignalContainer(
        np.exp(2j * np.pi * 500 * time),
        sample_rate,
        metadata={"center_frequency_hz": 915_000_000},
    )

    estimate = estimate_parameters(signal)

    assert estimate.carrier_frequency_hz == 915_000_000
    assert estimate.method == "hardware_center_frequency+welch_nth_power"


def test_symbol_rate_selector_prefers_supported_fundamental() -> None:
    rates = np.arange(1_001, dtype=float)
    spectrum = np.zeros(1_001, dtype=float)
    spectrum[100] = 8.0
    spectrum[200] = 10.0
    spectrum[300] = 7.0
    spectrum[400] = 5.0

    rate, confidence = _select_symbol_rate(rates, spectrum, np.array([100, 200, 300, 400]), np.array([8.0, 10.0, 7.0, 5.0]))

    assert rate == 100.0
    assert confidence is not None


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


def test_classical_evaluation_returns_snr_metrics() -> None:
    samples = [np.exp(1j * np.linspace(0, 20, 128)), np.exp(1j * np.linspace(0, 10, 128))]

    metrics = evaluate_classical_samples(samples, ["BPSK", "QPSK"], [-10, 0])

    assert metrics["samples"] == 2
    assert set(metrics["accuracy_by_snr"]) == {"-10", "0"}
    assert "confusion_matrix" in metrics


def test_expected_cnn_accuracy_interpolates_and_clamps() -> None:
    metrics = {"accuracy_by_snr": {"-10": 0.2, "0": 0.8}}

    assert expected_accuracy_at_snr(metrics, -20) == pytest.approx(0.2)
    assert expected_accuracy_at_snr(metrics, -5) == pytest.approx(0.5)
    assert expected_accuracy_at_snr(metrics, 10) == pytest.approx(0.8)
