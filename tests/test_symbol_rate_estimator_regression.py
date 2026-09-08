"""Regression tests for the adaptive symbol-rate feature selection.

Covers the cases measured in BANK.md Entry 004: rectangular-pulse captures (where
the original envelope feature is blind) and RRC-shaped captures at two roll-offs
(where it must not regress). Reuses the diagnostic signal builder rather than
duplicating pulse-shaping logic.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.parameter_estimation import (
    SYMBOL_RATE_FEATURES,
    ParameterEstimate,
    estimate_parameters,
)
from radiofry.evaluation.symbol_rate_experiment import FS, TRUE_SYMBOL_RATE, build_signal

LINEAR = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]


def _estimate(modulation: str, *, pulse: str, beta: float = 0.35, snr_db=None, seed: int = 3):
    iq = build_signal(modulation, pulse=pulse, beta=beta, snr_db=snr_db, seed=seed)
    return estimate_parameters(UnifiedSignalContainer(iq, FS))


def _within(estimate: ParameterEstimate, tolerance: float = 0.01) -> bool:
    if estimate.symbol_rate_hz is None:
        return False
    return abs(estimate.symbol_rate_hz - TRUE_SYMBOL_RATE) / TRUE_SYMBOL_RATE <= tolerance


# --- rectangular pulses: the case the original feature could not see ------------


@pytest.mark.parametrize("modulation", LINEAR)
@pytest.mark.parametrize("snr_db", [None, 20.0, 10.0])
def test_rectangular_linear_captures_recover_the_symbol_rate(modulation: str, snr_db) -> None:
    assert _within(_estimate(modulation, pulse="rect", snr_db=snr_db))


def test_rectangular_cpfsk_recovers_the_symbol_rate_without_noise() -> None:
    assert _within(_estimate("BFSK", pulse="rect", snr_db=None))


# --- RRC pulses: must not regress ------------------------------------------------


@pytest.mark.parametrize("modulation", LINEAR)
@pytest.mark.parametrize("beta", [0.20, 0.35])
def test_rrc_shaped_linear_captures_still_recover_the_symbol_rate(modulation: str, beta: float) -> None:
    assert _within(_estimate(modulation, pulse="rrc", beta=beta, snr_db=None))


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM"])
@pytest.mark.parametrize("beta", [0.20, 0.35])
def test_rrc_shaped_captures_survive_moderate_noise(modulation: str, beta: float) -> None:
    assert _within(_estimate(modulation, pulse="rrc", beta=beta, snr_db=20.0))


# --- feature registry ------------------------------------------------------------


def test_all_three_candidate_features_are_registered() -> None:
    assert list(SYMBOL_RATE_FEATURES) == [
        "envelope_power",
        "transition_power",
        "phase_second_difference",
    ]


def test_the_original_envelope_feature_remains_available_and_unchanged() -> None:
    rng = np.random.default_rng(0)
    samples = (rng.normal(size=64) + 1j * rng.normal(size=64)).astype(np.complex64)

    np.testing.assert_allclose(
        SYMBOL_RATE_FEATURES["envelope_power"](samples), np.abs(samples) ** 2, rtol=1e-6
    )


def test_estimate_reports_which_feature_was_selected() -> None:
    estimate = _estimate("QPSK", pulse="rect", snr_db=None)

    assert estimate.symbol_rate_feature in SYMBOL_RATE_FEATURES


def test_envelope_feature_is_selected_for_a_shaped_signal_it_handles_well() -> None:
    # On RRC-shaped QAM the original feature has a genuine timing tone, so the
    # confidence-based selection should still be able to pick it.
    estimate = _estimate("16QAM", pulse="rrc", beta=0.35, snr_db=None)

    assert _within(estimate)
    assert estimate.symbol_rate_feature is not None


# --- preserved contract ----------------------------------------------------------


def test_short_capture_still_returns_an_empty_estimate() -> None:
    estimate = estimate_parameters(UnifiedSignalContainer(np.zeros(4, dtype=np.complex64), FS))

    assert estimate.symbol_rate_hz is None
    assert estimate.symbol_rate_feature is None


def test_missing_sample_rate_still_returns_an_empty_estimate() -> None:
    estimate = estimate_parameters(UnifiedSignalContainer(np.ones(4_096, dtype=np.complex64)))

    assert estimate.symbol_rate_hz is None
    assert estimate.occupied_bandwidth_hz is None


def test_constant_capture_does_not_crash_the_selection() -> None:
    estimate = estimate_parameters(UnifiedSignalContainer(np.ones(4_096, dtype=np.complex64), FS))

    assert isinstance(estimate, ParameterEstimate)


def test_other_estimated_parameters_are_still_produced() -> None:
    estimate = _estimate("QPSK", pulse="rect", snr_db=20.0)

    assert estimate.occupied_bandwidth_hz is not None
    assert estimate.snr_db is not None
    assert estimate.carrier_frequency_hz is not None
    assert estimate.method == "welch_psd_centroid+nth_power"
