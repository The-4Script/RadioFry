"""Validity tests for the RRC diagnostic signals used in the feature comparison.

The comparison in `radiofry.evaluation.symbol_rate_experiment` is only meaningful
if the RRC-shaped signals it builds are genuine band-limited signals. These tests
check the shaping itself, not the estimator.

Reference: BANK.md Entry 004.
"""

import numpy as np
import pytest

from radiofry.evaluation.symbol_rate_experiment import (
    FEATURES,
    FS,
    SPS,
    TRUE_SYMBOL_RATE,
    build_signal,
    occupied_bandwidth,
    rrc_taps,
    source_symbols,
)

LINEAR = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]


def test_rrc_taps_are_symmetric_odd_length_and_unit_energy() -> None:
    taps = rrc_taps(0.35, span_symbols=10, samples_per_symbol=SPS)

    assert taps.size == 10 * SPS + 1
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-12)
    assert np.sum(taps**2) == pytest.approx(1.0, rel=1e-9)


def test_rrc_taps_handle_the_singular_points_without_nan() -> None:
    # beta = 0.25 places t = +/- T/(4*beta) exactly on a sample for sps = 8.
    taps = rrc_taps(0.25, span_symbols=10, samples_per_symbol=SPS)

    assert np.all(np.isfinite(taps))


@pytest.mark.parametrize("beta", [0.2, 0.35])
@pytest.mark.parametrize("modulation", LINEAR)
def test_rrc_signal_is_band_limited_to_roughly_one_plus_beta_times_symbol_rate(
    modulation: str, beta: float
) -> None:
    signal = build_signal(modulation, pulse="rrc", beta=beta, snr_db=None, seed=3)

    bandwidth = occupied_bandwidth(signal, FS)
    nominal = (1.0 + beta) * TRUE_SYMBOL_RATE

    assert 0.75 * nominal <= bandwidth <= 1.10 * nominal


def test_rect_signal_is_wider_than_the_equivalent_rrc_signal() -> None:
    rect = occupied_bandwidth(build_signal("QPSK", pulse="rect", snr_db=None, seed=3), FS)
    shaped = occupied_bandwidth(build_signal("QPSK", pulse="rrc", beta=0.35, snr_db=None, seed=3), FS)

    assert rect > shaped


@pytest.mark.parametrize("modulation", LINEAR)
def test_rrc_signal_recovers_its_source_symbols_through_a_matched_filter(modulation: str) -> None:
    # Proves the shaped waveform is a real, decodable RRC signal rather than noise.
    beta, span = 0.35, 10
    signal = build_signal(modulation, pulse="rrc", beta=beta, snr_db=None, seed=3)
    taps = rrc_taps(beta, span_symbols=span, samples_per_symbol=SPS)

    matched = np.convolve(signal, taps, mode="same")
    recovered = matched[:: SPS][span : -span]
    truth = source_symbols(modulation, seed=3)[span : -span]

    correlation = np.abs(np.vdot(truth, recovered)) / (np.linalg.norm(truth) * np.linalg.norm(recovered))
    assert correlation > 0.99


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK"])
def test_rrc_shaping_gives_constant_modulus_psk_a_varying_envelope(modulation: str) -> None:
    # This is the property that makes the current |x|^2 feature viable on shaped
    # signals but not on V1's rectangular pulses.
    rect = build_signal(modulation, pulse="rect", snr_db=None, seed=3)
    shaped = build_signal(modulation, pulse="rrc", beta=0.35, snr_db=None, seed=3)

    assert np.var(np.abs(rect) ** 2) < 1e-12
    assert np.var(np.abs(shaped) ** 2) > 0.01


def test_shaped_cpfsk_stays_constant_modulus() -> None:
    # CPFSK is exp(j*phase) by construction, so shaping the frequency pulse cannot
    # create envelope variation. The |x|^2 feature stays blind to it.
    shaped = build_signal("BFSK", pulse="rrc", beta=0.35, snr_db=None, seed=3)

    assert np.var(np.abs(shaped) ** 2) < 1e-9


def test_rect_signals_come_from_the_unmodified_v1_generator() -> None:
    from radiofry.synthetic_gen.v1 import SampleSpec, generate_source_bits, modulate

    spec = SampleSpec(
        modulation="QPSK",
        num_symbols=source_symbols("QPSK", seed=3).size,
        samples_per_symbol=SPS,
        sample_rate_hz=FS,
        snr_db=None,
        seed=3,
    )
    expected = modulate(generate_source_bits(spec), spec)

    np.testing.assert_allclose(build_signal("QPSK", pulse="rect", snr_db=None, seed=3), expected, atol=0)


def test_awgn_is_applied_when_an_snr_is_requested() -> None:
    clean = build_signal("QPSK", pulse="rrc", beta=0.35, snr_db=None, seed=3)
    noisy = build_signal("QPSK", pulse="rrc", beta=0.35, snr_db=0.0, seed=3)

    measured = 10 * np.log10(np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noisy - clean) ** 2))
    assert abs(measured) < 0.5


def test_all_three_candidate_features_are_registered() -> None:
    assert list(FEATURES) == ["abs_x_squared", "abs_diff_x_squared", "phase_second_difference_squared"]
