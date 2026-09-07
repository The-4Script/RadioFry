"""Scoring-primitive tests for the Synthetic V1 evaluation harness."""

import numpy as np
import pytest

from radiofry.evaluation.metrics import (
    best_aligned_ber,
    bit_error_rate,
    confusion_matrix,
    expected_family_for,
    relative_error,
    score_bits,
    signed_error,
)


def test_bit_error_rate_is_zero_for_identical_streams() -> None:
    bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)

    assert bit_error_rate(bits, bits) == 0.0


def test_bit_error_rate_is_one_when_every_bit_is_inverted() -> None:
    truth = np.array([0, 0, 1, 1], dtype=np.uint8)

    assert bit_error_rate(truth, 1 - truth) == 1.0


def test_bit_error_rate_compares_only_the_overlapping_prefix() -> None:
    truth = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
    recovered = np.array([0, 1, 1], dtype=np.uint8)

    assert bit_error_rate(truth, recovered) == pytest.approx(1 / 3)


def test_bit_error_rate_of_an_empty_stream_is_none() -> None:
    assert bit_error_rate(np.array([0, 1], dtype=np.uint8), np.array([], dtype=np.uint8)) is None


def test_best_aligned_ber_recovers_a_late_starting_stream() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 200, dtype=np.uint8)

    ber, offset = best_aligned_ber(truth, truth[5:], max_offset=32)

    assert ber == 0.0
    assert offset == 5


def test_best_aligned_ber_recovers_an_early_starting_stream() -> None:
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, 200, dtype=np.uint8)
    recovered = np.concatenate((rng.integers(0, 2, 4, dtype=np.uint8), truth))

    ber, offset = best_aligned_ber(truth, recovered, max_offset=32)

    assert ber == 0.0
    assert offset == -4


def test_best_aligned_ber_reports_the_unshifted_result_when_nothing_aligns() -> None:
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 2, 500, dtype=np.uint8)
    recovered = rng.integers(0, 2, 500, dtype=np.uint8)

    ber, _ = best_aligned_ber(truth, recovered, max_offset=8)

    assert 0.3 < ber < 0.7


def test_score_bits_marks_unavailable_when_demodulation_did_not_run() -> None:
    truth = np.array([0, 1, 0, 1], dtype=np.uint8)

    result = score_bits(truth, None, available=False, reason="demodulation_unavailable")

    assert result.status == "unavailable"
    assert result.reason == "demodulation_unavailable"
    assert result.strict is None
    assert result.aligned is None


def test_score_bits_marks_unavailable_for_an_empty_recovered_stream() -> None:
    truth = np.array([0, 1, 0, 1], dtype=np.uint8)

    result = score_bits(truth, np.array([], dtype=np.uint8), available=True)

    assert result.status == "unavailable"
    assert result.reason == "no_recovered_bits"


def test_score_bits_reports_strict_and_aligned_rates_with_counts() -> None:
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, 128, dtype=np.uint8)

    result = score_bits(truth, truth[3:], available=True)

    assert result.status == "ok"
    assert result.aligned == 0.0
    assert result.alignment_offset == 3
    assert result.strict > 0.0
    assert result.expected_bits == 128
    assert result.recovered_bits == 125
    assert result.compared_bits == 125


def test_expected_family_maps_v1_families_to_radiofry_vocabulary() -> None:
    assert expected_family_for("psk") == "PSK-like"
    assert expected_family_for("qam") == "QAM-like"
    assert expected_family_for("fsk") == "FSK-like"


def test_expected_family_rejects_an_unknown_family() -> None:
    with pytest.raises(KeyError):
        expected_family_for("analog")


def test_signed_error_and_relative_error_propagate_missing_estimates() -> None:
    assert signed_error(10.5, 10.0) == pytest.approx(0.5)
    assert signed_error(None, 10.0) is None
    assert relative_error(26_000.0, 25_000.0) == pytest.approx(0.04)
    assert relative_error(26_000.0, 0.0) is None
    assert relative_error(None, 25_000.0) is None


def test_confusion_matrix_counts_truth_prediction_pairs() -> None:
    pairs = [("BPSK", "BPSK"), ("BPSK", "QPSK"), ("QPSK", "QPSK")]

    matrix = confusion_matrix(pairs)

    assert matrix["BPSK"]["BPSK"] == 1
    assert matrix["BPSK"]["QPSK"] == 1
    assert matrix["QPSK"]["QPSK"] == 1
    assert matrix["QPSK"].get("BPSK", 0) == 0
