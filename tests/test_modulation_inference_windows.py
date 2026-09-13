"""Regression tests for multi-window CNN inference.

Covers the change measured in BANK.md Entry 010: long captures are scored over
several contiguous native-rate windows and the softmax vectors are averaged.
Single-window behaviour, short/exact captures and the output contract are
unchanged.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.models.modulation_inference import (
    DEFAULT_INFERENCE_WINDOWS,
    _fixed_iq,
    _window_frames,
    predict_modulation,
)
from radiofry.synthetic_gen.v1 import SampleSpec, generate_source_bits, modulate

CHECKPOINT = "models_saved/modulation_cnn.pt"
FS, SPS, FRAME = 200_000.0, 8, 128


def _capture(modulation: str = "QPSK", num_symbols: int = 4_096, snr_db: float | None = 20.0):
    spec = SampleSpec(modulation=modulation, num_symbols=num_symbols, samples_per_symbol=SPS,
                      sample_rate_hz=FS, snr_db=snr_db, seed=3)
    from radiofry.synthetic_gen.v1 import add_awgn
    iq = modulate(generate_source_bits(spec), spec)
    if snr_db is not None:
        iq = add_awgn(iq, snr_db, np.random.default_rng(3))[0]
    return UnifiedSignalContainer(iq, FS)


# --- window extraction -----------------------------------------------------------


def test_default_window_count_is_documented_and_small() -> None:
    assert DEFAULT_INFERENCE_WINDOWS == 4


def test_multi_window_extraction_returns_the_requested_number_of_frames() -> None:
    frames = _window_frames(_capture(), FRAME, 4)

    assert len(frames) == 4
    assert all(frame.shape == (2, FRAME) for frame in frames)


def test_windows_are_contiguous_native_rate_slices_not_interpolated() -> None:
    # A tone at fs/16 must keep its 2*pi/16 rad per-sample advance in every window;
    # interpolating across the capture would fold it to another frequency.
    phase = 2 * np.pi * np.arange(32_768) / 16
    signal = UnifiedSignalContainer(np.exp(1j * phase).astype(np.complex64), FS)

    for frame in _window_frames(signal, FRAME, 4):
        increments = np.diff(np.unwrap(np.angle(frame[0] + 1j * frame[1])))
        np.testing.assert_allclose(increments, 2 * np.pi / 16, atol=1e-4)


def test_windows_span_the_capture_and_are_distinct() -> None:
    frames = _window_frames(_capture(), FRAME, 4)

    joined = {frame.tobytes() for frame in frames}
    assert len(joined) == 4


def test_single_window_extraction_matches_the_centred_frame() -> None:
    signal = _capture()

    np.testing.assert_array_equal(_window_frames(signal, FRAME, 1)[0], _fixed_iq(signal, FRAME))


# --- short and exact-length captures are unchanged --------------------------------


@pytest.mark.parametrize("windows", [1, 4, 32])
def test_short_capture_always_yields_exactly_one_unchanged_frame(windows: int) -> None:
    signal = UnifiedSignalContainer(np.ones(40, dtype=np.complex64), FS)

    frames = _window_frames(signal, FRAME, windows)

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], _fixed_iq(signal, FRAME))


@pytest.mark.parametrize("windows", [1, 4, 32])
def test_exact_length_capture_always_yields_exactly_one_unchanged_frame(windows: int) -> None:
    rng = np.random.default_rng(0)
    samples = (rng.normal(size=FRAME) + 1j * rng.normal(size=FRAME)).astype(np.complex64)
    signal = UnifiedSignalContainer(samples, FS)

    frames = _window_frames(signal, FRAME, windows)

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], _fixed_iq(signal, FRAME))


def test_empty_capture_is_still_rejected() -> None:
    signal = UnifiedSignalContainer(np.array([], dtype=np.complex64), FS)

    with pytest.raises(ValueError, match="empty"):
        _window_frames(signal, FRAME, 4)


# --- inference contract -----------------------------------------------------------


def test_single_window_inference_matches_scoring_the_centred_frame_alone() -> None:
    # Backward compatibility: windows=1 must reproduce the previous behaviour.
    signal = _capture()
    start = (signal.iq.size - FRAME) // 2
    centred = UnifiedSignalContainer(signal.iq[start : start + FRAME], FS)

    whole = predict_modulation(signal, CHECKPOINT, windows=1)
    alone = predict_modulation(centred, CHECKPOINT, windows=1)

    assert whole.label == alone.label
    assert whole.confidence == pytest.approx(alone.confidence, abs=1e-6)


def test_multi_window_inference_returns_the_same_output_contract() -> None:
    result = predict_modulation(_capture(), CHECKPOINT, windows=4)

    assert result.available is True
    assert isinstance(result.label, str)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k) == 3
    assert all(isinstance(label, str) and isinstance(value, float) for label, value in result.top_k)


def test_confidence_equals_the_top_ranked_aggregate_probability() -> None:
    result = predict_modulation(_capture(), CHECKPOINT, windows=4)

    assert result.top_k[0][0] == result.label
    assert result.confidence == pytest.approx(result.top_k[0][1])


def test_aggregate_probabilities_stay_on_the_softmax_scale() -> None:
    # Averaging probability vectors yields a probability vector, so fusion's 0.4
    # threshold keeps its meaning. A vote share would not.
    result = predict_modulation(_capture(), CHECKPOINT, windows=4)

    assert sum(value for _, value in result.top_k) <= 1.0 + 1e-6
    assert all(0.0 <= value <= 1.0 for _, value in result.top_k)
    assert result.top_k[0][1] >= result.top_k[1][1] >= result.top_k[2][1]


def test_inference_is_deterministic() -> None:
    signal = _capture()

    first = predict_modulation(signal, CHECKPOINT, windows=4)
    second = predict_modulation(signal, CHECKPOINT, windows=4)

    assert first.label == second.label
    assert first.confidence == pytest.approx(second.confidence, abs=1e-9)
    assert first.top_k == second.top_k


def test_window_count_is_configurable_and_defaults_are_applied() -> None:
    signal = _capture()

    explicit = predict_modulation(signal, CHECKPOINT, windows=DEFAULT_INFERENCE_WINDOWS)
    default = predict_modulation(signal, CHECKPOINT)

    assert default.label == explicit.label
    assert default.confidence == pytest.approx(explicit.confidence, abs=1e-9)


@pytest.mark.parametrize("windows", [0, -5])
def test_non_positive_window_counts_fall_back_to_a_single_window(windows: int) -> None:
    signal = _capture()

    result = predict_modulation(signal, CHECKPOINT, windows=windows)
    single = predict_modulation(signal, CHECKPOINT, windows=1)

    assert result.label == single.label
    assert result.confidence == pytest.approx(single.confidence, abs=1e-9)


def test_legacy_checkpoint_hashes_are_accepted_for_backward_compatibility() -> None:
    result = predict_modulation(_capture(num_symbols=64), "models_saved/modulation_cnn.pt", windows=4)

    assert result.available is True
    assert result.label != "Unclassified"
    assert result.confidence > 0.0


def test_missing_checkpoint_still_reports_unavailable() -> None:
    result = predict_modulation(_capture(num_symbols=64), "models_saved/does_not_exist.pt", windows=4)

    assert result.available is False
    assert result.label == "Unclassified"
    assert result.confidence == 0.0
