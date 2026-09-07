"""Input-adapter tests for CNN modulation inference.

The CNN consumes a fixed-length frame. These tests pin the requirement that a
long capture is sampled as a contiguous window at the native sample rate rather
than decimated across the whole capture, which aliases the signal away.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.models.modulation_inference import _fixed_iq

SAMPLE_RATE = 200_000.0
FRAME = 128


def _tone(num_samples: int, cycles_per_sample: float = 1 / 16) -> UnifiedSignalContainer:
    phase = 2 * np.pi * cycles_per_sample * np.arange(num_samples)
    return UnifiedSignalContainer(np.exp(1j * phase).astype(np.complex64), SAMPLE_RATE)


def _as_complex(frame: np.ndarray) -> np.ndarray:
    return frame[0] + 1j * frame[1]


def test_long_capture_preserves_the_signal_frequency_instead_of_aliasing_it() -> None:
    # A tone at fs/16 advances 2*pi/16 rad per sample. Decimating a 32768-sample
    # capture down to 128 points steps ~258 samples at a time and folds this to a
    # completely different apparent frequency.
    signal = _tone(32_768, cycles_per_sample=1 / 16)

    frame = _fixed_iq(signal, FRAME)

    increments = np.diff(np.unwrap(np.angle(_as_complex(frame))))
    np.testing.assert_allclose(increments, 2 * np.pi / 16, atol=1e-4)


def test_long_capture_frame_is_a_contiguous_window_of_the_capture() -> None:
    rng = np.random.default_rng(0)
    samples = (rng.normal(size=8_192) + 1j * rng.normal(size=8_192)).astype(np.complex64)
    signal = UnifiedSignalContainer(samples, SAMPLE_RATE)

    frame = _fixed_iq(signal, FRAME)

    recovered = _as_complex(frame)
    start = (samples.size - FRAME) // 2
    expected = samples[start : start + FRAME]
    scale = np.linalg.norm(recovered) / np.linalg.norm(expected)
    np.testing.assert_allclose(recovered, expected * scale, atol=1e-4)


def test_frame_shape_and_dtype_are_unchanged() -> None:
    frame = _fixed_iq(_tone(32_768), FRAME)

    assert frame.shape == (2, FRAME)
    assert frame.dtype == np.float32


def test_frame_is_still_rms_normalised() -> None:
    samples = (3.0 + 4.0j) * np.exp(1j * 2 * np.pi * np.arange(4_096) / 16)
    signal = UnifiedSignalContainer(samples.astype(np.complex64), SAMPLE_RATE)

    frame = _fixed_iq(signal, FRAME)

    np.testing.assert_allclose(np.sqrt(np.mean(frame.astype(np.float64) ** 2)), 1.0, atol=1e-5)


def test_capture_of_exactly_frame_length_is_returned_unchanged_up_to_scaling() -> None:
    rng = np.random.default_rng(1)
    samples = (rng.normal(size=FRAME) + 1j * rng.normal(size=FRAME)).astype(np.complex64)
    signal = UnifiedSignalContainer(samples, SAMPLE_RATE)

    recovered = _as_complex(_fixed_iq(signal, FRAME))

    scale = np.linalg.norm(recovered) / np.linalg.norm(samples)
    np.testing.assert_allclose(recovered, samples * scale, atol=1e-4)


def test_short_capture_is_still_stretched_to_the_frame_length() -> None:
    signal = _tone(40)

    frame = _fixed_iq(signal, FRAME)

    assert frame.shape == (2, FRAME)
    assert np.all(np.isfinite(frame))


def test_single_sample_capture_does_not_crash() -> None:
    signal = UnifiedSignalContainer(np.array([1 + 1j], dtype=np.complex64), SAMPLE_RATE)

    frame = _fixed_iq(signal, FRAME)

    assert frame.shape == (2, FRAME)


def test_empty_capture_is_rejected() -> None:
    signal = UnifiedSignalContainer(np.array([], dtype=np.complex64), SAMPLE_RATE)

    with pytest.raises(ValueError, match="empty"):
        _fixed_iq(signal, FRAME)


def test_all_zero_capture_does_not_divide_by_zero() -> None:
    signal = UnifiedSignalContainer(np.zeros(1_024, dtype=np.complex64), SAMPLE_RATE)

    frame = _fixed_iq(signal, FRAME)

    assert np.all(frame == 0.0)
