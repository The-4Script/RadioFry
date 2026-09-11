"""One bad sample must not destroy a capture.

`preprocess` removes DC by subtracting the mean. `np.mean` of an array containing a single
NaN or Inf is non-finite, and subtracting it propagated that to **every** sample - so one
dropped sample from an SDR, or one gap in a WAV, turned 8192 good samples into 8192 bad
ones. Downstream the damage was silent: every parameter came back `None` while the
classifier still produced a label, so a report looked like a classified capture with
missing measurements rather than a corrupted one (BANK.md Entry 042).

Real recordings contain dropouts. These tests pin the repair and the honesty of it.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.parameter_estimation import estimate_parameters
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import generate_source_bits, modulate

FS = 200_000.0


def _capture(samples: int = 8_192, seed: int = 99_001) -> np.ndarray:
    spec = SampleSpec(modulation="QPSK", num_symbols=samples // 8,
                      samples_per_symbol=8, sample_rate_hz=FS, snr_db=20.0, seed=seed)
    clean = modulate(generate_source_bits(spec), spec)
    return np.asarray(add_awgn(clean, spec.snr_db,
                               np.random.default_rng([spec.seed, 2]))[0])


def _wrap(iq: np.ndarray) -> UnifiedSignalContainer:
    return UnifiedSignalContainer(np.asarray(iq, dtype=complex), FS, "iq")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_single_bad_sample_does_not_contaminate_the_capture(bad: float) -> None:
    capture = _capture()
    damaged = np.where(np.arange(capture.size) == 100, bad, capture)

    processed = preprocess(_wrap(damaged))

    assert np.all(np.isfinite(processed.iq)), (
        "one non-finite sample must not propagate through the mean subtraction")
    assert processed.iq.size == capture.size


def test_the_repair_is_counted_and_reported() -> None:
    capture = _capture()
    positions = np.arange(0, 5_000, 100)
    damaged = np.where(np.isin(np.arange(capture.size), positions), np.nan, capture)

    processed = preprocess(_wrap(damaged))

    assert processed.metadata["non_finite_samples"] == positions.size, (
        "the caller must be able to report how much of the capture was unusable")


def test_a_clean_capture_reports_zero_and_is_unchanged() -> None:
    capture = _capture()

    clean = preprocess(_wrap(capture))

    assert clean.metadata["non_finite_samples"] == 0
    assert np.all(np.isfinite(clean.iq))


def test_a_clean_capture_is_bit_identical_to_the_previous_behaviour() -> None:
    """The repair must be inert for data that never needed it."""
    capture = _capture()

    processed = preprocess(_wrap(capture))

    reference = capture.astype(np.complex64)
    reference = reference - np.mean(reference, dtype=np.complex64)
    reference = reference / float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    assert np.allclose(processed.iq, reference, atol=1e-6)


def test_parameters_survive_a_dropout_that_previously_erased_them() -> None:
    """The symptom that exposed the defect: every estimate became None."""
    capture = _capture()
    damaged = np.where(np.arange(capture.size) == 100, np.nan, capture)

    clean = estimate_parameters(preprocess(_wrap(capture)))
    repaired = estimate_parameters(preprocess(_wrap(damaged)))

    for name in ("snr_db", "occupied_bandwidth_hz", "symbol_rate_hz"):
        assert getattr(repaired, name) is not None, f"{name} was erased by one sample"
    assert repaired.symbol_rate_hz == pytest.approx(clean.symbol_rate_hz, rel=0.05)
    assert repaired.snr_db == pytest.approx(clean.snr_db, abs=1.0)


def test_a_wholly_non_finite_capture_still_degrades_gracefully() -> None:
    """Zeroing everything is the honest outcome; it must not raise or fabricate."""
    processed = preprocess(_wrap(np.full(1_024, np.nan, dtype=complex)))

    assert np.all(np.isfinite(processed.iq))
    assert processed.metadata["non_finite_samples"] == 1_024
    assert np.allclose(processed.iq, 0.0), "no signal must be invented from nothing"


def test_the_whole_pipeline_survives_a_dropout() -> None:
    from radiofry.pipeline import analyze_capture

    capture = _capture()
    damaged = np.where(np.arange(capture.size) == 100, np.nan, capture)

    report = analyze_capture(_wrap(damaged))

    parameters = report["stages"]["parameters"]
    assert parameters["snr_db"] is not None
    assert parameters["symbol_rate_hz"] is not None
    assert report["stages"]["fusion"]["label"] is not None
