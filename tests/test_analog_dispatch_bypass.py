"""Analog dispatch bypasses the digital symbol-rate path (BANK.md Entry 028).

Entry 025 measured the failure: `demodulate_capture` required a symbol rate, then
decimated by `sample_rate / symbol_rate_hz` before the analog branch, folding 2 of 3
message tones. These tests pin the bypass and, just as importantly, pin that the digital
branch keeps requiring a symbol rate.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.decoding.demodulators.dispatch import (
    ANALOG_LABELS, DispatchResult, demodulate_capture)
from radiofry.dsp.parameter_estimation import ParameterEstimate
from radiofry.synthetic_gen.v1.analog import (
    AnalogSampleSpec, generate_message, modulate_analog)
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import modulate

FS, N = 200_000.0, 8_192
CARRIER = 20_000.0


def _analog_signal(scheme: str, **kw):
    spec = AnalogSampleSpec(scheme=scheme, sample_rate_hz=FS, num_samples=N, seed=101, **kw)
    message, tones = generate_message(spec)
    return UnifiedSignalContainer(modulate_analog(message, spec), FS), message, tones


def _digital_signal(modulation: str, sps: int = 8):
    spec = SampleSpec(modulation=modulation, num_symbols=N // sps, samples_per_symbol=sps,
                      sample_rate_hz=FS, seed=101)
    bits = np.random.default_rng([101, 1]).integers(0, 2, spec.num_bits, dtype=np.uint8)
    return UnifiedSignalContainer(modulate(bits, spec), FS), bits


def _correlation(a, b) -> float:
    n = min(a.size, b.size)
    x, y = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / d) if d else 0.0


# --- the analog label set ----------------------------------------------------------------


def test_dispatch_publishes_the_analog_label_set() -> None:
    assert ANALOG_LABELS == frozenset({"AM-DSB", "AM-SSB", "WBFM"})


def test_the_analog_set_matches_what_fusion_can_emit() -> None:
    from radiofry.fusion.confidence_fusion import ANALOG_LABELS as FUSION_LABELS

    assert ANALOG_LABELS == FUSION_LABELS


# --- a null symbol rate must not block analog ---------------------------------------------


@pytest.mark.parametrize("modulation,scheme,kw", [
    ("AM-DSB", "am_dsb", {"carrier_offset_hz": CARRIER}),
    ("AM-SSB", "am_ssb", {"carrier_offset_hz": CARRIER, "sideband": "upper"}),
    ("WBFM", "wbfm", {"carrier_offset_hz": CARRIER, "frequency_deviation_hz": 15_000.0}),
])
def test_analog_demodulates_with_a_null_symbol_rate(modulation, scheme, kw) -> None:
    signal, _, _ = _analog_signal(scheme, **kw)

    dispatched = demodulate_capture(
        signal, modulation,
        ParameterEstimate(None, None, None, carrier_frequency_hz=CARRIER))

    assert dispatched.available is True
    assert dispatched.result.modulation == modulation


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "CPFSK", "GFSK", "QAM16", "QAM64"])
def test_digital_still_fails_on_a_null_symbol_rate(modulation: str) -> None:
    # Symbol rate must become optional for the analog branch ONLY.
    signal, _ = _digital_signal("BPSK")

    dispatched = demodulate_capture(signal, modulation, ParameterEstimate(None, None, None))

    assert dispatched.available is False
    assert "symbol-rate" in dispatched.message


# --- analog output must not depend on the symbol rate at all ------------------------------


@pytest.mark.parametrize("modulation,scheme,kw", [
    ("AM-DSB", "am_dsb", {"carrier_offset_hz": CARRIER}),
    ("AM-SSB", "am_ssb", {"carrier_offset_hz": CARRIER, "sideband": "upper"}),
    ("WBFM", "wbfm", {"carrier_offset_hz": CARRIER, "frequency_deviation_hz": 15_000.0}),
])
def test_a_wildly_different_symbol_rate_changes_nothing_for_analog(modulation, scheme, kw) -> None:
    signal, _, _ = _analog_signal(scheme, **kw)
    kwargs = dict(carrier_frequency_hz=CARRIER)

    a = demodulate_capture(signal, modulation, ParameterEstimate(None, None, None, **kwargs))
    b = demodulate_capture(signal, modulation, ParameterEstimate(None, None, 137.0, **kwargs))
    c = demodulate_capture(signal, modulation, ParameterEstimate(None, None, 91_000.0, **kwargs))

    assert np.array_equal(a.result.symbols, b.result.symbols)
    assert np.array_equal(a.result.symbols, c.result.symbols)


@pytest.mark.parametrize("modulation,scheme,kw", [
    ("AM-DSB", "am_dsb", {"carrier_offset_hz": CARRIER}),
    ("WBFM", "wbfm", {"carrier_offset_hz": CARRIER, "frequency_deviation_hz": 15_000.0}),
])
def test_analog_demodulators_receive_the_full_rate_capture(modulation, scheme, kw) -> None:
    # Decimation would leave far fewer samples than the capture holds.
    signal, _, _ = _analog_signal(scheme, **kw)

    dispatched = demodulate_capture(
        signal, modulation, ParameterEstimate(None, None, 2_000.0, carrier_frequency_hz=CARRIER))

    # demodulate_fm returns N-1 samples (it differentiates); AM returns N.
    assert dispatched.result.symbols.size >= N - 1


def test_the_dispatch_message_says_no_symbol_rate_was_used() -> None:
    signal, _, _ = _analog_signal("wbfm", carrier_offset_hz=CARRIER,
                                  frequency_deviation_hz=15_000.0)

    dispatched = demodulate_capture(
        signal, "WBFM", ParameterEstimate(None, None, 2_000.0, carrier_frequency_hz=CARRIER))

    assert "samples per symbol" not in dispatched.message
    assert "full-rate" in dispatched.message


# --- message recovery: the actual point of the change --------------------------------------


def test_am_dsb_recovery_survives_dispatch_now() -> None:
    signal, message, _ = _analog_signal("am_dsb", carrier_offset_hz=CARRIER)

    dispatched = demodulate_capture(
        signal, "AM-DSB", ParameterEstimate(None, None, 3_857.0, carrier_frequency_hz=CARRIER))

    assert _correlation(dispatched.result.symbols, message) > 0.99


def test_wbfm_recovery_survives_dispatch_now() -> None:
    signal, message, _ = _analog_signal("wbfm", carrier_offset_hz=CARRIER,
                                        frequency_deviation_hz=15_000.0)

    dispatched = demodulate_capture(
        signal, "WBFM", ParameterEstimate(None, None, 2_929.0, carrier_frequency_hz=CARRIER))

    assert _correlation(dispatched.result.symbols, message[1:]) > 0.99


def test_am_ssb_recovery_is_good_with_an_accurate_carrier() -> None:
    signal, message, _ = _analog_signal("am_ssb", carrier_offset_hz=CARRIER, sideband="upper")

    dispatched = demodulate_capture(
        signal, "AM-SSB", ParameterEstimate(None, None, None, carrier_frequency_hz=CARRIER))

    assert _correlation(dispatched.result.symbols, message) > 0.99


def test_am_ssb_still_fails_with_the_currently_estimated_carrier() -> None:
    """Separates dispatch corruption from carrier-estimation error.

    Dispatch is fixed; the estimator's 22147 Hz (true 20000 Hz) is a different problem
    and is NOT solved here. Pinned so the two are never conflated.
    """
    signal, message, _ = _analog_signal("am_ssb", carrier_offset_hz=CARRIER, sideband="upper")

    dispatched = demodulate_capture(
        signal, "AM-SSB", ParameterEstimate(None, None, None, carrier_frequency_hz=22_147.2))

    assert dispatched.available is True, "dispatch itself must still succeed"
    assert abs(_correlation(dispatched.result.symbols, message)) < 0.2


def test_ssb_uses_the_full_sample_rate_not_a_decimated_one() -> None:
    # A stale symbol rate previously scaled the SSB down-conversion rate; if that still
    # happened, an accurate carrier would no longer recover the message.
    signal, message, _ = _analog_signal("am_ssb", carrier_offset_hz=CARRIER, sideband="upper")

    dispatched = demodulate_capture(
        signal, "AM-SSB",
        ParameterEstimate(None, None, 2_001.0, carrier_frequency_hz=CARRIER))

    assert _correlation(dispatched.result.symbols, message) > 0.99


# --- digital behaviour must be untouched ------------------------------------------------------


@pytest.mark.parametrize("modulation,order", [("BPSK", 2), ("QPSK", 4)])
def test_digital_psk_dispatch_is_unchanged(modulation: str, order: int) -> None:
    signal, bits = _digital_signal(modulation)

    dispatched = demodulate_capture(signal, modulation,
                                    ParameterEstimate(None, None, FS / 8))

    assert dispatched.available is True
    assert dispatched.result.bits.size > 0
    assert "samples per symbol" in dispatched.message


@pytest.mark.parametrize("modulation", ["QAM16", "QAM64"])
def test_digital_qam_dispatch_is_unchanged(modulation: str) -> None:
    signal, _ = _digital_signal("16QAM" if modulation == "QAM16" else "64QAM")

    dispatched = demodulate_capture(signal, modulation, ParameterEstimate(None, None, FS / 8))

    assert dispatched.available is True
    assert "samples per symbol" in dispatched.message


def test_digital_fsk_still_uses_the_fsk_timing_search() -> None:
    signal, _ = _digital_signal("BFSK")

    dispatched = demodulate_capture(signal, "CPFSK", ParameterEstimate(None, None, FS / 8))

    assert dispatched.available is True
    assert "coarse timing search" in dispatched.message


def test_digital_still_requires_a_sample_rate() -> None:
    signal, _ = _digital_signal("BPSK")
    without_rate = UnifiedSignalContainer(signal.iq, None)

    dispatched = demodulate_capture(without_rate, "BPSK", ParameterEstimate(None, None, 25_000.0))

    assert dispatched.available is False


def test_unclassified_is_still_skipped_before_anything_else() -> None:
    signal, _, _ = _analog_signal("wbfm", carrier_offset_hz=CARRIER,
                                  frequency_deviation_hz=15_000.0)

    dispatched = demodulate_capture(signal, "Unclassified", ParameterEstimate(None, None, None))

    assert dispatched.available is False
    assert "unclassified" in dispatched.message.lower()


def test_an_unknown_label_is_still_rejected() -> None:
    signal, _ = _digital_signal("BPSK")

    dispatched = demodulate_capture(signal, "NBFM", ParameterEstimate(None, None, FS / 8))

    assert dispatched.available is False
    assert "No demodulator is registered" in dispatched.message


# --- API compatibility ----------------------------------------------------------------------------


def test_dispatch_result_semantics_are_unchanged() -> None:
    signal, _, _ = _analog_signal("am_dsb", carrier_offset_hz=CARRIER)

    dispatched = demodulate_capture(signal, "AM-DSB", ParameterEstimate(None, None, None))

    assert isinstance(dispatched, DispatchResult)
    assert isinstance(dispatched.available, bool)
    assert isinstance(dispatched.message, str)
    assert dispatched.result.modulation == "AM-DSB"


def test_analog_still_returns_the_threshold_bits_the_ber_guard_refuses() -> None:
    # Entry 021 behaviour: dispatch synthesises median-threshold bits for analog. They
    # remain physically meaningless and the harness must keep refusing to score them.
    signal, _, _ = _analog_signal("am_dsb", carrier_offset_hz=CARRIER)

    dispatched = demodulate_capture(signal, "AM-DSB", ParameterEstimate(None, None, None))

    assert dispatched.result.bits.size > 0
    assert set(np.unique(dispatched.result.bits)) <= {0, 1}
