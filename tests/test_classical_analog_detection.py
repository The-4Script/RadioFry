"""Positive family-level analog evidence in the classical detector (BANK.md Entry 027).

Entry 025 found `analog-like` was the bare `else` after three digital tests - a
fallback-by-elimination, not a detector. These tests pin a positive test based on
`frequency_cv`, the impulsiveness of the instantaneous frequency: a digital signal jumps
at symbol boundaries and sits still between them, so std(dphi) >> mean|dphi|; an analog
signal's instantaneous frequency moves smoothly and continuously.

The detector still decides at FAMILY level only. It never names an analog modulation.
"""

import numpy as np
import pytest

from radiofry.dsp.cyclostationary import (
    ANALOG_FREQUENCY_CV_MAX,
    CLASSICAL_THRESHOLDS,
    estimate_modulation_family,
)
from radiofry.dsp.preprocessing import preprocess
from radiofry.contracts import UnifiedSignalContainer
from radiofry.synthetic_gen.v1.analog import (
    AnalogSampleSpec, generate_message, modulate_analog)
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import modulate

FS, N = 200_000.0, 8_192


def _analog(scheme: str, snr_db: float | None = 20.0, seed: int = 101, **kw):
    spec = AnalogSampleSpec(scheme=scheme, sample_rate_hz=FS, num_samples=N,
                            snr_db=snr_db, seed=seed, **kw)
    clean = modulate_analog(generate_message(spec)[0], spec)
    iq = clean if snr_db is None else add_awgn(
        clean, snr_db, np.random.default_rng([seed, 2]))[0]
    return preprocess(UnifiedSignalContainer(iq, FS)).iq


def _digital(modulation: str, snr_db: float = 20.0, seed: int = 101, sps: int = 8):
    spec = SampleSpec(modulation=modulation, num_symbols=N // sps, samples_per_symbol=sps,
                      sample_rate_hz=FS, snr_db=snr_db, seed=seed)
    bits = np.random.default_rng([seed, 1]).integers(0, 2, spec.num_bits, dtype=np.uint8)
    iq = add_awgn(modulate(bits, spec), snr_db, np.random.default_rng([seed, 2]))[0]
    return preprocess(UnifiedSignalContainer(iq, FS)).iq


# --- the new positive test ------------------------------------------------------------


def test_the_analog_threshold_is_published_alongside_the_others() -> None:
    assert ANALOG_FREQUENCY_CV_MAX == 0.9
    # Reuses the existing fourth-power constant rather than inventing a second one.
    assert CLASSICAL_THRESHOLDS["fourth_power_psk"] == 0.2


@pytest.mark.parametrize("scheme,kw", [
    ("am_ssb", {"carrier_offset_hz": 20_000.0, "sideband": "upper"}),
    ("wbfm", {"carrier_offset_hz": 20_000.0, "frequency_deviation_hz": 15_000.0}),
    ("am_dsb", {"carrier_offset_hz": 20_000.0}),
])
def test_analog_schemes_are_detected_at_usable_snr(scheme: str, kw: dict) -> None:
    assert estimate_modulation_family(_analog(scheme, 20.0, **kw)).family == "analog-like"


@pytest.mark.parametrize("snr_db", [20.0, 15.0, 10.0])
def test_am_ssb_is_detected_across_usable_snr(snr_db: float) -> None:
    iq = _analog("am_ssb", snr_db, carrier_offset_hz=20_000.0, sideband="upper")

    assert estimate_modulation_family(iq).family == "analog-like"


def test_analog_confidence_is_evidence_based_not_a_flat_constant() -> None:
    # Two analog captures with clearly different frequency_cv must not report the
    # same hard-coded 0.45 the old fallback returned.
    strong = estimate_modulation_family(_analog("am_dsb", 20.0, carrier_offset_hz=20_000.0))
    weak = estimate_modulation_family(_analog("am_ssb", 10.0, carrier_offset_hz=20_000.0,
                                              sideband="upper"))

    assert strong.family == weak.family == "analog-like"
    assert strong.confidence > weak.confidence
    assert 0.5 <= weak.confidence <= strong.confidence <= 0.9


def test_the_evidence_dictionary_still_exposes_the_three_features() -> None:
    estimate = estimate_modulation_family(_analog("wbfm", 20.0, carrier_offset_hz=20_000.0,
                                                 frequency_deviation_hz=15_000.0))

    assert set(estimate.evidence) == {"amplitude_cv", "frequency_cv", "fourth_power_line"}
    assert estimate.evidence["frequency_cv"] < ANALOG_FREQUENCY_CV_MAX


# --- digital controls must not move ------------------------------------------------------


@pytest.mark.parametrize("modulation,expected", [
    ("BPSK", "FSK-like"), ("QPSK", "FSK-like"), ("8PSK", "FSK-like"),
    ("BFSK", "FSK-like"), ("GFSK", "FSK-like"),
    ("16QAM", "QAM-like"), ("64QAM", "QAM-like"), ("PAM4", "QAM-like"),
])
def test_digital_controls_keep_their_existing_family(modulation: str, expected: str) -> None:
    # Pins CURRENT behaviour, including the pre-existing quirk that high-SNR PSK is
    # reported FSK-like. This entry must not change any of it.
    assert estimate_modulation_family(_digital(modulation, 20.0)).family == expected


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "16QAM", "64QAM", "BFSK", "GFSK", "PAM4"])
@pytest.mark.parametrize("snr_db", [20.0, 10.0])
def test_no_digital_control_is_ever_called_analog(modulation: str, snr_db: float) -> None:
    assert estimate_modulation_family(_digital(modulation, snr_db)).family != "analog-like"


@pytest.mark.parametrize("sps", [4, 16, 32])
def test_heavier_oversampling_does_not_make_digital_look_analog(sps: int) -> None:
    # The nearest digital neighbour measured was BFSK at sps=4 (frequency_cv 1.032).
    for modulation in ("BFSK", "GFSK", "QPSK"):
        estimate = estimate_modulation_family(_digital(modulation, 20.0, sps=sps))
        assert estimate.family != "analog-like", f"{modulation} sps={sps}"
        assert estimate.evidence["frequency_cv"] > ANALOG_FREQUENCY_CV_MAX


# --- decision boundary and ordering --------------------------------------------------------


def test_fsk_keeps_priority_over_the_analog_test() -> None:
    # A constant-envelope signal with an impulsive instantaneous frequency is FSK,
    # and the FSK branch is evaluated first.
    estimate = estimate_modulation_family(_digital("BFSK", 20.0))

    assert estimate.family == "FSK-like"


def test_a_strong_fourth_power_line_blocks_the_analog_verdict() -> None:
    # A real-valued signal has phase 0, so exp(4j*phase) has a perfect line: that is
    # carrier structure, not analog evidence, and must veto the analog branch even
    # though its frequency_cv is 0.
    real_signal = np.linspace(0.5, 1.5, 4_096).astype(np.complex64)

    estimate = estimate_modulation_family(real_signal)

    assert estimate.evidence["frequency_cv"] < ANALOG_FREQUENCY_CV_MAX
    assert estimate.evidence["fourth_power_line"] > CLASSICAL_THRESHOLDS["fourth_power_psk"]
    assert estimate.family != "analog-like"


def test_analog_like_is_no_longer_the_generic_fallback() -> None:
    import inspect
    from radiofry.dsp import cyclostationary

    source = inspect.getsource(cyclostationary.estimate_modulation_family)
    # Ignore comments: it is the assignment in the final `else` that matters.
    tail = "\n".join(line for line in source.rsplit("else:", 1)[1].splitlines()
                     if not line.strip().startswith("#"))

    assert "analog-like" not in tail, "analog-like is still the catch-all else branch"
    assert '"unknown"' in tail


# --- API compatibility --------------------------------------------------------------------


def test_short_input_still_returns_unknown() -> None:
    estimate = estimate_modulation_family(np.zeros(2, dtype=np.complex64))

    assert estimate.family == "unknown"
    assert estimate.confidence == 0.0
    assert estimate.evidence == {}


def test_existing_detector_contract_is_unchanged() -> None:
    estimate = estimate_modulation_family(_digital("QPSK", 20.0))

    assert isinstance(estimate.family, str)
    assert 0.0 <= estimate.confidence <= 1.0
    assert isinstance(estimate.evidence, dict)


# --- documented limitation: baseband AM-DSB ---------------------------------------------------


def test_baseband_am_dsb_is_documented_as_undetectable_after_preprocessing() -> None:
    """`preprocess` removes DC, which for carrier-offset-0 AM-DSB IS the carrier.

    The residual is bipolar, so its phase flips by pi at every message zero crossing and
    its frequency_cv becomes MORE impulsive than any digital control. No family-level
    rule can recover analog identity from that, and this entry does not pretend to.
    Pinned so the limitation is visible rather than forgotten.
    """
    raw_spec = AnalogSampleSpec(scheme="am_dsb", sample_rate_hz=FS, num_samples=N,
                                carrier_offset_hz=0.0, seed=101)
    raw = modulate_analog(generate_message(raw_spec)[0], raw_spec)
    processed = preprocess(UnifiedSignalContainer(raw, FS)).iq

    raw_cv = float(np.std(np.diff(np.unwrap(np.angle(raw)))) /
                   (np.mean(np.abs(np.diff(np.unwrap(np.angle(raw))))) + 1e-12))
    estimate = estimate_modulation_family(processed)

    assert raw_cv < 0.1, "the carrier-bearing waveform has a near-static phase"
    assert estimate.evidence["frequency_cv"] > 1.0, "DC removal made it look impulsive"
    assert estimate.family != "analog-like"
