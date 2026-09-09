"""Conservative analog subtype routing (BANK.md Entry 032).

Entry 031 measured that the CNN cannot be trusted for the analog subtype - for AM-SSB
LSB it never emits `AM-SSB` at all, in any top-3, in any capture. So once the classical
detector has positively identified `analog-like` (0/480 digital false positives, Entry
031), the subtype is decided deterministically from envelope statistics and the CNN is
not allowed to override it.

The outer safety gate is the classical `analog-like` verdict. Nothing here changes what
happens when the classical family is digital.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.decoding.demodulators.dispatch import demodulate_capture
from radiofry.dsp.cyclostationary import estimate_modulation_family
from radiofry.dsp.parameter_estimation import ParameterEstimate, estimate_parameters
from radiofry.dsp.preprocessing import preprocess
from radiofry.fusion.confidence_fusion import (
    ANALOG_AMPLITUDE_CV_SSB_MIN,
    ANALOG_ENVELOPE_FLATNESS_MAX,
    fuse_modulation,
    select_analog_subtype,
)
from radiofry.synthetic_gen.v1.analog import (
    AnalogSampleSpec, generate_message, modulate_analog)
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import modulate

FS, N = 200_000.0, 8_192
CARRIER = 20_000.0

ANALOG_CASES = {
    "AM-DSB": ("AM-DSB", dict(scheme="am_dsb", carrier_offset_hz=CARRIER)),
    "AM-SSB-USB": ("AM-SSB", dict(scheme="am_ssb", carrier_offset_hz=CARRIER,
                                  sideband="upper")),
    "AM-SSB-LSB": ("AM-SSB", dict(scheme="am_ssb", carrier_offset_hz=CARRIER,
                                  sideband="lower")),
    "WBFM": ("WBFM", dict(scheme="wbfm", carrier_offset_hz=CARRIER,
                          frequency_deviation_hz=15_000.0)),
}


def _analog(kw, snr_db=20.0, seed=211):
    spec = AnalogSampleSpec(sample_rate_hz=FS, num_samples=N, snr_db=snr_db, seed=seed, **kw)
    message, _ = generate_message(spec)
    iq, _ = add_awgn(modulate_analog(message, spec), snr_db,
                     np.random.default_rng([seed, 2]))
    return preprocess(UnifiedSignalContainer(iq, FS))


def _digital(modulation, snr_db=20.0, seed=211, sps=8):
    spec = SampleSpec(modulation=modulation, num_symbols=N // sps, samples_per_symbol=sps,
                      sample_rate_hz=FS, snr_db=snr_db, seed=seed)
    bits = np.random.default_rng([seed, 1]).integers(0, 2, spec.num_bits, dtype=np.uint8)
    iq, _ = add_awgn(modulate(bits, spec), snr_db, np.random.default_rng([seed, 2]))
    return preprocess(UnifiedSignalContainer(iq, FS))


def _route(signal):
    """The production path, minus the CNN, which the subtype gate must dominate."""
    classical = estimate_modulation_family(signal.iq)
    fusion = fuse_modulation("QPSK", 0.95, classical.family,
                             classical_evidence=classical.evidence)
    return classical, fusion


# --- the new evidence field ------------------------------------------------------------


def test_the_detector_now_publishes_envelope_flatness() -> None:
    estimate = estimate_modulation_family(_analog(ANALOG_CASES["WBFM"][1]).iq)

    assert "envelope_flatness" in estimate.evidence
    assert 0.0 <= estimate.evidence["envelope_flatness"] <= 1.0


def test_the_detector_family_decision_is_unchanged_by_the_new_field() -> None:
    # envelope_flatness is recorded as evidence only; it must not alter the family logic.
    for modulation, expected in [("BPSK", "FSK-like"), ("16QAM", "QAM-like"),
                                 ("BFSK", "FSK-like"), ("PAM4", "QAM-like")]:
        assert estimate_modulation_family(_digital(modulation).iq).family == expected


# --- the subtype rule itself -------------------------------------------------------------


def test_thresholds_are_published() -> None:
    assert ANALOG_ENVELOPE_FLATNESS_MAX == 0.40
    assert ANALOG_AMPLITUDE_CV_SSB_MIN == 0.37


@pytest.mark.parametrize("case", list(ANALOG_CASES))
@pytest.mark.parametrize("snr_db", [20.0, 15.0, 10.0])
def test_analog_subtype_is_selected_correctly(case: str, snr_db: float) -> None:
    expected, kw = ANALOG_CASES[case]
    signal = _analog(kw, snr_db=snr_db)
    classical = estimate_modulation_family(signal.iq)
    if classical.family != "analog-like":
        pytest.skip("classical detector did not admit this capture; not a routing case")

    assert select_analog_subtype(classical.evidence) == expected


def test_constant_envelope_selects_wbfm() -> None:
    assert select_analog_subtype(
        {"envelope_flatness": 0.55, "amplitude_cv": 0.07}) == "WBFM"


def test_high_amplitude_variation_selects_ssb() -> None:
    assert select_analog_subtype(
        {"envelope_flatness": 0.05, "amplitude_cv": 0.46}) == "AM-SSB"


def test_moderate_amplitude_variation_selects_dsb() -> None:
    assert select_analog_subtype(
        {"envelope_flatness": 0.06, "amplitude_cv": 0.22}) == "AM-DSB"


def test_envelope_flatness_boundary_at_zero_point_four() -> None:
    just_below = {"envelope_flatness": 0.399, "amplitude_cv": 0.46}
    just_above = {"envelope_flatness": 0.401, "amplitude_cv": 0.46}

    assert select_analog_subtype(just_below) == "AM-SSB"
    assert select_analog_subtype(just_above) == "WBFM"


def test_amplitude_cv_boundary_at_zero_point_three_seven() -> None:
    just_below = {"envelope_flatness": 0.05, "amplitude_cv": 0.369}
    exactly_at = {"envelope_flatness": 0.05, "amplitude_cv": 0.37}

    assert select_analog_subtype(just_below) == "AM-DSB"
    assert select_analog_subtype(exactly_at) == "AM-SSB"


def test_envelope_flatness_is_checked_before_amplitude_cv() -> None:
    # A low-SNR WBFM capture can reach amplitude_cv 0.35; flatness must win.
    assert select_analog_subtype(
        {"envelope_flatness": 0.56, "amplitude_cv": 0.45}) == "WBFM"


def test_missing_evidence_yields_no_subtype() -> None:
    assert select_analog_subtype({}) is None
    assert select_analog_subtype({"amplitude_cv": 0.4}) is None


# --- the CNN must not override ------------------------------------------------------------


@pytest.mark.parametrize("cnn_label,cnn_conf", [
    ("QPSK", 0.99), ("PAM4", 0.95), ("BPSK", 0.85), ("QAM64", 0.99),
])
def test_a_confident_wrong_digital_cnn_label_cannot_override_analog(cnn_label, cnn_conf) -> None:
    signal = _analog(ANALOG_CASES["AM-SSB-LSB"][1])
    classical = estimate_modulation_family(signal.iq)

    fusion = fuse_modulation(cnn_label, cnn_conf, classical.family,
                             classical_evidence=classical.evidence)

    assert fusion.label == "AM-SSB"
    assert fusion.analog_route == "classical_subtype"


@pytest.mark.parametrize("cnn_label", ["WBFM", "AM-DSB"])
def test_a_confident_wrong_analog_cnn_label_cannot_override_analog(cnn_label: str) -> None:
    # Entry 031: AM-SSB USB is called WBFM at 0.44-0.49 and routed to demodulate_fm.
    signal = _analog(ANALOG_CASES["AM-SSB-USB"][1])
    classical = estimate_modulation_family(signal.iq)

    fusion = fuse_modulation(cnn_label, 0.99, classical.family,
                             classical_evidence=classical.evidence)

    assert fusion.label == "AM-SSB"
    assert fusion.analog_route == "classical_subtype"


def test_the_true_label_need_not_appear_in_the_cnn_top_k() -> None:
    # The decisive Entry 031 case: the CNN never emits AM-SSB for LSB.
    signal = _analog(ANALOG_CASES["AM-SSB-LSB"][1])
    classical = estimate_modulation_family(signal.iq)

    fusion = fuse_modulation("PAM4", 0.588, classical.family,
                             ranked_alternatives=(("QAM64", 0.2), ("QPSK", 0.1)),
                             classical_evidence=classical.evidence)

    assert fusion.label == "AM-SSB"


def test_subtype_confidence_is_the_classical_family_confidence() -> None:
    """Documented meaning: this is the classical detector's analog-FAMILY confidence.

    It is NOT a calibrated probability for the subtype. The subtype itself is a
    deterministic threshold decision with no probability attached, and inventing one
    would be exactly the fake calibration this project has avoided.
    """
    signal = _analog(ANALOG_CASES["WBFM"][1])
    classical = estimate_modulation_family(signal.iq)

    fusion = fuse_modulation("QPSK", 0.99, classical.family,
                             classical_confidence=classical.confidence,
                             classical_evidence=classical.evidence)

    assert fusion.label == "WBFM"
    assert fusion.trust_score == pytest.approx(classical.confidence)
    assert fusion.review_recommended is True


# --- digital must be untouched ----------------------------------------------------------------


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
                                        "BFSK", "GFSK", "PAM4"])
@pytest.mark.parametrize("snr_db", [20.0, 10.0])
def test_digital_controls_are_never_routed_to_an_analog_demodulator(modulation, snr_db) -> None:
    classical, fusion = _route(_digital(modulation, snr_db=snr_db))

    assert classical.family != "analog-like"
    assert fusion.label not in {"AM-DSB", "AM-SSB", "WBFM"}
    assert fusion.analog_route == ""


def test_digital_fusion_behaviour_is_byte_identical_without_evidence() -> None:
    with_none = fuse_modulation("QPSK", 0.9, "PSK-like")
    with_evidence = fuse_modulation("QPSK", 0.9, "PSK-like",
                                    classical_evidence={"envelope_flatness": 0.5,
                                                        "amplitude_cv": 0.5})

    assert with_none.label == with_evidence.label == "QPSK"
    assert with_none.trust_score == with_evidence.trust_score
    assert with_evidence.analog_route == ""


def test_omitting_evidence_preserves_the_entry_026_fallback_exactly() -> None:
    # Backward compatibility: callers that pass no evidence get the old behaviour.
    result = fuse_modulation("BPSK", 0.267, "analog-like",
                             ranked_alternatives=(("AM-SSB", 0.152),))

    assert result.label == "AM-SSB"
    assert result.analog_route == "cnn_alternative"
    assert result.analog_fallback is True


# --- end to end: the label must reach the right demodulator ---------------------------------------


@pytest.mark.parametrize("case", list(ANALOG_CASES))
def test_the_routed_label_reaches_its_analog_demodulator(case: str) -> None:
    expected, kw = ANALOG_CASES[case]
    signal = _analog(kw)
    classical, fusion = _route(signal)
    parameters = estimate_parameters(signal)

    dispatched = demodulate_capture(signal, fusion.label, parameters)

    assert fusion.label == expected
    assert dispatched.available is True
    assert dispatched.result.modulation == expected


def test_analog_routing_stays_full_rate() -> None:
    signal = _analog(ANALOG_CASES["AM-DSB"][1])
    _, fusion = _route(signal)

    dispatched = demodulate_capture(
        signal, fusion.label, ParameterEstimate(None, None, 2_000.0, carrier_frequency_hz=CARRIER))

    assert "full-rate" in dispatched.message
    assert dispatched.result.symbols.size >= N - 1


def test_a_null_symbol_rate_still_works_for_routed_analog() -> None:
    signal = _analog(ANALOG_CASES["WBFM"][1])
    _, fusion = _route(signal)

    dispatched = demodulate_capture(signal, fusion.label,
                                    ParameterEstimate(None, None, None))

    assert dispatched.available is True


def test_analog_ber_remains_unavailable() -> None:
    from radiofry.evaluation.harness import _score_report

    record: dict = {"truth_symbol_rate_hz": None, "truth_carrier_hz": CARRIER,
                    "truth_snr_db": 20.0, "expected_family": "analog-like",
                    "expected_cnn_label": "AM-SSB", "truth_interleaver": "none",
                    "truth_fec": "none"}
    report = {"stages": {"demodulation": {"available": True,
                                          "result": {"bits": [1, 0, 1], "modulation": "AM-SSB"}}}}

    _score_report(record, report, np.array([], dtype=np.uint8), has_bits=False)

    assert record["ber_status"] == "unavailable"
    assert record["ber_reason"] == "analog_no_transmitted_bits"


def test_pipeline_passes_classical_evidence_into_fusion() -> None:
    import inspect
    from radiofry import pipeline

    source = inspect.getsource(pipeline.analyze_capture)

    assert "classical_evidence" in source


def test_the_gate_never_fires_when_the_classical_family_is_digital() -> None:
    """The outer safety gate is `analog-like`, not the CNN.

    Entry 032 recorded here that GFSK at samples-per-symbol 16 drew a confident `WBFM`
    from the CNN (0.52-0.90) while the classical detector correctly said `FSK-like`, and
    that fusion accepted it - pre-existing behaviour this entry was required not to
    change. **Entry 034 fixed it**: a positive classical digital verdict now blocks a CNN
    analog label. This test keeps its original purpose - the Entry 032 subtype gate must
    not fire on a digital family - and additionally pins the Entry 034 outcome.
    """
    with_gate = fuse_modulation(
        "WBFM", 0.853, "FSK-like",
        classical_evidence={"envelope_flatness": 0.57, "amplitude_cv": 0.07},
        classical_confidence=0.86,
    )

    assert with_gate.analog_route == "", "the subtype gate must not have fired"
    assert with_gate.label != "WBFM", "Entry 034: a digital family blocks an analog label"
    assert with_gate.digital_family_block is True


def test_the_gate_contributes_no_digital_false_positives_of_its_own() -> None:
    # Every non-analog classical family must leave the CNN decision untouched, even
    # when the envelope evidence would otherwise select an analog subtype.
    evidence = {"envelope_flatness": 0.57, "amplitude_cv": 0.07}
    for family in ("PSK-like", "FSK-like", "QAM-like", "unknown"):
        result = fuse_modulation("QPSK", 0.9, family, classical_evidence=evidence)
        assert result.label == "QPSK", family
        assert result.analog_route == "", family
