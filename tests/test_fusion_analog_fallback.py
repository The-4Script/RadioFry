"""Analog-aware fusion fallback (BANK.md Entry 026).

Entry 025 found that `fuse_modulation` uses `classical_family` only to scale the trust
score: the returned label is always the CNN's label or "Unclassified", so a classical
`analog-like` verdict could never produce an analog label. These tests pin the smallest
fix - recovering an analog label from the CNN's own ranked alternatives - and, just as
importantly, pin the cases that must NOT change.

The global 0.4 threshold is not altered by any test here.
"""

import numpy as np
import pytest

from radiofry.fusion.confidence_fusion import ANALOG_LABELS, fuse_modulation


# --- Case A: an accepted digital CNN decision is untouched ---------------------------


def test_accepted_digital_prediction_is_unchanged_by_the_fallback() -> None:
    result = fuse_modulation("QPSK", 0.9, "PSK-like")

    assert result.label == "QPSK"
    assert result.trust_score == pytest.approx(0.99)
    assert result.review_recommended is False
    assert result.analog_fallback is False


def test_accepted_digital_prediction_survives_analog_classical_evidence() -> None:
    # Above threshold, so the CNN wins even though the classical detector disagrees.
    # This is deliberately NOT changed: analog evidence must not override a confident CNN.
    result = fuse_modulation(
        "QPSK", 0.85, "analog-like",
        ranked_alternatives=(("AM-DSB", 0.10), ("WBFM", 0.05)),
    )

    assert result.label == "QPSK"
    assert result.analog_fallback is False
    assert result.review_recommended is True  # families disagree, as before


# --- Case B: the fallback itself ------------------------------------------------------


def test_low_confidence_digital_with_analog_evidence_recovers_an_analog_label() -> None:
    result = fuse_modulation(
        "BPSK", 0.267, "analog-like",
        ranked_alternatives=(("AM-SSB", 0.152), ("8PSK", 0.129)),
    )

    assert result.label == "AM-SSB"
    assert result.analog_fallback is True
    assert result.classical_family == "analog-like"


def test_the_fallback_does_not_invent_a_confidence() -> None:
    # Trust must be derived from the alternative's own softmax value, not fabricated.
    result = fuse_modulation(
        "BPSK", 0.267, "analog-like",
        ranked_alternatives=(("AM-SSB", 0.152),),
    )

    assert result.trust_score == pytest.approx(min(1.0, 0.152 * 1.1))
    assert result.trust_score < 0.267, "trust must not exceed the rejected primary"


def test_the_fallback_still_recommends_review() -> None:
    result = fuse_modulation(
        "BPSK", 0.267, "analog-like",
        ranked_alternatives=(("WBFM", 0.20),),
    )

    assert result.review_recommended is True


# --- Case C: no analog alternative -> existing rejection ------------------------------


def test_low_confidence_digital_with_no_analog_alternative_is_still_rejected() -> None:
    result = fuse_modulation(
        "BPSK", 0.267, "analog-like",
        ranked_alternatives=(("8PSK", 0.129), ("QPSK", 0.100)),
    )

    assert result.label == "Unclassified"
    assert result.analog_fallback is False
    assert result.review_recommended is True


def test_low_confidence_with_no_alternatives_at_all_is_still_rejected() -> None:
    result = fuse_modulation("BPSK", 0.267, "analog-like")

    assert result.label == "Unclassified"
    assert result.analog_fallback is False


# --- Case D: digital classical family -> nothing changes -------------------------------


def test_low_confidence_digital_with_digital_evidence_is_untouched() -> None:
    # The fallback must be gated on analog-like evidence, not merely on low confidence.
    result = fuse_modulation(
        "BPSK", 0.30, "PSK-like",
        ranked_alternatives=(("AM-DSB", 0.25), ("WBFM", 0.20)),
    )

    assert result.label == "Unclassified"
    assert result.analog_fallback is False


def test_the_documented_rejection_case_from_the_existing_suite_is_preserved() -> None:
    # test_dsp_fusion.py::test_fusion_rejects_low_confidence_and_penalizes_disagreement
    result = fuse_modulation("QPSK", 0.3, "FSK-like")

    assert result.label == "Unclassified"
    assert result.review_recommended
    assert result.trust_score == pytest.approx(0.15)


# --- Case E: an accepted analog CNN prediction ------------------------------------------


def test_accepted_analog_prediction_is_preserved_without_the_fallback() -> None:
    result = fuse_modulation("WBFM", 0.476, "analog-like")

    assert result.label == "WBFM"
    assert result.analog_fallback is False, "already accepted; the fallback must not fire"
    assert result.trust_score == pytest.approx(min(1.0, 0.476 * 1.1))
    assert result.review_recommended is False


def test_a_low_confidence_analog_primary_is_not_rescued_by_itself() -> None:
    # The primary is analog but below threshold; the fallback only rescues a DIGITAL
    # primary, so this stays rejected unless another analog label ranks below it.
    result = fuse_modulation("WBFM", 0.30, "analog-like")

    assert result.label == "Unclassified"
    assert result.analog_fallback is False


# --- Case F: several analog alternatives -> highest ranked wins --------------------------


def test_the_highest_ranked_analog_alternative_wins() -> None:
    result = fuse_modulation(
        "PAM4", 0.394, "analog-like",
        ranked_alternatives=(("QAM64", 0.35), ("AM-DSB", 0.331), ("WBFM", 0.180)),
    )

    assert result.label == "AM-DSB"
    assert result.analog_fallback is True


def test_ranking_order_is_respected_not_alphabetical_order() -> None:
    result = fuse_modulation(
        "PAM4", 0.30, "analog-like",
        ranked_alternatives=(("WBFM", 0.28), ("AM-DSB", 0.20), ("AM-SSB", 0.10)),
    )

    assert result.label == "WBFM"


def test_the_analog_label_set_matches_the_dispatch_routes() -> None:
    # Entry 028 moved the analog routes into dispatch's own ANALOG_LABELS set and a
    # dedicated helper, so the two sets are now compared directly instead of by
    # scraping the source of demodulate_capture.
    from radiofry.decoding.demodulators.dispatch import ANALOG_LABELS as DISPATCH_LABELS

    assert ANALOG_LABELS == frozenset({"AM-DSB", "AM-SSB", "WBFM"})
    assert ANALOG_LABELS == DISPATCH_LABELS, "fusion may emit a label dispatch cannot route"


# --- the fallback must reach dispatch ------------------------------------------------------


def test_a_recovered_analog_label_actually_reaches_its_analog_demodulator() -> None:
    from radiofry.contracts import UnifiedSignalContainer
    from radiofry.decoding.demodulators.dispatch import demodulate_capture
    from radiofry.dsp.parameter_estimation import ParameterEstimate
    from radiofry.synthetic_gen.v1.analog import (
        AnalogSampleSpec, generate_message, modulate_analog)

    spec = AnalogSampleSpec(scheme="am_dsb", sample_rate_hz=200_000.0, num_samples=8_192,
                            carrier_offset_hz=0.0, seed=7)
    signal = UnifiedSignalContainer(modulate_analog(generate_message(spec)[0], spec), 200_000.0)
    fusion = fuse_modulation("PAM4", 0.394, "analog-like",
                             ranked_alternatives=(("AM-DSB", 0.331),))

    dispatched = demodulate_capture(
        signal, fusion.label, ParameterEstimate(None, None, 3_857.4, carrier_frequency_hz=0.0))

    assert fusion.label == "AM-DSB"
    assert dispatched.available is True
    assert dispatched.result.modulation == "AM-DSB"


# --- backward compatibility of the API -------------------------------------------------------


def test_the_legacy_alternatives_argument_still_works() -> None:
    result = fuse_modulation("QPSK", 0.9, "PSK-like", alternatives=("8PSK", "BPSK"))

    assert result.alternatives == ("8PSK", "BPSK")
    assert result.label == "QPSK"


def test_positional_call_signature_is_unchanged() -> None:
    result = fuse_modulation("QAM16", 0.9, "QAM-like")

    assert result.label == "QAM16"


def test_threshold_default_is_still_zero_point_four() -> None:
    import inspect

    signature = inspect.signature(fuse_modulation)

    assert signature.parameters["threshold"].default == 0.4


# --- the pipeline must actually supply ranked alternatives -------------------------------------


def test_pipeline_passes_ranked_alternatives_into_fusion() -> None:
    import inspect
    from radiofry import pipeline

    source = inspect.getsource(pipeline.analyze_capture)

    assert "ranked_alternatives" in source, "pipeline still drops CNN confidences"
