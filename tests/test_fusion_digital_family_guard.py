"""Digital-family safety guard in fusion (BANK.md Entry 034).

Entry 033 measured a routing defect: heavily oversampled GFSK draws a confident `WBFM`
from the CNN (up to 0.999) while the classical detector correctly reports `FSK-like`, and
fusion accepted the CNN - sending a digital capture into an analog demodulator.

This is the mirror of the Entry 032 gate. There, a classical `analog-like` verdict beats
the CNN. Here, a classical *digital* verdict blocks a CNN *analog* label. Both directions
must coexist.

The rule does NOT say "classical always wins" - it is narrowly about analog labels being
blocked by a positive digital family verdict.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.cyclostationary import estimate_modulation_family
from radiofry.dsp.preprocessing import preprocess
from radiofry.fusion.confidence_fusion import (
    ANALOG_LABELS,
    DIGITAL_FAMILIES,
    DIGITAL_FAMILY_MIN_CONFIDENCE,
    fuse_modulation,
)
from radiofry.models.modulation_inference import predict_modulation
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import modulate

FS, N = 200_000.0, 8_192
CHECKPOINT = "models_saved/modulation_cnn.pt"


def _digital_signal(modulation: str, sps: int, snr_db: float = 20.0, seed: int = 401):
    spec = SampleSpec(modulation=modulation, num_symbols=N // sps, samples_per_symbol=sps,
                      sample_rate_hz=FS, snr_db=snr_db, seed=seed)
    bits = np.random.default_rng([seed, 1]).integers(0, 2, spec.num_bits, dtype=np.uint8)
    iq, _ = add_awgn(modulate(bits, spec), snr_db, np.random.default_rng([seed, 2]))
    return preprocess(UnifiedSignalContainer(iq, FS))


# --- the constants -----------------------------------------------------------------------


def test_the_digital_family_set_is_explicit_and_excludes_unknown() -> None:
    assert DIGITAL_FAMILIES == frozenset({"PSK-like", "FSK-like", "QAM-like"})
    assert "analog-like" not in DIGITAL_FAMILIES
    assert "unknown" not in DIGITAL_FAMILIES


def test_the_confidence_floor_sits_below_every_measured_digital_verdict() -> None:
    # Measured over 144 digital captures (8 classes x sps 8/16/32 x SNR 20/10/0):
    # FSK-like 0.828-1.000, QAM-like 0.557-0.709. 0.5 clears all of them and excludes
    # the detector's "unknown" non-verdict, which scores 0.2.
    assert DIGITAL_FAMILY_MIN_CONFIDENCE == 0.5


# --- required blocking matrix ---------------------------------------------------------------


@pytest.mark.parametrize("family", ["FSK-like", "PSK-like", "QAM-like"])
@pytest.mark.parametrize("analog_label", ["WBFM", "AM-DSB", "AM-SSB"])
def test_a_digital_family_blocks_every_analog_label(family, analog_label) -> None:
    result = fuse_modulation(analog_label, 0.99, family, classical_confidence=0.86)

    assert result.label != analog_label
    assert result.label not in ANALOG_LABELS
    assert result.digital_family_block is True
    assert result.analog_route == ""


def test_blocking_holds_even_at_maximum_cnn_confidence() -> None:
    result = fuse_modulation("WBFM", 1.0, "FSK-like", classical_confidence=0.86)

    assert result.label not in ANALOG_LABELS


def test_a_weak_classical_digital_verdict_does_not_block() -> None:
    # Below the floor the classical verdict is not trusted enough to override the CNN.
    result = fuse_modulation("WBFM", 0.99, "FSK-like", classical_confidence=0.30)

    assert result.label == "WBFM"
    assert result.digital_family_block is False


def test_unknown_family_does_not_block() -> None:
    # "unknown" is the detector's non-verdict, not a positive digital verdict.
    result = fuse_modulation("WBFM", 0.99, "unknown", classical_confidence=0.2)

    assert result.label == "WBFM"
    assert result.digital_family_block is False


# --- fallback behaviour ------------------------------------------------------------------------


def test_a_digital_alternative_above_threshold_is_retained() -> None:
    # The CNN's own ranking supplies the replacement; no label is invented.
    result = fuse_modulation(
        "WBFM", 0.538, "FSK-like", classical_confidence=0.86,
        ranked_alternatives=(("GFSK", 0.462), ("AM-DSB", 0.000)))

    assert result.label == "GFSK"
    assert result.digital_family_block is True
    assert result.trust_score == pytest.approx(min(1.0, 0.462 * 1.1))


def test_a_digital_alternative_below_threshold_is_rejected_not_forced() -> None:
    # Wrong analog demodulation must be prevented; a rejection is the safe outcome, and
    # a 0.001 alternative must not be promoted just because it is digital.
    result = fuse_modulation(
        "WBFM", 0.999, "FSK-like", classical_confidence=0.86,
        ranked_alternatives=(("GFSK", 0.001), ("AM-DSB", 0.000)))

    assert result.label == "Unclassified"
    assert result.digital_family_block is True


def test_no_digital_alternative_at_all_yields_rejection() -> None:
    result = fuse_modulation(
        "WBFM", 0.99, "FSK-like", classical_confidence=0.86,
        ranked_alternatives=(("AM-DSB", 0.30), ("AM-SSB", 0.10)))

    assert result.label == "Unclassified"


def test_the_family_is_not_turned_into_a_subtype() -> None:
    # "FSK-like" must never become a concrete label such as CPFSK by fiat.
    result = fuse_modulation("WBFM", 0.99, "FSK-like", classical_confidence=0.86)

    assert result.label == "Unclassified"
    assert result.label not in {"CPFSK", "GFSK", "BPSK", "QPSK", "8PSK",
                                "QAM16", "QAM64", "PAM4"}


def test_the_retained_alternative_must_itself_be_digital() -> None:
    result = fuse_modulation(
        "WBFM", 0.99, "QAM-like", classical_confidence=0.60,
        ranked_alternatives=(("AM-DSB", 0.90), ("QAM64", 0.55)))

    assert result.label == "QAM64", "the analog alternative must be skipped"


# --- existing digital behaviour must be untouched -------------------------------------------------


def test_a_digital_cnn_label_with_a_digital_family_is_unchanged() -> None:
    accepted = fuse_modulation("QPSK", 0.9, "PSK-like", classical_confidence=0.8)
    rejected = fuse_modulation("QPSK", 0.3, "PSK-like", classical_confidence=0.8)

    assert accepted.label == "QPSK"
    assert accepted.digital_family_block is False
    assert rejected.label == "Unclassified"
    assert rejected.digital_family_block is False


def test_cross_family_digital_disagreement_still_behaves_as_before() -> None:
    # QAM16 under an FSK-like verdict is a digital/digital conflict; the guard is only
    # about analog labels and must not touch this.
    result = fuse_modulation("QAM16", 0.9, "FSK-like", classical_confidence=0.86)

    assert result.label == "QAM16"
    assert result.digital_family_block is False
    assert result.review_recommended is True


def test_the_existing_rejection_case_is_preserved() -> None:
    result = fuse_modulation("QPSK", 0.3, "FSK-like")

    assert result.label == "Unclassified"
    assert result.trust_score == pytest.approx(0.15)


# --- Entry 032 must keep working (the opposite direction) --------------------------------------------


def test_entry_032_analog_gate_still_fires_for_a_digital_cnn_label() -> None:
    evidence = {"envelope_flatness": 0.02, "amplitude_cv": 0.46}

    result = fuse_modulation("PAM4", 0.99, "analog-like",
                             classical_evidence=evidence, classical_confidence=0.75)

    assert result.label == "AM-SSB"
    assert result.analog_route == "classical_subtype"
    assert result.digital_family_block is False


def test_entry_032_analog_gate_still_overrides_a_wrong_analog_cnn_label() -> None:
    evidence = {"envelope_flatness": 0.57, "amplitude_cv": 0.07}

    result = fuse_modulation("AM-SSB", 0.99, "analog-like",
                             classical_evidence=evidence, classical_confidence=0.75)

    assert result.label == "WBFM"
    assert result.analog_route == "classical_subtype"


def test_entry_026_fallback_still_works() -> None:
    result = fuse_modulation("BPSK", 0.267, "analog-like",
                             ranked_alternatives=(("AM-SSB", 0.152),))

    assert result.label == "AM-SSB"
    assert result.analog_route == "cnn_alternative"


# --- the Entry 033 regression, through the real pipeline ------------------------------------------------


@pytest.mark.parametrize("sps", [16, 32])
def test_gfsk_high_oversampling_never_routes_to_an_analog_demodulator(sps: int) -> None:
    """The exact Entry 033 failure: GFSK at sps 16/32 accepted as WBFM."""
    blocked = 0
    for snr_db in (20.0, 15.0):
        for seed in (401, 409, 419, 421, 431):
            signal = _digital_signal("GFSK", sps, snr_db, seed)
            classical = estimate_modulation_family(signal.iq)
            prediction = predict_modulation(signal, CHECKPOINT)
            if not prediction.available:
                pytest.skip(f"CNN unavailable in this environment: {prediction.message}")
            ranked = tuple((l, c) for l, c in prediction.top_k[1:])

            fusion = fuse_modulation(
                prediction.label, prediction.confidence, classical.family,
                ranked_alternatives=ranked,
                classical_evidence=classical.evidence,
                classical_confidence=classical.confidence)

            assert fusion.label not in ANALOG_LABELS, (
                f"sps={sps} snr={snr_db} seed={seed} classical={classical.family} "
                f"cnn={prediction.label}@{prediction.confidence:.3f} -> {fusion.label}")
            assert fusion.analog_route == ""
            if prediction.label in ANALOG_LABELS:
                blocked += 1
                assert fusion.digital_family_block is True
    assert blocked > 0, "fixture no longer reproduces the Entry 033 condition"


def test_the_blocked_decision_is_either_a_digital_label_or_a_rejection() -> None:
    # Inspect the complete decision, not just "not WBFM".
    signal = _digital_signal("GFSK", 16, 20.0, 401)
    classical = estimate_modulation_family(signal.iq)
    prediction = predict_modulation(signal, CHECKPOINT)
    if not prediction.available:
        pytest.skip(f"CNN unavailable in this environment: {prediction.message}")
    ranked = tuple((l, c) for l, c in prediction.top_k[1:])

    fusion = fuse_modulation(prediction.label, prediction.confidence, classical.family,
                             ranked_alternatives=ranked,
                             classical_evidence=classical.evidence,
                             classical_confidence=classical.confidence)

    assert classical.family == "FSK-like"
    assert prediction.label == "WBFM"
    assert fusion.label in {"GFSK", "CPFSK", "Unclassified"}
    assert fusion.review_recommended is True
