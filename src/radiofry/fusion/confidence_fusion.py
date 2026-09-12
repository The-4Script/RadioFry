"""Trust scoring and open-set rejection for modulation decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionResult:
    label: str
    trust_score: float
    review_recommended: bool
    classical_family: str
    alternatives: tuple[str, ...] = ()
    # True when `label` came from analog logic rather than the CNN's own top-1.
    # Defaults to False so existing readers of this dataclass are unaffected.
    analog_fallback: bool = False
    # Which analog path produced the label: "" (none), "cnn_alternative" (Entry 026,
    # recovered from the CNN's ranked alternatives) or "classical_subtype" (Entry 032,
    # decided from classical envelope evidence, ignoring the CNN).
    analog_route: str = ""
    # True when a CNN analog label was refused because the classical detector returned a
    # positive DIGITAL family verdict (Entry 034). Defaulted, so existing readers are
    # unaffected.
    digital_family_block: bool = False
    analog_subtype_margin: float | None = None


_FAMILY_BY_LABEL = {
    "BPSK": "PSK-like", "QPSK": "PSK-like", "8PSK": "PSK-like",
    "CPFSK": "FSK-like", "GFSK": "FSK-like",
    "QAM16": "QAM-like", "QAM64": "QAM-like", "PAM4": "QAM-like",
    "AM-DSB": "analog-like", "AM-SSB": "analog-like", "WBFM": "analog-like",
}

ANALOG_FAMILY = "analog-like"

# The analog labels the CNN can emit AND `dispatch.demodulate_capture` can route.
# Both sides must agree, so this set is asserted against dispatch in the tests.
ANALOG_LABELS = frozenset({"AM-DSB", "AM-SSB", "WBFM"})

# The classical detector's three POSITIVE digital verdicts. "unknown" is deliberately
# excluded: it is the detector's non-verdict, not evidence of anything (Entry 034).
DIGITAL_FAMILIES = frozenset({"PSK-like", "FSK-like", "QAM-like"})

# Sufficiency floor for trusting a digital family verdict. Measured over 144 digital
# captures (8 classes x samples-per-symbol 8/16/32 x SNR 20/10/0 dB): FSK-like scored
# 0.828-1.000 and QAM-like 0.557-0.709, so every real digital verdict clears 0.5 while
# the detector's "unknown" non-verdict (0.2) does not.
DIGITAL_FAMILY_MIN_CONFIDENCE = 0.5


def _highest_ranked_digital(
    ranked_alternatives: tuple[tuple[str, float], ...],
) -> tuple[str, float] | None:
    """Highest-ranked NON-analog label the CNN itself offered, if any."""

    for label, confidence in ranked_alternatives:
        if label in _FAMILY_BY_LABEL and label not in ANALOG_LABELS:
            return label, float(confidence)
    return None


# Analog subtype thresholds (BANK.md Entries 031, 032), measured not guessed.
#
# `envelope_flatness` separates constant-envelope WBFM (0.54-0.58) from the amplitude
# modulations (<= 0.33); `amplitude_cv` separates AM-DSB (0.22-0.30) from AM-SSB
# (0.45-0.48). Validated 60/60 on unseen seeds at 20/15/10 dB, with 0/480 digital
# captures able to reach the rule at all - the classical `analog-like` verdict is the
# outer gate and never admitted a digital control.
ANALOG_ENVELOPE_FLATNESS_MAX = 0.40
ANALOG_AMPLITUDE_CV_SSB_MIN = 0.37


def select_analog_subtype(evidence: dict[str, float]) -> str | None:
    """Deterministic analog subtype from classical envelope evidence.

    Returns None when the required evidence is missing, so callers cannot accidentally
    route on a partial feature set. This is a threshold decision, not a probability:
    no confidence is attached to the subtype itself.

    Known limitation (Entry 031): the AM-DSB/AM-SSB boundary rides on modulation depth.
    It is correct up to a depth of roughly 0.8; at depth >= 0.9 an AM-DSB capture reaches
    `amplitude_cv` 0.379 and is read as AM-SSB. The project default is 0.5.
    """

    flatness = evidence.get("envelope_flatness")
    amplitude_cv = evidence.get("amplitude_cv")
    if flatness is None or amplitude_cv is None:
        return None
    if flatness > ANALOG_ENVELOPE_FLATNESS_MAX:
        # Constant envelope: the message is not in |s|, so this is FM.
        return "WBFM"
    if amplitude_cv >= ANALOG_AMPLITUDE_CV_SSB_MIN:
        return "AM-SSB"
    return "AM-DSB"


def analog_subtype_margin(evidence: dict[str, float]) -> float | None:
    """Return distance from the nearest analog subtype threshold."""

    flatness = evidence.get("envelope_flatness")
    amplitude_cv = evidence.get("amplitude_cv")
    if flatness is None or amplitude_cv is None:
        return None
    if flatness > ANALOG_ENVELOPE_FLATNESS_MAX:
        return float(flatness - ANALOG_ENVELOPE_FLATNESS_MAX)
    return float(abs(amplitude_cv - ANALOG_AMPLITUDE_CV_SSB_MIN))


def _recover_analog_alternative(
    ranked_alternatives: tuple[tuple[str, float], ...],
) -> tuple[str, float] | None:
    """Highest-ranked analog label among the CNN's own alternatives, if any.

    Ranking is the CNN's, not ours: the first match in the supplied order wins. No
    label is invented and no confidence is fabricated - the alternative's own softmax
    value is carried through so trust stays honest.
    """

    for label, confidence in ranked_alternatives:
        if label in ANALOG_LABELS:
            return label, float(confidence)
    return None


def fuse_modulation(
    ml_label: str,
    ml_confidence: float,
    classical_family: str,
    *,
    threshold: float = 0.4,
    alternatives: tuple[str, ...] = (),
    ranked_alternatives: tuple[tuple[str, float], ...] = (),
    classical_evidence: dict[str, float] | None = None,
    classical_confidence: float | None = None,
) -> FusionResult:
    """Combine CNN confidence with an independent family hypothesis.

    Entry 026 adds one narrow path. When the CNN's top-1 is a *digital* label that the
    existing threshold already rejects, and the independent classical detector says
    `analog-like`, the CNN's ranked alternatives are searched for an analog label. That
    turns a decision which was being discarded into a routable one; it never overrides an
    accepted CNN decision, and `threshold` is unchanged.
    """

    probability = max(0.0, min(1.0, float(ml_confidence)))
    agrees = _FAMILY_BY_LABEL.get(ml_label) == classical_family
    trust = min(1.0, probability * 1.1) if agrees else probability * 0.5
    rejected = probability < threshold

    # Entry 032. Once the classical detector has positively identified analog - the
    # outer safety gate, 0/480 digital false positives - the subtype comes from the
    # signal's own envelope statistics and the CNN does not get a vote. Entry 031
    # measured why: for AM-SSB LSB the CNN never emits `AM-SSB` in any top-3, so there
    # is nothing to re-rank. Only active when the caller supplies evidence, so every
    # existing call site keeps its previous behaviour exactly.
    if classical_family == ANALOG_FAMILY and classical_evidence is not None:
        subtype = select_analog_subtype(classical_evidence)
        if subtype is not None:
            return FusionResult(
                label=subtype,
                # The classical detector's analog-FAMILY confidence. The subtype is a
                # deterministic threshold decision and carries no probability of its
                # own; none is invented here.
                trust_score=(trust if classical_confidence is None
                             else max(0.0, min(1.0, float(classical_confidence)))),
                # A rule, not a trained model: always worth a human look.
                review_recommended=True,
                classical_family=classical_family,
                alternatives=alternatives,
                analog_fallback=True,
                analog_route="classical_subtype",
                analog_subtype_margin=analog_subtype_margin(classical_evidence),
            )

    # Entry 034 - the mirror of the Entry 032 gate. A CNN analog label must not override
    # an independent, positive classical DIGITAL verdict: Entry 033 measured GFSK at high
    # oversampling drawing `WBFM` at up to 0.999 while the detector correctly said
    # `FSK-like`, which sent a digital capture into an analog demodulator. Narrow by
    # construction - it fires only on analog labels, so digital/digital disagreements are
    # untouched.
    if (
        ml_label in ANALOG_LABELS
        and classical_family in DIGITAL_FAMILIES
        and (classical_confidence is None
             or float(classical_confidence) >= DIGITAL_FAMILY_MIN_CONFIDENCE)
    ):
        # Prefer the CNN's own highest-ranked digital alternative, but only if it clears
        # the same acceptance threshold every other prediction must clear. Otherwise
        # reject: a wrong analog demodulation is worse than refusing to answer, and the
        # family is never turned into a subtype by fiat.
        alternative = _highest_ranked_digital(ranked_alternatives)
        if alternative is not None and alternative[1] >= threshold:
            label, confidence = alternative
            return FusionResult(
                label=label,
                trust_score=(min(1.0, confidence * 1.1)
                             if _FAMILY_BY_LABEL.get(label) == classical_family
                             else confidence * 0.5),
                review_recommended=True,
                classical_family=classical_family,
                alternatives=alternatives,
                digital_family_block=True,
            )
        return FusionResult(
            label="Unclassified",
            trust_score=probability * 0.5,
            review_recommended=True,
            classical_family=classical_family,
            alternatives=alternatives,
            digital_family_block=True,
        )

    if rejected and classical_family == ANALOG_FAMILY and ml_label not in ANALOG_LABELS:
        recovered = _recover_analog_alternative(ranked_alternatives)
        if recovered is not None:
            label, confidence = recovered
            return FusionResult(
                label=label,
                # The recovered label agrees with the classical family by construction,
                # so it earns the same agreement bonus - applied to its own confidence.
                trust_score=min(1.0, confidence * 1.1),
                # Still a low-confidence recovery: a human should look at it.
                review_recommended=True,
                classical_family=classical_family,
                alternatives=alternatives,
                analog_fallback=True,
                analog_route="cnn_alternative",
            )

    return FusionResult(
        label="Unclassified" if rejected else ml_label,
        trust_score=trust,
        review_recommended=rejected or not agrees,
        classical_family=classical_family,
        alternatives=alternatives,
    )
