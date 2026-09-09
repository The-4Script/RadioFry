"""Trust scoring and open-set rejection for modulation decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionResult:
    label: str
    trust_score: float
    review_recommended: bool
    classical_family: str
    alternatives: tuple[str, ...] = ()
    # True when `label` came from the analog fallback rather than the CNN's own
    # top-1. Defaults to False so existing readers of this dataclass are unaffected.
    analog_fallback: bool = False


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
            )

    return FusionResult(
        label="Unclassified" if rejected else ml_label,
        trust_score=trust,
        review_recommended=rejected or not agrees,
        classical_family=classical_family,
        alternatives=alternatives,
    )
