"""Trust scoring and open-set rejection for modulation decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionResult:
    label: str
    trust_score: float
    review_recommended: bool
    classical_family: str
    alternatives: tuple[str, ...] = ()


_FAMILY_BY_LABEL = {
    "BPSK": "PSK-like", "QPSK": "PSK-like", "8PSK": "PSK-like",
    "CPFSK": "FSK-like", "GFSK": "FSK-like",
    "QAM16": "QAM-like", "QAM64": "QAM-like", "PAM4": "QAM-like",
    "AM-DSB": "analog-like", "AM-SSB": "analog-like", "WBFM": "analog-like",
}


def fuse_modulation(
    ml_label: str,
    ml_confidence: float,
    classical_family: str,
    *,
    threshold: float = 0.4,
    alternatives: tuple[str, ...] = (),
) -> FusionResult:
    """Combine CNN confidence with an independent family hypothesis."""

    probability = max(0.0, min(1.0, float(ml_confidence)))
    agrees = _FAMILY_BY_LABEL.get(ml_label) == classical_family
    trust = min(1.0, probability * 1.1) if agrees else probability * 0.5
    rejected = probability < threshold
    return FusionResult(
        label="Unclassified" if rejected else ml_label,
        trust_score=trust,
        review_recommended=rejected or not agrees,
        classical_family=classical_family,
        alternatives=alternatives,
    )
