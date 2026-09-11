"""What the implemented fusion rules do, mapped out by running them.

This module explains `confidence_fusion.fuse_modulation` - it does not model it. Every
cell of every landscape here is produced by *calling the real function*, so the picture
cannot drift from the behaviour it claims to describe. Re-deriving the thresholds in a
second place would create exactly the failure this feature exists to prevent: a
plausible-looking surface that no longer matches the code.

What the axes are, and what they are not
---------------------------------------
`fuse_modulation` takes CNN label and confidence, the classical family, the classical
confidence, the CNN's ranked alternatives, and the classical envelope evidence. It does
**not** take SNR. SNR shapes the inputs upstream - a noisy capture produces a less
confident CNN and different envelope statistics - but it is not an argument to the
decision, and no threshold in the fusion code refers to it. Plotting SNR as an axis of
the decision surface would therefore be inventing a dimension the logic does not have, so
the axes here are the two continuous quantities fusion actually reads:

    x = CNN confidence      (the 0.4 acceptance threshold lives here)
    y = classical confidence (the 0.5 digital-family sufficiency floor lives here)

Everything else - which label the CNN emitted, which family the classical detector
returned, the ranked alternatives, the envelope evidence - is *categorical context* that
selects which regime you are looking at, and is carried in `LandscapeScenario`.

The landscape is a map of implemented decision boundaries. It is not a probability
surface, and nothing here is learned.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from radiofry.fusion.confidence_fusion import (
    ANALOG_FAMILY,
    ANALOG_LABELS,
    DIGITAL_FAMILIES,
    DIGITAL_FAMILY_MIN_CONFIDENCE,
    FusionResult,
    fuse_modulation,
)

DEFAULT_THRESHOLD = 0.4
DEFAULT_GRID = 81

# Every branch `fuse_modulation` can take, named for what happened rather than for the
# code path. Ordered from "trusted" to "refused" so a legend reads top to bottom.
ACCEPTED = "accepted"
ACCEPTED_REVIEW = "accepted_review"
ANALOG_SUBTYPE = "analog_subtype"
ANALOG_RECOVERY = "analog_recovery"
GUARD_REROUTE = "guard_reroute"
GUARD_BLOCK = "guard_block"
REJECTED = "rejected"

OUTCOMES: dict[str, dict[str, str]] = {
    ACCEPTED: {
        "label": "Accepted",
        "detail": "CNN confidence clears the threshold and the classical family agrees.",
    },
    ACCEPTED_REVIEW: {
        "label": "Accepted, review advised",
        "detail": "CNN confidence clears the threshold but the classical family "
                  "disagrees, so trust is halved and review is flagged.",
    },
    ANALOG_SUBTYPE: {
        "label": "Analog subtype (classical)",
        "detail": "The classical detector positively identified analog, so the subtype "
                  "is decided from envelope statistics and the CNN does not vote.",
    },
    ANALOG_RECOVERY: {
        "label": "Analog recovered from alternatives",
        "detail": "A rejected digital top-1 plus a classical analog verdict; an analog "
                  "label was recovered from the CNN's own ranked alternatives.",
    },
    GUARD_REROUTE: {
        "label": "Digital-family guard: rerouted",
        "detail": "A CNN analog label was refused against a positive classical digital "
                  "verdict, and the CNN's best digital alternative was used instead.",
    },
    GUARD_BLOCK: {
        "label": "Digital-family guard: blocked",
        "detail": "A CNN analog label was refused against a positive classical digital "
                  "verdict, and no digital alternative cleared the threshold.",
    },
    REJECTED: {
        "label": "Rejected (Unclassified)",
        "detail": "CNN confidence is below the acceptance threshold and no recovery "
                  "path applied.",
    },
}


@dataclass(frozen=True)
class LandscapeScenario:
    """The categorical context that selects which fusion regime is being mapped.

    These are the arguments to `fuse_modulation` that are not continuous. Holding them
    fixed while sweeping the two confidences is what makes a 2D landscape well defined.
    """

    ml_label: str = "QPSK"
    classical_family: str = "PSK-like"
    ranked_alternatives: tuple[tuple[str, float], ...] = ()
    classical_evidence: dict[str, float] | None = None
    threshold: float = DEFAULT_THRESHOLD

    def describe(self) -> str:
        parts = [f"CNN label {self.ml_label}", f"classical {self.classical_family}"]
        if self.ranked_alternatives:
            parts.append("alternatives " + ", ".join(
                f"{name} {value:.3f}" for name, value in self.ranked_alternatives[:3]))
        if self.classical_evidence:
            parts.append("envelope evidence supplied")
        return " · ".join(parts)


@dataclass(frozen=True)
class LandscapeGrid:
    """Outcomes of the real fusion function over the two confidence axes."""

    scenario: LandscapeScenario
    cnn_axis: np.ndarray
    classical_axis: np.ndarray
    outcomes: np.ndarray          # [classical, cnn] of outcome keys
    labels: np.ndarray            # [classical, cnn] of decided labels
    trust: np.ndarray             # [classical, cnn] float
    review: np.ndarray            # [classical, cnn] bool
    warnings: tuple[str, ...] = field(default=())

    @property
    def present_outcomes(self) -> list[str]:
        """Outcome keys that actually occur here, in the canonical order."""
        seen = set(np.unique(self.outcomes).tolist())
        return [key for key in OUTCOMES if key in seen]

    def outcome_at(self, cnn_confidence: float, classical_confidence: float) -> str:
        """Nearest grid cell, for placing a capture on the map."""
        column = int(np.argmin(np.abs(self.cnn_axis - cnn_confidence)))
        row = int(np.argmin(np.abs(self.classical_axis - classical_confidence)))
        return str(self.outcomes[row, column])


def classify_outcome(result: FusionResult, scenario: LandscapeScenario) -> str:
    """Name the branch a `FusionResult` came from, reading only its own fields.

    Deliberately derived from the result rather than from a copy of the conditions: if a
    guard is added or changed in `confidence_fusion`, this keeps describing what actually
    happened instead of silently mislabelling it.
    """

    if result.digital_family_block:
        return GUARD_BLOCK if result.label == "Unclassified" else GUARD_REROUTE
    if result.analog_route == "classical_subtype":
        return ANALOG_SUBTYPE
    if result.analog_route == "cnn_alternative":
        return ANALOG_RECOVERY
    if result.label == "Unclassified":
        return REJECTED
    return ACCEPTED_REVIEW if result.review_recommended else ACCEPTED


def evaluate_landscape(
    scenario: LandscapeScenario,
    resolution: int = DEFAULT_GRID,
) -> LandscapeGrid:
    """Run the real fusion function across the CNN x classical confidence plane.

    Cost is `resolution^2` calls to a pure-Python function with no array work, so an
    81x81 map is a few thousand calls and finishes in milliseconds. Nothing is
    interpolated: every cell is an evaluated decision.
    """

    size = max(2, int(resolution))
    cnn_axis = np.linspace(0.0, 1.0, size)
    classical_axis = np.linspace(0.0, 1.0, size)

    outcomes = np.empty((size, size), dtype=object)
    labels = np.empty((size, size), dtype=object)
    trust = np.zeros((size, size), dtype=float)
    review = np.zeros((size, size), dtype=bool)

    for row, classical_confidence in enumerate(classical_axis):
        for column, cnn_confidence in enumerate(cnn_axis):
            result = fuse_modulation(
                scenario.ml_label,
                float(cnn_confidence),
                scenario.classical_family,
                threshold=scenario.threshold,
                alternatives=tuple(name for name, _ in scenario.ranked_alternatives),
                ranked_alternatives=scenario.ranked_alternatives,
                classical_evidence=scenario.classical_evidence,
                classical_confidence=float(classical_confidence),
            )
            outcomes[row, column] = classify_outcome(result, scenario)
            labels[row, column] = result.label
            trust[row, column] = result.trust_score
            review[row, column] = result.review_recommended

    warnings: list[str] = []
    if len(set(np.unique(outcomes).tolist())) == 1:
        warnings.append(
            "Every point in this plane takes the same branch, so neither confidence "
            "changes the decision in this context. Change the CNN label or the classical "
            "family to reach a different regime.")
    if np.all(outcomes == outcomes[:1, :]):
        warnings.append(
            "The decision does not vary with classical confidence here: in this context "
            "fusion reads the classical *family*, and uses its confidence only for the "
            "digital-family guard and for analog trust.")
    return LandscapeGrid(scenario, cnn_axis, classical_axis, outcomes, labels, trust,
                         review, tuple(warnings))


@dataclass(frozen=True)
class CapturePosition:
    """Where one analyzed capture sits, and what fusion actually did with it."""

    cnn_label: str | None
    cnn_confidence: float | None
    classical_family: str | None
    classical_confidence: float | None
    snr_db: float | None
    fused_label: str | None
    trust_score: float | None
    review_recommended: bool
    outcome: str | None
    guards: tuple[str, ...] = ()
    reproduced: bool | None = None
    notes: tuple[str, ...] = ()

    @property
    def plottable(self) -> bool:
        return (self.cnn_confidence is not None
                and self.classical_confidence is not None)


def scenario_from_report(report: dict | None,
                         threshold: float = DEFAULT_THRESHOLD
                         ) -> LandscapeScenario | None:
    """Rebuild the exact categorical context a recorded analysis presented to fusion."""

    stages = (report or {}).get("stages") or {}
    cnn = stages.get("cnn_modulation") or {}
    classical = stages.get("classical_modulation") or {}
    if not isinstance(cnn, dict) or not isinstance(classical, dict):
        return None
    label = cnn.get("label")
    family = classical.get("family")
    if not label or not family:
        return None

    # top_k[1:] is what the pipeline passes as `ranked_alternatives`.
    ranked = tuple(
        (str(name), float(value))
        for name, value in (cnn.get("top_k") or [])[1:]
        if isinstance(value, (int, float)))
    evidence = classical.get("evidence")
    return LandscapeScenario(
        ml_label=str(label),
        classical_family=str(family),
        ranked_alternatives=ranked,
        classical_evidence=evidence if isinstance(evidence, dict) else None,
        threshold=threshold,
    )


def describe_capture(report: dict | None,
                     threshold: float = DEFAULT_THRESHOLD) -> CapturePosition:
    """Extract a capture's fusion inputs, and re-run fusion to confirm they reproduce it.

    `reproduced` is the honesty check that matters: if re-running the real function on
    the recorded inputs does not return the recorded label, the point being plotted does
    not belong to the landscape around it, and the UI says so rather than drawing it
    anyway.
    """

    stages = (report or {}).get("stages") or {}
    cnn = stages.get("cnn_modulation") or {}
    classical = stages.get("classical_modulation") or {}
    fusion = stages.get("fusion") or {}
    parameters = stages.get("parameters") or {}
    if not isinstance(fusion, dict):
        fusion = {}

    def number(source: dict, key: str) -> float | None:
        value = source.get(key) if isinstance(source, dict) else None
        return float(value) if isinstance(value, (int, float)) else None

    guards: list[str] = []
    if fusion.get("digital_family_block"):
        guards.append("digital-family guard (Entry 034)")
    route = fusion.get("analog_route") or ""
    if route == "classical_subtype":
        guards.append("analog subtype gate (Entry 032)")
    elif route == "cnn_alternative":
        guards.append("analog recovery from alternatives (Entry 026)")

    notes: list[str] = []
    reproduced: bool | None = None
    outcome: str | None = None
    scenario = scenario_from_report(report, threshold)
    cnn_confidence = number(cnn, "confidence")
    classical_confidence = number(classical, "confidence")

    if scenario is not None and cnn_confidence is not None:
        replayed = fuse_modulation(
            scenario.ml_label, cnn_confidence, scenario.classical_family,
            threshold=threshold,
            alternatives=tuple(name for name, _ in scenario.ranked_alternatives),
            ranked_alternatives=scenario.ranked_alternatives,
            classical_evidence=scenario.classical_evidence,
            classical_confidence=classical_confidence,
        )
        outcome = classify_outcome(replayed, scenario)
        recorded_label = fusion.get("label")
        reproduced = (recorded_label is None or replayed.label == recorded_label)
        if reproduced is False:
            notes.append(
                f"Re-running fusion on the recorded inputs returned "
                f"{replayed.label!r}, but the report recorded {recorded_label!r}. The "
                "plotted point may not correspond to this landscape.")
    elif cnn.get("available") is False:
        notes.append(
            "No CNN prediction was available for this capture, so it has no position on "
            "the CNN-confidence axis.")

    return CapturePosition(
        cnn_label=cnn.get("label"),
        cnn_confidence=cnn_confidence,
        classical_family=classical.get("family"),
        classical_confidence=classical_confidence,
        snr_db=number(parameters, "snr_db"),
        fused_label=fusion.get("label"),
        trust_score=number(fusion, "trust_score"),
        review_recommended=bool(fusion.get("review_recommended")),
        outcome=outcome,
        guards=tuple(guards),
        reproduced=reproduced,
        notes=tuple(notes),
    )


def threshold_lines(scenario: LandscapeScenario) -> list[dict]:
    """The literal constants the plotted boundaries come from.

    Returned so the UI can annotate the map with the same numbers the code uses, rather
    than with numbers typed into the page.
    """

    lines = [{
        "axis": "cnn",
        "value": float(scenario.threshold),
        "name": "CNN acceptance threshold",
        "source": "fuse_modulation(threshold=...)",
    }]
    if (scenario.ml_label in ANALOG_LABELS
            and scenario.classical_family in DIGITAL_FAMILIES):
        lines.append({
            "axis": "classical",
            "value": float(DIGITAL_FAMILY_MIN_CONFIDENCE),
            "name": "Digital-family sufficiency floor",
            "source": "DIGITAL_FAMILY_MIN_CONFIDENCE",
        })
    return lines


def scenario_variants(base: LandscapeScenario) -> dict[str, LandscapeScenario]:
    """Named contexts that reach each distinct fusion regime.

    Offered so an analyst can see the guards that this capture did not trigger. Each is
    still evaluated by the real function; only the categorical context changes.
    """

    return {
        "This capture": base,
        "CNN analog vs classical digital (guard)": replace(
            base, ml_label="WBFM", classical_family="FSK-like"),
        "Classical analog with envelope evidence": replace(
            base, classical_family=ANALOG_FAMILY,
            classical_evidence={"envelope_flatness": 0.55, "amplitude_cv": 0.20}),
        "Classical analog, no evidence": replace(
            base, classical_family=ANALOG_FAMILY, classical_evidence=None),
        "CNN and classical agree": replace(
            base, classical_family="PSK-like", ml_label="QPSK"),
    }
