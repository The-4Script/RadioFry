"""Guards on the cross-model comparison.

The dangerous failure here is not a crash - it is a comparison that quietly favours one model.
V3 emits 8 classes and a model trained on this dataset emits 24, with 5 overlapping. Scoring
V3 on a 5-way argmax over only its mappable outputs would be an easier task than the one the
candidate faces, and the resulting "V3 vs candidate" line would be meaningless.

Also guarded: the calibration summary must actually detect inverted confidence, because that
was the single most important property found on Dataset 1 and a report that misses it is
worse than no report.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is part of the optional ml extras")
pytest.importorskip("h5py", reason="h5py is part of the optional dev/training extras")

from radiofry.datasets.radioml2018 import CLASSES, LABEL_MAP  # noqa: E402
from radiofry.evaluation.radioml_benchmark import (  # noqa: E402
    MAPPABLE,
    EvaluationResult,
    _calibration,
    evaluate_checkpoint,
    write_report,
)

DATASET = Path.home() / "Documents" / "RadioFry" / "dataset 2"
CHECKPOINT = Path("models_saved/modulation_cnn_v3_spsaug.pt")

requires_dataset = pytest.mark.skipif(
    not (DATASET.is_dir() and any(DATASET.glob("*.hdf5"))),
    reason="RadioML 2018.01A is not present on this machine")
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason="production checkpoint not present")


# --- the comparison must stay honest -------------------------------------------------------------


def test_the_mappable_set_is_exactly_the_five_exact_counterparts() -> None:
    assert set(MAPPABLE) == {"BPSK", "QPSK", "8PSK", "16QAM", "64QAM"}
    assert set(MAPPABLE) == set(LABEL_MAP)


def test_every_mappable_class_exists_in_the_dataset_label_space() -> None:
    for name in MAPPABLE:
        assert name in CLASSES


def test_unmapped_classes_are_the_remaining_nineteen() -> None:
    unmapped = [c for c in CLASSES if c not in LABEL_MAP]

    assert len(unmapped) == 19
    assert "GMSK" in unmapped and "4ASK" in unmapped and "FM" in unmapped


def test_an_unknown_truth_space_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_checkpoint(CHECKPOINT, DATASET, truth_space="whatever", per_config=1)


# --- calibration ------------------------------------------------------------------------------------


def test_calibration_detects_inverted_confidence() -> None:
    """Wrong answers more confident than right ones: the Dataset 1 finding."""

    confidence = np.array([0.95, 0.97, 0.99, 0.30, 0.35, 0.40])
    correct = np.array([False, False, False, True, True, True])

    summary = _calibration(confidence, correct)

    assert summary["calibration_inverted"] is True
    assert summary["median_confidence_wrong"] > summary["median_confidence_correct"]
    assert summary["wrong_above_0.9"] == 3
    assert summary["wrong_above_0.99"] == 1


def test_calibration_reports_healthy_confidence_as_not_inverted() -> None:
    confidence = np.array([0.99, 0.98, 0.40, 0.35])
    correct = np.array([True, True, False, False])

    summary = _calibration(confidence, correct)

    assert summary["calibration_inverted"] is False
    assert summary["wrong_above_0.9"] == 0


def test_calibration_handles_a_perfect_model_without_crashing() -> None:
    summary = _calibration(np.array([0.9, 0.8]), np.array([True, True]))

    assert summary["median_confidence_wrong"] is None
    assert summary["wrong_total"] == 0
    assert "calibration_inverted" not in summary


def test_calibration_handles_a_model_that_is_never_right() -> None:
    summary = _calibration(np.array([0.9, 0.8]), np.array([False, False]))

    assert summary["median_confidence_correct"] is None
    assert summary["wrong_total"] == 2


# --- the result record --------------------------------------------------------------------------------


def test_a_result_serialises_every_field_a_report_needs(tmp_path) -> None:
    result = EvaluationResult(checkpoint="x.pt", model_sha256="abc", labels=["A", "B"],
                              frames=10, overall_accuracy=0.5)

    payload = result.as_dict()
    for key in ("checkpoint", "model_sha256", "labels", "frames", "overall_accuracy",
                "per_class", "per_snr", "confusion", "calibration", "notes"):
        assert key in payload

    path = write_report({"run": result}, tmp_path / "report.json")
    assert path.is_file()


# --- against the real artefacts -------------------------------------------------------------------------


@requires_dataset
@requires_checkpoint
def test_v3_is_scored_across_its_whole_label_space_not_just_the_mappable_five() -> None:
    """If the model were restricted to 5 outputs the comparison would be flattered."""

    result = evaluate_checkpoint(CHECKPOINT, DATASET, split="test", per_config=2, seed=1)

    assert len(result.labels) == 8, "V3 must remain free to answer any of its 8 classes"
    assert any("among 8 classes" in note for note in result.notes)
    assert 0.0 <= result.overall_accuracy <= 1.0


@requires_dataset
@requires_checkpoint
def test_evaluation_covers_the_requested_frames_and_reports_per_class(tmp_path) -> None:
    result = evaluate_checkpoint(CHECKPOINT, DATASET, split="test",
                                 classes=("BPSK", "QPSK"), snr_db=(30,),
                                 per_config=3, seed=2)

    assert result.frames == 2 * 3
    assert set(result.per_class) == {"BPSK", "QPSK"}
    assert set(result.per_snr) == {30}
    for entry in result.per_class.values():
        assert entry["n"] == 3
