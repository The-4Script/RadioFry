import json

from gui.modulation_helpers import _expected_cnn_accuracy, low_snr_bias_labels
from gui.status import correlation_status


def test_expected_cnn_accuracy_reads_metrics_fixture(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy_by_snr": {"-10": 0.2, "0": 0.8}}), encoding="utf-8")

    assert _expected_cnn_accuracy(-5, metrics_path) == 0.5


def test_low_snr_bias_labels_are_derived_from_confusion_matrix() -> None:
    metrics = {
        "classes": ["A", "B", "C"],
        "confusion_matrices": {"low": [[8, 1, 1], [1, 4, 5], [6, 2, 2]]},
    }

    assert low_snr_bias_labels(metrics) == {"A"}


def test_correlation_status_requires_a_detected_match() -> None:
    assert correlation_status({"sync_pattern": None, "sync_match_score": 0.0}) == ("Needs review", "review")
    assert correlation_status({"sync_pattern": "hdlc", "sync_match_score": 0.0}) == ("Available", "ready")
    assert correlation_status({"sync_pattern": None, "sync_match_score": 0.75}) == ("Available", "ready")