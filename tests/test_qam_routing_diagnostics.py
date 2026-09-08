"""Structural facts ruled out as causes of the QAM routing failure.

These pin contracts that the Entry 008 investigation verified are NOT broken:
checkpoint label consistency, fusion's pass-through/reject-only behaviour, and
faithful label-to-order dispatch. They stay valid after any future fix.

Reference: BANK.md Entry 008.
"""

import json

import pytest

from radiofry.fusion.confidence_fusion import fuse_modulation
from radiofry.models.artifact_integrity import metrics_path

CHECKPOINT = "models_saved/modulation_cnn.pt"


def _checkpoint_labels() -> list[str]:
    torch = pytest.importorskip("torch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return list(payload["labels"])


def test_checkpoint_labels_match_the_metrics_class_list() -> None:
    metrics = json.loads(metrics_path(CHECKPOINT).read_text(encoding="utf-8"))

    assert _checkpoint_labels() == list(metrics["classes"])


def test_checkpoint_labels_are_sorted_as_training_builds_them() -> None:
    # train_modulation.py builds `labels = sorted({...})` and indexes targets by it,
    # so a sorted stored list rules out a label-permutation bug.
    labels = _checkpoint_labels()

    assert labels == sorted(labels)


def test_both_qam_classes_are_present_in_the_checkpoint() -> None:
    labels = _checkpoint_labels()

    assert "QAM16" in labels and "QAM64" in labels


@pytest.mark.parametrize("label", ["QAM16", "QAM64", "BPSK", "QPSK", "8PSK"])
def test_fusion_never_substitutes_a_different_modulation(label: str) -> None:
    # Fusion either forwards the CNN label or rejects it; it cannot turn QAM16 into
    # QAM64. Any mis-routing therefore originates upstream of fusion.
    accepted = fuse_modulation(label, 0.9, "QAM-like")
    rejected = fuse_modulation(label, 0.1, "QAM-like")

    assert accepted.label == label
    assert rejected.label == "Unclassified"


def test_fusion_forwards_a_confident_label_even_when_the_family_disagrees() -> None:
    result = fuse_modulation("QAM16", 0.9, "FSK-like")

    assert result.label == "QAM16"
    assert result.review_recommended is True


def test_dispatch_maps_each_qam_label_to_its_own_order() -> None:
    import numpy as np

    from radiofry.contracts import UnifiedSignalContainer
    from radiofry.decoding.demodulators.dispatch import demodulate_capture
    from radiofry.dsp.parameter_estimation import ParameterEstimate

    rng = np.random.default_rng(0)
    signal = UnifiedSignalContainer(
        (rng.normal(size=4_096) + 1j * rng.normal(size=4_096)).astype(np.complex64), 200_000.0
    )
    parameters = ParameterEstimate(None, None, 25_000.0)

    for label, expected in [("QAM16", "QAM16"), ("QAM64", "QAM64")]:
        result = demodulate_capture(signal, label, parameters)
        assert result.available
        assert result.result.modulation == expected
