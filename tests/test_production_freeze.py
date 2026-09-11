"""The frozen production backend, pinned so drift is detectable.

The backend was frozen before real-data collection so that any later real-data result is
measured against a known, unchanging system. These are identity assertions, not quality
assertions: they do not say the model is good, they say it is *the same model* the recorded
baselines were measured on (BANK.md Entry 044).

If one of these fails, either something changed that should not have, or a deliberate
change was made and the freeze needs re-cutting with a new BANK entry recording why.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional ml extra")

from radiofry import pipeline  # noqa: E402
from radiofry.models.artifact_integrity import hash_torch_state_dict  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- the frozen identities -------------------------------------------------------------
FROZEN_CHECKPOINT = "models_saved/modulation_cnn_v3_spsaug.pt"
FROZEN_STATE_DICT_SHA256 = (
    "a7b02533a7c7129c5435abb7fa12f95fb1af90188115c7c3cb96d9e580ce5a40")
FROZEN_V1_SHA256 = (
    "d6d3f918687d0700a43e46211c3f04b9ef74be9d8b232eac5d0e4cf4bf2390ab")
FROZEN_LABELS = ("8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK")

CHECKPOINT_PATH = PROJECT_ROOT / FROZEN_CHECKPOINT
V1_CAPTURES = PROJECT_ROOT / "data" / "synthetic_v1" / "captures"

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_PATH.is_file(),
    reason="production checkpoint is gitignored and absent from this checkout")
requires_v1 = pytest.mark.skipif(not V1_CAPTURES.exists(),
                                 reason="frozen V1 dataset not present")


@requires_checkpoint
def test_the_production_default_points_at_the_frozen_checkpoint() -> None:
    assert pipeline.DEFAULT_MODULATION_MODEL == FROZEN_CHECKPOINT, (
        "the production default moved; re-cut the freeze and record why")


@requires_checkpoint
def test_the_frozen_checkpoint_weights_are_unchanged() -> None:
    """The identity every recorded baseline was measured against."""
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    assert hash_torch_state_dict(payload["state_dict"]) == FROZEN_STATE_DICT_SHA256


@requires_checkpoint
def test_the_frozen_checkpoint_configuration_is_unchanged() -> None:
    """Inference configuration is part of the freeze, not just the weights.

    The 128-sample frame and the 4-channel iqap representation are what the 95.38%
    baseline was measured with; changing either invalidates it even if the weights match.
    """
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    assert tuple(payload["labels"]) == FROZEN_LABELS
    assert payload["input_channels"] == 4
    assert payload["features"] == "iqap"
    assert payload["sample_length"] == 128


@requires_checkpoint
def test_the_inference_window_configuration_is_unchanged() -> None:
    """Entry 040 rejected longer windows and Entry 041 confirmed 128x4 is optimal."""
    from radiofry.models.modulation_inference import DEFAULT_INFERENCE_WINDOWS

    assert DEFAULT_INFERENCE_WINDOWS == 4


@requires_v1
def test_the_frozen_v1_dataset_is_byte_identical() -> None:
    """Also asserted elsewhere; repeated here so the freeze is checkable in one place."""
    digest = hashlib.sha256()
    for capture in sorted(V1_CAPTURES.glob("*.iq"), key=lambda path: path.name):
        digest.update(capture.read_bytes())

    assert digest.hexdigest() == FROZEN_V1_SHA256


@requires_checkpoint
@requires_v1
def test_the_frozen_system_still_reproduces_a_known_decision() -> None:
    """An end-to-end anchor: same capture, same checkpoint, same answer.

    Hash equality proves the weights are unchanged; this proves the *pipeline around
    them* still produces the same decision from the same bytes.
    """
    import json

    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.models.modulation_inference import predict_modulation
    from radiofry.pipeline import load_capture

    truth = json.loads(
        (V1_CAPTURES / "QPSK_snr20dB_r000.json").read_text(encoding="utf-8"))
    entry = next(f for f in truth["files"] if f["file_format"] == "iq")
    signal = load_capture(V1_CAPTURES / "QPSK_snr20dB_r000.iq",
                          sample_rate=truth["signal"]["sample_rate_hz"],
                          iq_format=IQFormat(entry["dtype"], entry["byte_order"]))

    prediction = predict_modulation(signal, pipeline.DEFAULT_MODULATION_MODEL)

    assert prediction.label == "QPSK"
    assert prediction.confidence > 0.99, (
        f"the frozen system gave {prediction.confidence:.4f} for a capture it scored "
        "0.9982 on when the freeze was cut")


@requires_checkpoint
def test_the_frozen_system_is_deterministic() -> None:
    """No hidden randomness: the same bytes must give the same answer every time."""
    from radiofry.contracts import UnifiedSignalContainer
    from radiofry.models.modulation_inference import predict_modulation

    rng = np.random.default_rng(20260911)
    iq = ((rng.normal(size=8_192) + 1j * rng.normal(size=8_192))
          / np.sqrt(2)).astype(np.complex64)
    signal = UnifiedSignalContainer(iq, 200_000.0, "iq")

    first = predict_modulation(signal, pipeline.DEFAULT_MODULATION_MODEL)
    second = predict_modulation(signal, pipeline.DEFAULT_MODULATION_MODEL)

    assert first.label == second.label
    assert first.confidence == pytest.approx(second.confidence, abs=1e-9)
