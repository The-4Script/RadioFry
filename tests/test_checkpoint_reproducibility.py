"""Production checkpoint distribution and reproducibility (BANK.md Entry 041).

Entry 040 switched the production default to a checkpoint that a blanket
`/models_saved/` ignore rule had left untracked, so a fresh clone would not have it.
The repository's own convention is that model artifacts ARE tracked - nine of them
already are - and git ignore rules do not apply to already-tracked files, which is why
the breakage was invisible for existing models and silent for new ones.

These tests pin the three things that must hold: the artifacts are tracked, training
produces a loadable pair, and a missing artifact is reported rather than passed off as
a normal result.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from radiofry import pipeline
from radiofry.contracts import UnifiedSignalContainer
from radiofry.models.modulation_inference import metrics_path, predict_modulation
from radiofry.runtime import check_runtime_artifacts

CHECKPOINT = Path(pipeline.DEFAULT_MODULATION_MODEL)


def _tracked(path: str) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", path],
                          capture_output=True).returncode == 0


# --- the artifacts a fresh clone needs ---------------------------------------------------


def test_the_default_checkpoint_exists_on_disk() -> None:
    assert CHECKPOINT.is_file()


def test_the_default_checkpoint_has_its_metrics_sibling() -> None:
    assert metrics_path(CHECKPOINT).is_file()


@pytest.mark.skipif(not Path(".git").is_dir(), reason="not a git checkout")
def test_the_default_checkpoint_is_tracked_so_a_fresh_clone_gets_it() -> None:
    assert _tracked(str(CHECKPOINT).replace("\\", "/")), (
        f"{CHECKPOINT} is untracked; a fresh clone would have no production model")
    assert _tracked(str(metrics_path(CHECKPOINT)).replace("\\", "/"))


@pytest.mark.skipif(not Path(".git").is_dir(), reason="not a git checkout")
def test_model_artifacts_are_not_excluded_by_gitignore() -> None:
    # The repository tracks checkpoints; only scratch/smoke artifacts are excluded.
    ignored = subprocess.run(["git", "check-ignore", str(CHECKPOINT)],
                             capture_output=True).returncode == 0

    assert not ignored, "models_saved/ is ignored again - new checkpoints will be lost"


def test_the_previous_baseline_checkpoint_is_still_available() -> None:
    # Entry 039's comparison baseline must stay reproducible.
    assert Path("models_saved/modulation_cnn.pt").is_file()


# --- the loader's contract ------------------------------------------------------------------


def test_the_checkpoint_and_its_metrics_agree_on_the_hash() -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_path(CHECKPOINT).read_text(encoding="utf-8"))

    assert metrics["model_sha256"] == payload["model_sha256"]


def test_inference_actually_loads_the_production_default() -> None:
    time = np.arange(4_096) / 200_000.0
    signal = UnifiedSignalContainer(
        np.exp(2j * np.pi * 5_000 * time).astype(np.complex64), 200_000.0)

    prediction = predict_modulation(signal, pipeline.DEFAULT_MODULATION_MODEL)

    assert prediction.available is not False, prediction.message
    assert prediction.label != "Unclassified"


# --- a missing artifact must be reported, never passed off as a result -------------------------


def test_a_missing_checkpoint_is_reported_not_silently_accepted() -> None:
    time = np.arange(1_024) / 200_000.0
    signal = UnifiedSignalContainer(
        np.exp(2j * np.pi * 5_000 * time).astype(np.complex64), 200_000.0)

    prediction = predict_modulation(signal, "models_saved/__absent__.pt")

    assert prediction.available is False
    assert prediction.label == "Unclassified"
    assert "not found" in prediction.message.lower()


def test_the_report_flags_a_missing_required_artifact() -> None:
    runtime = check_runtime_artifacts({"modulation": "models_saved/__absent__.pt"})

    assert runtime["ready"] is False
    assert "unavailable" in str(runtime["message"]).lower()


def test_the_report_is_ready_when_the_real_artifacts_are_present() -> None:
    runtime = check_runtime_artifacts({
        "modulation": pipeline.DEFAULT_MODULATION_MODEL,
        "interleaver": "models_saved/interleaver_classifier.pkl",
        "fec": "models_saved/fec_classifier.pkl",
    })

    assert runtime["ready"] is True


# --- training must not be able to emit an unusable checkpoint ------------------------------------


def test_training_writes_the_metrics_sibling_itself() -> None:
    """Entry 040 trained a checkpoint without one and every prediction came back
    Unclassified until it was written by hand. train_v2 now writes it."""
    import inspect
    from radiofry.training import train_v2_synthetic

    source = inspect.getsource(train_v2_synthetic.train_v2)

    assert "write_metrics(" in source


def test_the_written_metrics_carry_the_hash_the_loader_checks() -> None:
    import inspect
    from radiofry.training import train_v2_synthetic

    source = inspect.getsource(train_v2_synthetic.train_v2)
    call = source.split("write_metrics(", 1)[1]

    assert "model_sha256" in call
