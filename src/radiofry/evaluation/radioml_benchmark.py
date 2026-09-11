"""Evaluate any checkpoint on RadioML 2018.01A held-out frames.

Two checkpoints with different label spaces have to be compared without flattering either.
The frozen V3 model emits 8 digital classes; a model trained here emits the dataset's 24. Five
classes overlap exactly (BPSK, QPSK, 8PSK, 16QAM->QAM16, 64QAM->QAM64).

The comparison is therefore made on the **restricted view**: frames of those five classes only,
with each model free to answer across its own full label space. V3 chooses among 8, the
candidate among 24, and both are scored on the same frames against the same truth. Scoring V3
on a 5-way argmax over just its mappable outputs would be a different, easier task, so both
numbers are reported and labelled.

Frames of the other 19 classes are **out of label space** for V3, not errors: a model cannot
be right about a class it has no output for. They are reported separately.

Inference uses the frozen production contract exactly - 128-sample windows at the four fixed
positions, `iqap` channels, mean-softmax - via `train_realworld.build_windows`, which is
asserted element-for-element against the runtime path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from radiofry.datasets.radioml2018 import (
    CLASSES,
    LABEL_MAP,
    load_indices,
    select_indices,
)
from radiofry.training.train_realworld import build_windows, inference_window_starts

FRAME_SAMPLES = 1024
MAPPABLE = tuple(LABEL_MAP)                    # dataset-side names with a V3 counterpart


@dataclass
class EvaluationResult:
    """Everything measured, kept together so a report cannot quote one number alone."""

    checkpoint: str
    model_sha256: str
    labels: list[str]
    frames: int
    overall_accuracy: float
    per_class: dict = field(default_factory=dict)
    per_snr: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"checkpoint": self.checkpoint, "model_sha256": self.model_sha256,
                "labels": self.labels, "frames": self.frames,
                "overall_accuracy": self.overall_accuracy, "per_class": self.per_class,
                "per_snr": self.per_snr, "confusion": self.confusion,
                "calibration": self.calibration, "notes": self.notes}


def load_model(checkpoint_path: str | Path, device):
    """Load a checkpoint and return (model, labels, sha256, payload)."""

    import torch
    from radiofry.models.artifact_integrity import hash_torch_state_dict
    from radiofry.models.modulation_cnn import ModulationCNN

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = list(payload["labels"])
    model = ModulationCNN(int(payload.get("input_channels", 4)), len(labels))
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    return model, labels, hash_torch_state_dict(payload["state_dict"]), payload


def predict(model, frames: np.ndarray, device, *, batch: int = 2048) -> np.ndarray:
    """Frame-level mean-softmax over the four production windows. Returns (N, classes)."""

    import torch

    starts = inference_window_starts(FRAME_SAMPLES, 4)
    out = []
    for begin in range(0, len(frames), batch):
        chunk = frames[begin:begin + batch]
        stacked = np.concatenate(
            [build_windows(chunk, np.full(len(chunk), s)) for s in starts])
        with torch.inference_mode():
            tensor = torch.from_numpy(stacked).to(device, non_blocking=True)
            probabilities = torch.softmax(model(tensor), dim=1).float().cpu().numpy()
        out.append(probabilities.reshape(len(starts), len(chunk), -1).mean(axis=0))
    return np.concatenate(out)


def _calibration(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Confidence behaviour, including the property that mattered most on Dataset 1.

    A model whose wrong answers are more confident than its right ones is worse than
    inaccurate: it is unsafe, and no threshold can fix it.
    """

    right = confidence[correct]
    wrong = confidence[~correct]
    summary = {
        "median_confidence_correct": float(np.median(right)) if right.size else None,
        "median_confidence_wrong": float(np.median(wrong)) if wrong.size else None,
        "max_confidence_wrong": float(wrong.max()) if wrong.size else None,
        "wrong_above_0.9": int((wrong >= 0.9).sum()),
        "wrong_above_0.99": int((wrong >= 0.99).sum()),
        "wrong_total": int(wrong.size),
    }
    if right.size and wrong.size:
        summary["calibration_inverted"] = bool(np.median(wrong) > np.median(right))
    return summary


def evaluate_checkpoint(checkpoint_path: str | Path, dataset_root: str | Path, *,
                        split: str = "test",
                        classes: tuple[str, ...] = MAPPABLE,
                        snr_db: tuple[int, ...] | None = None,
                        per_config: int = 80,
                        seed: int = 99,
                        device=None,
                        truth_space: str = "radiofry") -> EvaluationResult:
    """Score one checkpoint on held-out frames.

    `truth_space` selects how the dataset's class name is compared with the model's output:
      "radiofry" - truth is the mapped production label (BPSK, QPSK, 8PSK, QAM16, QAM64),
                   which is what the frozen V3 emits;
      "dataset"  - truth is the dataset's own class name, for a model trained on those.
    The model is always free to answer across its entire label space.
    """

    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, digest, _ = load_model(checkpoint_path, device)

    indices = select_indices(dataset_root, split, classes=classes, snr_db=snr_db,
                             per_config=per_config, seed=seed)
    subset = load_indices(dataset_root, split, indices, dtype="float16")
    frames = subset.complex_frames()

    if truth_space == "radiofry":
        truth = subset.radiofry_label
    elif truth_space == "dataset":
        truth = subset.modulation
    else:
        raise ValueError("truth_space must be 'radiofry' or 'dataset'")

    probabilities = predict(model, frames, device)
    chosen = np.array([labels[i] for i in probabilities.argmax(axis=1)])
    confidence = probabilities.max(axis=1)
    correct = chosen == truth

    result = EvaluationResult(
        checkpoint=str(checkpoint_path), model_sha256=digest, labels=labels,
        frames=int(len(subset)), overall_accuracy=float(correct.mean()))

    for name in sorted(set(truth.tolist())):
        mask = truth == name
        predictions = chosen[mask]
        counts = {}
        for value in predictions:
            counts[value] = counts.get(value, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        result.per_class[name] = {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "most_common_predictions": top,
        }

    for snr in sorted(set(subset.snr_db.tolist())):
        mask = subset.snr_db == snr
        result.per_snr[int(snr)] = {"n": int(mask.sum()),
                                    "accuracy": float(correct[mask].mean())}

    for name in sorted(set(truth.tolist())):
        mask = truth == name
        row = {}
        for value in labels:
            count = int((chosen[mask] == value).sum())
            if count:
                row[value] = count
        result.confusion[name] = row

    result.calibration = _calibration(confidence, correct)

    unreachable = sorted(set(truth.tolist()) - set(labels))
    if unreachable:
        result.notes.append(
            f"classes with no output in this model (cannot be scored as correct): "
            f"{unreachable}")
    result.notes.append(f"model chooses among {len(labels)} classes")
    return result


def out_of_label_space(checkpoint_path: str | Path, dataset_root: str | Path, *,
                       split: str = "test", per_config: int = 40, seed: int = 98,
                       device=None) -> dict:
    """What a model emits for classes it has no output for. Not scored as errors."""

    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, _, _ = load_model(checkpoint_path, device)

    unmapped = tuple(c for c in CLASSES if c not in LABEL_MAP)
    indices = select_indices(dataset_root, split, classes=unmapped,
                             per_config=per_config, seed=seed)
    subset = load_indices(dataset_root, split, indices, dtype="float16")
    probabilities = predict(model, subset.complex_frames(), device)
    chosen = np.array([labels[i] for i in probabilities.argmax(axis=1)])
    confidence = probabilities.max(axis=1)

    per_class = {}
    for name in unmapped:
        mask = subset.modulation == name
        if not mask.any():
            continue
        counts = {}
        for value in chosen[mask]:
            counts[value] = counts.get(value, 0) + 1
        per_class[name] = sorted(counts.items(), key=lambda kv: -kv[1])[:3]

    return {"frames": int(len(subset)),
            "classes": list(unmapped),
            "median_confidence": float(np.median(confidence)),
            "above_0.9": int((confidence >= 0.9).sum()),
            "emits": per_class,
            "note": ("an N-class model with no 'none of these' output must pick something; "
                     "these are not errors")}


def write_report(results: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: (v.as_dict() if isinstance(v, EvaluationResult) else v)
               for k, v in results.items()}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
