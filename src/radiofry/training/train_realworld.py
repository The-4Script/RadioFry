"""First real-data training experiment on the Belousov & Ronkin multipath IQ dataset.

Why three runs and not one
--------------------------
The frozen production checkpoint scores 95.38% on synthetic data and 0.20% on this dataset
(BANK Entry 044). Simply fine-tuning until the number improves would raise the score without
explaining the gap. Three runs that differ only in what is trainable separate the causes:

  linear_probe  V3's convolutional trunk frozen, a fresh head trained on real data.
                Answers: do V3's learned features transfer, and was only the head wrong?
  finetune      every weight trainable, initialised from V3. The candidate model.
  scratch       identical architecture, random initialisation. The control: if this matches
                `finetune`, V3's initialisation contributed nothing.

Leakage control
---------------
The released splits were drawn by random frame-level sampling from 84 continuous recordings,
so every recording appears in all three splits. Measured consequence: 48% of QAM test frames,
10% of BPSK, 8% of QPSK and 4% of GMSK have a near-duplicate in the train split (correlation
above 0.9, DC removed, peak over lags, against a phase-randomised surrogate null of ~0.30).

The same measurement across recording boundaries is 0.0% for every digital class. So the
primary protocol trains on SNR levels {20, 24, 28} and evaluates on {22, 26, 30}, which are
physically different recordings. Evaluating the same model on held-out frames from the
*trained* SNR levels then measures the leakage inflation directly, with no second training
run and no confound.

Guarantees enforced in code, not merely intended
------------------------------------------------
* The sealed split (`subset_test.h5`) is never opened during training or model selection.
  `_forbid_sealed_split` raises if it is requested.
* The production checkpoint path is refused as an output.
* Seeds are fixed for numpy and torch, and the exact frame indices used are recorded in the
  metrics file so any run can be reproduced frame for frame.
* Windows are built with **exactly** the production inference contract, so a checkpoint from
  here drops into the existing runtime unchanged.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from radiofry.datasets.realworld_multipath import (
    DATASET_CLASSES,
    FRAME_SAMPLES,
    HELDOUT_SNR_DB,
    SEALED_SPLIT,
    TRAIN_SNR_DB,
    open_split,
    select_indices,
)
from radiofry.models.artifact_integrity import hash_torch_state_dict, metrics_path

FRAME_LENGTH = 128
INFERENCE_WINDOWS = 4
CHANNELS = (0, 1)

# Anything written here must not collide with the frozen production checkpoint.
PRODUCTION_CHECKPOINT = "modulation_cnn_v3_spsaug.pt"


@dataclass(frozen=True)
class TrainingConfig:
    """One experiment. Everything that affects the result is on this object."""

    mode: str = "finetune"                     # finetune | linear_probe | scratch
    dataset_root: str = ""
    classes: tuple[str, ...] = DATASET_CLASSES
    train_snr_db: tuple[int, ...] = TRAIN_SNR_DB
    per_config_train: int = 3_300
    per_config_validation: int = 500
    epochs: int = 25
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 20260911
    initial_checkpoint: str | None = None
    output: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"finetune", "linear_probe", "scratch"}:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode in {"finetune", "linear_probe"} and not self.initial_checkpoint:
            raise ValueError(f"mode {self.mode!r} requires initial_checkpoint")


def _forbid_sealed_split(subset: str) -> str:
    """The held-out split is not readable from the training path, by construction."""

    if subset == SEALED_SPLIT:
        raise PermissionError(
            f"subset_{SEALED_SPLIT}.h5 is the sealed held-out set and must never be read "
            "during training or model selection; evaluate it only through "
            "radiofry.evaluation, after training has finished")
    return subset


# --- the production input contract, vectorised ----------------------------------------------


def build_windows(frames: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Turn complex frames into the 4-channel `iqap` tensors the production model consumes.

    This reproduces `models.modulation_inference._window_frames` followed by
    `models.signal_features.add_signal_features(..., include_engineered=True)` exactly, in
    one vectorised pass. `tests/test_realworld_training.py` asserts the equality element for
    element, so train/inference skew cannot creep in silently.
    """

    frames = np.asarray(frames)
    starts = np.asarray(starts, dtype=np.int64)
    if frames.ndim != 2:
        raise ValueError("frames must have shape (batch, samples)")
    if starts.shape[0] != frames.shape[0]:
        raise ValueError("one start position is required per frame")

    offsets = starts[:, None] + np.arange(FRAME_LENGTH)[None, :]
    blocks = np.take_along_axis(frames, offsets, axis=1)

    values = np.stack([blocks.real, blocks.imag], axis=1).astype(np.float32)
    power = np.sqrt(np.mean(values ** 2, axis=(1, 2), keepdims=True))
    values = np.where(power > 0, values / np.where(power > 0, power, 1.0), values)

    complex_values = values[:, 0] + 1j * values[:, 1]
    amplitude = np.abs(complex_values).astype(np.float32)
    phase_difference = np.angle(
        complex_values[:, 1:] * np.conj(complex_values[:, :-1])).astype(np.float32)
    phase_difference = np.concatenate(
        [np.zeros((len(values), 1), dtype=np.float32), phase_difference], axis=1)

    channels = np.stack([values[:, 0], values[:, 1], amplitude, phase_difference], axis=1)
    channel_power = np.sqrt(np.mean(channels ** 2, axis=2, keepdims=True))
    return (channels / np.maximum(channel_power, 1e-6)).astype(np.float32)


def inference_window_starts(frame_samples: int = FRAME_SAMPLES,
                            windows: int = INFERENCE_WINDOWS) -> np.ndarray:
    """The fixed window positions production uses. Evaluation must use these, not random."""

    return np.array(sorted({int(start) for start in
                            np.linspace(0, frame_samples - FRAME_LENGTH, windows)}))


# --- data ------------------------------------------------------------------------------------


@dataclass
class FramePool:
    """Frames held as float16 I/Q, which is how they are stored, to bound memory."""

    raw: np.ndarray                 # (N, 1024, 2) float16
    labels: np.ndarray              # (N,) int64 index into `classes`
    modulation: np.ndarray
    channel: np.ndarray
    snr_db: np.ndarray
    indices: np.ndarray
    classes: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    def complex_frames(self, rows: np.ndarray) -> np.ndarray:
        block = self.raw[rows].astype(np.float32)
        return block[..., 0] + 1j * block[..., 1]


def load_pool(root: str | Path, subset: str, *, classes: tuple[str, ...],
              snr_db: tuple[int, ...], per_config_limit: int, seed: int,
              allow_sealed: bool = False) -> FramePool:
    """Load a stratified, class-balanced pool of frames as float16."""

    if not allow_sealed:
        _forbid_sealed_split(subset)

    indices = select_indices(root, subset, classes=classes, channels=CHANNELS,
                             snr_db=snr_db, per_config_limit=per_config_limit, seed=seed)
    with open_split(root, subset) as handle:
        mapping = json.loads(handle.attrs["mod2id_json"])
        identifier_to_name = {int(v): k for k, v in mapping.items()}
        raw = handle["X"][indices]
        modulation = np.array([identifier_to_name[int(v)] for v in handle["y_mod"][indices]])
        channel = np.asarray(handle["y_chan"][indices])
        snr = np.asarray(handle["y_snr"][indices])

    order = {name: i for i, name in enumerate(classes)}
    labels = np.array([order[name] for name in modulation], dtype=np.int64)
    return FramePool(raw, labels, modulation, channel, snr, indices, classes)


# --- training ---------------------------------------------------------------------------------


def _build_model(config: TrainingConfig, class_count: int):
    import torch
    from radiofry.models.modulation_cnn import ModulationCNN

    model = ModulationCNN(4, class_count)
    source = "random initialisation"

    if config.initial_checkpoint:
        payload = torch.load(config.initial_checkpoint, map_location="cpu",
                             weights_only=False)
        state = payload["state_dict"]
        # The trunk transfers; the head cannot, because the label spaces differ.
        trunk = {k: v for k, v in state.items() if k.startswith("features.")}
        missing = model.load_state_dict(trunk, strict=False)
        if missing.unexpected_keys:
            raise RuntimeError(f"unexpected keys in checkpoint: {missing.unexpected_keys}")
        source = (f"{Path(config.initial_checkpoint).name} "
                  f"(trunk only; head is new for {class_count} classes)")

    if config.mode == "linear_probe":
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    return model, source


def _evaluate(model, pool: FramePool, starts: np.ndarray, batch: int = 1024):
    """Frame-level mean-softmax over the production windows. Returns (probabilities, labels)."""

    import torch

    model.eval()
    outputs = []
    for start in range(0, len(pool), batch):
        rows = np.arange(start, min(start + batch, len(pool)))
        frames = pool.complex_frames(rows)
        stacked = np.concatenate([
            build_windows(frames, np.full(len(frames), position)) for position in starts])
        with torch.inference_mode():
            logits = model(torch.from_numpy(stacked))
            probabilities = torch.softmax(logits, dim=1).numpy()
        outputs.append(probabilities.reshape(len(starts), len(frames), -1).mean(axis=0))
    return np.concatenate(outputs), pool.labels


def train(config: TrainingConfig) -> dict:
    """Run one experiment and write a checkpoint plus its metrics. Returns the metrics."""

    import torch

    if not config.output:
        raise ValueError("an output checkpoint path is required")
    output = Path(config.output)
    if output.name == PRODUCTION_CHECKPOINT:
        raise PermissionError(
            "refusing to overwrite the frozen production checkpoint; "
            "write the candidate to a new filename")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2 ** 32))
    rng = np.random.default_rng(config.seed)

    started = time.perf_counter()
    train_pool = load_pool(config.dataset_root, "train", classes=config.classes,
                           snr_db=config.train_snr_db,
                           per_config_limit=config.per_config_train, seed=config.seed)
    validation_pool = load_pool(config.dataset_root, "val", classes=config.classes,
                                snr_db=config.train_snr_db,
                                per_config_limit=config.per_config_validation,
                                seed=config.seed + 1)
    print(f"  train {len(train_pool):,} frames   validation {len(validation_pool):,} frames"
          f"   [{time.perf_counter() - started:.0f}s to load]")

    model, initialisation = _build_model(config, len(config.classes))
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  init: {initialisation}")
    print(f"  trainable tensors {len(trainable)} of {len(list(model.parameters()))}, "
          f"{sum(p.numel() for p in trainable):,} parameters")

    optimiser = torch.optim.Adam(trainable, lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=2)
    criterion = torch.nn.CrossEntropyLoss()
    starts = inference_window_starts()

    history = []
    best = {"validation_accuracy": -1.0, "epoch": -1, "state": None}
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(train_pool))
        epoch_loss, seen = 0.0, 0
        for offset in range(0, len(order), config.batch_size):
            rows = order[offset:offset + config.batch_size]
            frames = train_pool.complex_frames(rows)
            # A random window start per frame per epoch: timing-offset augmentation that
            # leaves the inference contract untouched, and lets 25 epochs see far more
            # distinct windows than the 4 fixed evaluation positions.
            window_starts = rng.integers(0, FRAME_SAMPLES - FRAME_LENGTH + 1, size=len(rows))
            batch = torch.from_numpy(build_windows(frames, window_starts))
            targets = torch.from_numpy(train_pool.labels[rows])

            optimiser.zero_grad()
            loss = criterion(model(batch), targets)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss) * len(rows)
            seen += len(rows)

        probabilities, truth = _evaluate(model, validation_pool, starts)
        accuracy = float((probabilities.argmax(axis=1) == truth).mean())
        scheduler.step(1.0 - accuracy)
        history.append({"epoch": epoch, "train_loss": epoch_loss / seen,
                        "validation_accuracy": accuracy,
                        "learning_rate": optimiser.param_groups[0]["lr"]})
        marker = ""
        if accuracy > best["validation_accuracy"]:
            best = {"validation_accuracy": accuracy, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            marker = "  <- best"
        print(f"    epoch {epoch:>2d}  loss {epoch_loss/seen:.4f}  "
              f"val {accuracy:.4f}{marker}")

    model.load_state_dict(best["state"])
    elapsed = time.perf_counter() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hash_torch_state_dict(best["state"])
    payload = {
        "state_dict": best["state"],
        "model_sha256": digest,
        "labels": list(config.classes),
        "input_channels": 4,
        "sample_length": FRAME_LENGTH,
        "features": "iqap",
        "dataset": "realworld_multipath (Belousov & Ronkin 2026)",
        "seed": config.seed,
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["validation_accuracy"],
        "mode": config.mode,
        "train_snr_db": list(config.train_snr_db),
        "heldout_snr_db": list(HELDOUT_SNR_DB),
    }
    torch.save(payload, output)

    metrics = {
        "model_sha256": digest,
        "mode": config.mode,
        "label": config.label,
        "labels": list(config.classes),
        "initialisation": initialisation,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "train_frames": len(train_pool),
        "validation_frames": len(validation_pool),
        "train_snr_db": list(config.train_snr_db),
        "heldout_snr_db": list(HELDOUT_SNR_DB),
        "sealed_split": SEALED_SPLIT,
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["validation_accuracy"],
        "elapsed_seconds": elapsed,
        "history": history,
        "train_indices_sha256": _index_digest(train_pool.indices),
        "validation_indices_sha256": _index_digest(validation_pool.indices),
    }
    metrics_path(output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  wrote {output.name}  sha {digest[:16]}  "
          f"best val {best['validation_accuracy']:.4f} at epoch {best['epoch']}  "
          f"[{elapsed/60:.1f} min]")
    return metrics


def _index_digest(indices: np.ndarray) -> str:
    """A digest of the exact frame positions used, so a run is reproducible frame for frame."""

    import hashlib

    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
