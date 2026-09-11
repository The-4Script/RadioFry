"""Real-data training on RadioML 2018.01A, on the GPU.

Three runs that differ only in what is trainable, carried over unchanged from the plan in
`docs/REALWORLD_DATASET.md` section 9:

  linear_probe  V3's convolutional trunk frozen, a fresh head trained on real data.
                Answers: do V3's learned features transfer, or was only the head wrong?
  finetune      every weight trainable, trunk initialised from V3. The candidate model.
  scratch       identical architecture, random initialisation. The control: if this matches
                `finetune`, V3's initialisation contributed nothing.

Interpretation is fixed in advance so the result cannot be rationalised after the fact:
  E1 high                -> the features transfer; the 0.20% was a head / label-space problem
  E1 low and E2 ~= E3    -> the learned representation itself does not transfer
  E2 > E3                -> synthetic pre-training is a useful initialisation regardless

Why the label space is the dataset's own 24 classes
---------------------------------------------------
V3 has 8 outputs; this dataset has 24, of which only 5 overlap. The head cannot transfer, so
only `features.*` is loaded and `finetune` is accurately "trunk transfer with a new head".
Training on the 5 mappable classes alone would let a 5-way model be compared against V3's
8-way score, which flatters it. The V3 comparison is instead made on the restricted view -
frames of the 5 mappable classes, each model free across its own full label space.

Guarantees enforced in code
---------------------------
* Training refuses to run on the CPU. `device.verify_cuda` proves a kernel launches and an
  optimiser step changes weights before any data is loaded.
* The sealed test split is not reachable from the training path without an explicit flag.
* The frozen production checkpoint cannot be used as an output path.
* Seeds are fixed, and the exact frame indices are content-addressed into the metrics file.
* Windows are built with `train_realworld.build_windows`, which is asserted element-for-element
  against the production inference path by `tests/test_realworld_training.py`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from radiofry.datasets.radioml2018 import (
    CLASSES,
    LABEL_MAP,
    SEALED_SPLIT,
    SNR_LEVELS,
    index_digest,
    load_indices,
    select_indices,
    split_manifest,
    verify_layout,
)
from radiofry.models.artifact_integrity import hash_torch_state_dict, metrics_path
from radiofry.training.device import describe_environment, verify_cuda
from radiofry.training.train_realworld import (
    FRAME_LENGTH,
    PRODUCTION_CHECKPOINT,
    build_windows,
    inference_window_starts,
)

FRAME_SAMPLES = 1024


@dataclass(frozen=True)
class RadioMLTrainingConfig:
    """One experiment. Everything that affects the result is on this object."""

    mode: str = "finetune"                      # finetune | linear_probe | scratch
    dataset_root: str = ""
    classes: tuple[str, ...] = CLASSES
    snr_db: tuple[int, ...] = SNR_LEVELS
    per_config_train: int = 400
    per_config_validation: int = 100
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 20260911
    initial_checkpoint: str | None = None
    output: str = ""
    label: str = ""
    require_capability: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"finetune", "linear_probe", "scratch"}:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode in {"finetune", "linear_probe"} and not self.initial_checkpoint:
            raise ValueError(f"mode {self.mode!r} requires initial_checkpoint")


@dataclass
class Pool:
    """Frames as float16 I/Q, converted to complex per batch to bound memory."""

    raw: np.ndarray
    labels: np.ndarray
    modulation: np.ndarray
    snr_db: np.ndarray
    indices: np.ndarray

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    def complex_frames(self, rows: np.ndarray) -> np.ndarray:
        block = self.raw[rows].astype(np.float32)
        return (block[..., 0] + 1j * block[..., 1]).astype(np.complex64)


def load_pool(root, split, *, classes, snr_db, per_config, seed,
              allow_sealed: bool = False) -> Pool:
    if split == SEALED_SPLIT and not allow_sealed:
        raise PermissionError(
            f"the '{SEALED_SPLIT}' split is the sealed held-out set and must not be read "
            "during training or model selection")
    indices = select_indices(root, split, classes=classes, snr_db=snr_db,
                             per_config=per_config, seed=seed)
    subset = load_indices(root, split, indices, dtype="float16")
    order = {name: i for i, name in enumerate(classes)}
    labels = np.array([order[name] for name in subset.modulation], dtype=np.int64)
    return Pool(subset.raw, labels, subset.modulation, subset.snr_db, indices)


def _build_model(config: RadioMLTrainingConfig, class_count: int):
    import torch
    from radiofry.models.modulation_cnn import ModulationCNN

    model = ModulationCNN(4, class_count)
    source = "random initialisation"

    if config.initial_checkpoint:
        payload = torch.load(config.initial_checkpoint, map_location="cpu",
                             weights_only=False)
        trunk = {k: v for k, v in payload["state_dict"].items() if k.startswith("features.")}
        if not trunk:
            raise RuntimeError("checkpoint carries no features.* weights to transfer")
        result = model.load_state_dict(trunk, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"unexpected keys: {result.unexpected_keys}")
        source = (f"{Path(config.initial_checkpoint).name} trunk only "
                  f"({len(trunk)} tensors); head is new for {class_count} classes")

    if config.mode == "linear_probe":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model, source


def evaluate(model, pool: Pool, device, *, batch: int = 2048):
    """Frame-level mean-softmax over the four production windows."""

    import torch

    starts = inference_window_starts(FRAME_SAMPLES, 4)
    model.eval()
    out = []
    for begin in range(0, len(pool), batch):
        rows = np.arange(begin, min(begin + batch, len(pool)))
        frames = pool.complex_frames(rows)
        stacked = np.concatenate(
            [build_windows(frames, np.full(len(frames), s)) for s in starts])
        with torch.inference_mode():
            tensor = torch.from_numpy(stacked).to(device, non_blocking=True)
            probabilities = torch.softmax(model(tensor), dim=1).float().cpu().numpy()
        out.append(probabilities.reshape(len(starts), len(frames), -1).mean(axis=0))
    return np.concatenate(out), pool.labels


def train(config: RadioMLTrainingConfig) -> dict:
    """Run one experiment on the GPU. Raises rather than falling back to the CPU."""

    import torch

    if not config.output:
        raise ValueError("an output checkpoint path is required")
    output = Path(config.output)
    if output.name == PRODUCTION_CHECKPOINT:
        raise PermissionError("refusing to overwrite the frozen production checkpoint")

    report = verify_cuda(require_capability=config.require_capability)
    device = torch.device("cuda")
    print(f"  device: {report.device_name} (sm_{report.capability}, "
          f"{report.total_memory_gb:.1f} GB), torch {report.torch_version} / "
          f"CUDA {report.cuda_version}")

    if not verify_layout(config.dataset_root):
        raise RuntimeError("dataset block layout does not match the stored labels")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed % (2 ** 32))
    rng = np.random.default_rng(config.seed)

    started = time.perf_counter()
    train_pool = load_pool(config.dataset_root, "train", classes=config.classes,
                           snr_db=config.snr_db, per_config=config.per_config_train,
                           seed=config.seed)
    validation_pool = load_pool(config.dataset_root, "val", classes=config.classes,
                                snr_db=config.snr_db,
                                per_config=config.per_config_validation,
                                seed=config.seed + 1)
    print(f"  train {len(train_pool):,} frames   validation {len(validation_pool):,} frames  "
          f"[{time.perf_counter()-started:.0f}s to load]")

    model, initialisation = _build_model(config, len(config.classes))
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  init: {initialisation}")
    print(f"  trainable {sum(p.numel() for p in trainable):,} of "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    optimiser = torch.optim.Adam(trainable, lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    criterion = torch.nn.CrossEntropyLoss()

    history: list[dict] = []
    best = {"validation_accuracy": -1.0, "epoch": -1, "state": None}
    train_started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(train_pool))
        total_loss, seen = 0.0, 0
        for offset in range(0, len(order), config.batch_size):
            rows = order[offset:offset + config.batch_size]
            frames = train_pool.complex_frames(rows)
            # One random window start per frame per epoch: timing-offset augmentation that
            # leaves the inference contract untouched.
            starts = rng.integers(0, FRAME_SAMPLES - FRAME_LENGTH + 1, size=len(rows))
            batch = torch.from_numpy(build_windows(frames, starts)).to(device,
                                                                       non_blocking=True)
            targets = torch.from_numpy(train_pool.labels[rows]).to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.detach().item() * len(rows)
            seen += len(rows)

        probabilities, truth = evaluate(model, validation_pool, device)
        accuracy = float((probabilities.argmax(axis=1) == truth).mean())
        scheduler.step(1.0 - accuracy)
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "validation_accuracy": accuracy,
                        "learning_rate": optimiser.param_groups[0]["lr"]})
        marker = ""
        if accuracy > best["validation_accuracy"]:
            best = {"validation_accuracy": accuracy, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            marker = "  <- best"
        print(f"    epoch {epoch:>2d}  loss {total_loss/seen:.4f}  val {accuracy:.4f}{marker}")

    elapsed = time.perf_counter() - started
    training_seconds = time.perf_counter() - train_started
    model.load_state_dict(best["state"])

    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hash_torch_state_dict(best["state"])
    torch.save({
        "state_dict": best["state"], "model_sha256": digest,
        "labels": list(config.classes), "input_channels": 4,
        "sample_length": FRAME_LENGTH, "features": "iqap",
        "dataset": "radioml2018.01a", "seed": config.seed,
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["validation_accuracy"],
        "mode": config.mode,
    }, output)

    metrics = {
        "model_sha256": digest,
        "mode": config.mode,
        "label": config.label,
        "dataset": {"name": "RadioML 2018.01A", "path": str(config.dataset_root),
                    "file": "GOLD_XYZ_OSC.0001_1024.hdf5",
                    "class_ordering": "measured FIXED ordering (BANK Entry 047)"},
        "classes": list(config.classes),
        "snr_db": list(config.snr_db),
        "split": json.loads(split_manifest(config.dataset_root)),
        "recording_grouping": ("none available - RadioML publishes no capture identifier; "
                               "independence inferred from measured absence of cross-SNR "
                               "waveform reuse, temporal contiguity and near-duplicates"),
        "sealed_split": SEALED_SPLIT,
        "initialisation": initialisation,
        "architecture": "ModulationCNN(input_channels=4, num_classes=%d)" % len(config.classes),
        "input_contract": {"frame_length": FRAME_LENGTH, "channels": "iqap",
                           "inference_windows": 4,
                           "window_starts": inference_window_starts(FRAME_SAMPLES, 4).tolist(),
                           "training_windows": "one random start per frame per epoch"},
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "optimiser": "Adam",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)",
        "train_frames": len(train_pool),
        "validation_frames": len(validation_pool),
        "per_config_train": config.per_config_train,
        "per_config_validation": config.per_config_validation,
        "train_snr_distribution": {str(int(s)): int((train_pool.snr_db == s).sum())
                                   for s in sorted(set(train_pool.snr_db.tolist()))},
        "device": report.as_dict(),
        "environment": describe_environment(),
        "elapsed_seconds": elapsed,
        "training_seconds": training_seconds,
        "best_epoch": best["epoch"],
        "best_validation_accuracy": best["validation_accuracy"],
        "history": history,
        "train_indices_sha256": index_digest(train_pool.indices),
        "validation_indices_sha256": index_digest(validation_pool.indices),
        "radiofry_label_map": dict(LABEL_MAP),
    }
    metrics_path(output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  wrote {output.name}  sha {digest[:16]}  best val "
          f"{best['validation_accuracy']:.4f} @ epoch {best['epoch']}  "
          f"[{training_seconds/60:.1f} min training]")
    return metrics
