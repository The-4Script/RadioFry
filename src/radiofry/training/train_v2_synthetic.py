"""Train a modulation classifier on the V1 synthetic generator (V2.0 baseline).

Design notes that matter for defensibility:

* Windows are extracted with **exactly** the runtime inference preprocessing
  (`models.modulation_inference._fixed_iq` normalisation, then
  `models.signal_features.add_signal_features`), so there is no train/inference skew.
* The split happens at **capture level**. Every window of a capture lands in exactly
  one of train/validation/test, so windows of one generated signal can never straddle
  the split.
* Training seeds live in a range disjoint from the frozen V1 dataset seeds, and the
  disjointness is asserted, so the V1 benchmark stays a genuine held-out set.
* A second "generalisation" capture set is built from a different seed range and is
  never used for training or model selection.

The frozen V1 dataset is never read or modified here.
"""

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.preprocessing import preprocess
from radiofry.models.artifact_integrity import hash_torch_state_dict, metrics_path
from radiofry.models.signal_features import add_signal_features
from radiofry.synthetic_gen.v1 import (
    MODULATIONS,
    SampleSpec,
    add_awgn,
    generate_source_bits,
    modulate,
)

FRAME_LENGTH = 128
SAMPLE_RATE_HZ = 200_000.0
SAMPLES_PER_SYMBOL = 8
# Entry 040: the V2 checkpoint was trained at a single oversampling factor, and Entry 039
# measured the consequence - fused accuracy 68.8% at sps 8 but 16.7% at sps 4 and 10.4%
# at sps 32. Training across the sweep lifts that to 95.8-100%. The capture SAMPLE COUNT
# is held constant, so higher oversampling means proportionally fewer symbols, matching
# what the runtime actually sees.
SAMPLES_PER_SYMBOL_SWEEP: tuple[int, ...] = (4, 8, 16, 32)
NUM_SYMBOLS = 1_024
SNR_SWEEP_DB: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0)
FSK_INDICES: tuple[float, ...] = (0.5, 1.0)

# Disjoint seed ranges. V1 dataset seeds come from blake2b digests of the generator's
# own key strings; these explicit low integers cannot collide with them in practice and
# the disjointness is asserted in build_dataset().
TRAIN_SEED_BASE = 1_000
GENERALISATION_SEED_BASE = 9_000_000


@dataclass(frozen=True)
class CaptureSpec:
    """One generated capture. `capture_id` is the unit the split operates on."""

    modulation: str
    label: str
    snr_db: float
    seed: int
    fsk_modulation_index: float | None
    samples_per_symbol: int = SAMPLES_PER_SYMBOL

    @property
    def capture_id(self) -> str:
        index = "" if self.fsk_modulation_index is None else f"_h{self.fsk_modulation_index}"
        return f"{self.modulation}{index}_sps{self.samples_per_symbol}_snr{self.snr_db:g}dB_s{self.seed}"


def build_capture_specs(
    *,
    replicates: int,
    seed_base: int,
    modulations: Sequence[str] = tuple(MODULATIONS),
    snr_sweep_db: Sequence[float] = SNR_SWEEP_DB,
    samples_per_symbol_sweep: Sequence[int] = SAMPLES_PER_SYMBOL_SWEEP,
) -> list[CaptureSpec]:
    """Enumerate captures; FSK is swept over both modulation indices."""

    specs: list[CaptureSpec] = []
    counter = 0
    for modulation in modulations:
        is_fsk = MODULATIONS[modulation].family == "fsk"
        indices: tuple[float | None, ...] = FSK_INDICES if is_fsk else (None,)
        for index in indices:
            for sps in samples_per_symbol_sweep:
                for snr_db in snr_sweep_db:
                    for _ in range(replicates):
                        specs.append(
                            CaptureSpec(
                                modulation=modulation,
                                label=MODULATIONS[modulation].radiofry_label,
                                snr_db=float(snr_db),
                                seed=seed_base + counter,
                                fsk_modulation_index=index,
                                samples_per_symbol=int(sps),
                            )
                        )
                        counter += 1
    return specs


def render_capture(spec: CaptureSpec) -> np.ndarray:
    """Generate one capture and apply the production preprocessing stage."""

    sample_spec = SampleSpec(
        modulation=spec.modulation,
        # Constant capture length: fewer symbols as oversampling rises.
        num_symbols=NUM_SYMBOLS * SAMPLES_PER_SYMBOL // spec.samples_per_symbol,
        samples_per_symbol=spec.samples_per_symbol,
        sample_rate_hz=SAMPLE_RATE_HZ,
        snr_db=spec.snr_db,
        seed=spec.seed,
        fsk_modulation_index=spec.fsk_modulation_index,
    )
    clean = modulate(generate_source_bits(sample_spec), sample_spec)
    noisy = add_awgn(clean, spec.snr_db, np.random.default_rng([spec.seed, 2]))[0]
    return preprocess(UnifiedSignalContainer(noisy, SAMPLE_RATE_HZ)).iq


def frames_from_capture(iq: np.ndarray, *, frame_length: int = FRAME_LENGTH) -> np.ndarray:
    """Non-overlapping windows, normalised exactly as runtime inference does."""

    count = iq.size // frame_length
    frames = np.empty((count, 4, frame_length), dtype=np.float32)
    for index in range(count):
        block = iq[index * frame_length : (index + 1) * frame_length]
        values = np.stack([block.real, block.imag]).astype(np.float32)
        power = np.sqrt(np.mean(values**2))
        if power > 0:
            values = values / power
        frames[index] = add_signal_features(values, include_engineered=True)
    return frames


def split_capture_specs(
    specs: Sequence[CaptureSpec], *, seed: int, proportions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> dict[str, list[CaptureSpec]]:
    """Deterministic capture-level split, stratified by (label, snr, fsk index)."""

    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1")
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[CaptureSpec]] = {}
    for spec in specs:
        groups.setdefault((spec.label, spec.snr_db, spec.fsk_modulation_index), []).append(spec)
    out: dict[str, list[CaptureSpec]] = {"train": [], "validation": [], "test": []}
    for key in sorted(groups, key=str):
        members = sorted(groups[key], key=lambda s: s.seed)
        order = rng.permutation(len(members))
        n_train = int(round(proportions[0] * len(members)))
        n_val = int(round(proportions[1] * len(members)))
        for position, index in enumerate(order):
            bucket = "train" if position < n_train else "validation" if position < n_train + n_val else "test"
            out[bucket].append(members[index])
    return out


def build_arrays(specs: Iterable[CaptureSpec], labels: Sequence[str]) -> dict[str, np.ndarray]:
    """Render captures into frames plus per-frame provenance."""

    label_index = {name: position for position, name in enumerate(labels)}
    frames, targets, snrs, capture_ids = [], [], [], []
    for spec in specs:
        window = frames_from_capture(render_capture(spec))
        frames.append(window)
        targets.append(np.full(window.shape[0], label_index[spec.label], dtype=np.int64))
        snrs.append(np.full(window.shape[0], spec.snr_db, dtype=np.float32))
        capture_ids.extend([spec.capture_id] * window.shape[0])
    return {
        "frames": np.concatenate(frames),
        "targets": np.concatenate(targets),
        "snrs": np.concatenate(snrs),
        "capture_ids": np.asarray(capture_ids),
    }


def build_dataset(
    *, replicates: int, split_seed: int, v1_dataset_dir: str | Path | None = None
) -> dict[str, object]:
    """Build the full capture-level split and assert it cannot leak into V1."""

    specs = build_capture_specs(replicates=replicates, seed_base=TRAIN_SEED_BASE)
    if v1_dataset_dir is not None:
        v1_seeds = _v1_capture_seeds(v1_dataset_dir)
        collisions = sorted({s.seed for s in specs} & v1_seeds)
        if collisions:
            raise ValueError(f"training seeds collide with frozen V1 capture seeds: {collisions}")
    labels = sorted({spec.label for spec in specs})
    split = split_capture_specs(specs, seed=split_seed)
    return {"labels": labels, "split": split, "specs": specs}


def _v1_capture_seeds(dataset_dir: str | Path) -> set[int]:
    seeds = set()
    for path in Path(dataset_dir, "captures").glob("*.json"):
        seeds.add(int(json.loads(path.read_text(encoding="utf-8"))["seeds"]["seed"]))
    return seeds


def train_v2(
    output_path: str | Path,
    *,
    replicates: int = 24,
    split_seed: int = 20_260_908,
    torch_seed: int = 7,
    epochs: int = 60,
    batch_size: int = 256,
    patience: int = 6,
    learning_rate: float = 1e-3,
    v1_dataset_dir: str | Path | None = "data/synthetic_v1",
    verbose: bool = True,
) -> dict[str, object]:
    """Train, select on validation loss, and write a checkpoint plus metrics."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from radiofry.models.modulation_cnn import ModulationCNN

    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)

    dataset = build_dataset(replicates=replicates, split_seed=split_seed, v1_dataset_dir=v1_dataset_dir)
    labels: list[str] = dataset["labels"]  # type: ignore[assignment]
    split: dict[str, list[CaptureSpec]] = dataset["split"]  # type: ignore[assignment]

    if verbose:
        print(f"captures: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))
    arrays = {name: build_arrays(specs, labels) for name, specs in split.items()}
    if verbose:
        for name, data in arrays.items():
            print(f"  {name}: {data['frames'].shape[0]} frames from {len(set(data['capture_ids']))} captures")

    model = ModulationCNN(input_channels=4, num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()

    def loader(name: str, shuffle: bool) -> DataLoader:
        data = arrays[name]
        return DataLoader(
            TensorDataset(torch.from_numpy(data["frames"]), torch.from_numpy(data["targets"])),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader, val_loader = loader("train", True), loader("validation", False)
    history: list[dict[str, float]] = []
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * batch.shape[0]
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.inference_mode():
            for batch, target in val_loader:
                logits = model(batch)
                val_loss += float(loss_fn(logits, target)) * batch.shape[0]
                correct += int((logits.argmax(dim=1) == target).sum())
                total += batch.shape[0]
        val_loss /= total
        val_accuracy = correct / total
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if verbose:
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"val_acc {val_accuracy:.4f}{'  *best' if improved else ''}")
        if stale >= patience:
            if verbose:
                print(f"  early stop after {epoch} epochs (patience {patience})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started

    checkpoint = {
        "state_dict": best_state,
        "model_sha256": hash_torch_state_dict(best_state),
        "labels": labels,
        "input_channels": 4,
        "sample_length": FRAME_LENGTH,
        "samples_per_symbol_sweep": list(SAMPLES_PER_SYMBOL_SWEEP),
        "features": "iqap",
        "dataset": "radiofry.synthetic_gen.v1 (V2.0 training build)",
        "seed": torch_seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    return {
        "checkpoint_path": str(output),
        "labels": labels,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_seconds": elapsed,
        "model_sha256": checkpoint["model_sha256"],
        "split_sizes": {name: len(specs) for name, specs in split.items()},
        "frame_counts": {name: int(data["frames"].shape[0]) for name, data in arrays.items()},
        "arrays": arrays,
        "model": model,
        "split": split,
    }


def write_metrics(checkpoint_path: str | Path, payload: dict[str, object]) -> Path:
    """Write the sibling metrics file the inference loader requires."""

    target = metrics_path(checkpoint_path)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target
