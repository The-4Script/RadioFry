"""Train and evaluate the modulation CNN on RML2016.10a."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the RadioFry modulation CNN")
    parser.add_argument("--data", default="data/RML2016.10a_dict.pkl")
    parser.add_argument("--output", default="models_saved/modulation_cnn.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--sample-length", type=int, default=128, help="Training frame length; use 1024 for native RML2018 frames.")
    parser.add_argument("--quiet", action="store_true", help="Disable epoch progress output.")
    args = parser.parse_args()
    metrics = train_modulation(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size, max_samples=args.max_samples, patience=args.patience, device=args.device, sample_length=args.sample_length, verbose=not args.quiet)
    print(json.dumps(metrics, indent=2))


def _stratified_indices(targets: np.ndarray, snrs: np.ndarray, limit: int, seed: int) -> np.ndarray:
    keys = np.array([f"{target}:{snr}" for target, snr in zip(targets, snrs)])
    groups = {key: np.flatnonzero(keys == key) for key in np.unique(keys)}
    if limit < len(groups):
        raise ValueError(f"max_samples must be at least the number of modulation/SNR strata ({len(groups)})")
    generator = np.random.default_rng(seed)
    base, remainder = divmod(limit, len(groups))
    selected = []
    for index, group in enumerate(groups.values()):
        count = min(len(group), base + (index < remainder))
        selected.append(generator.choice(group, size=count, replace=False))
    return np.sort(np.concatenate(selected))


def _canonicalize_frames(frames: np.ndarray, sample_length: int = 128) -> np.ndarray:
    """Match training frames to the fixed-length runtime inference contract."""

    if frames.ndim != 3 or frames.shape[2] == sample_length:
        return frames
    positions = np.linspace(0, frames.shape[2] - 1, sample_length).round().astype(np.int64)
    return frames[:, :, positions]


def load_rml_dataset(path: str | Path, *, max_samples: int | None = None, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load RML2016 pickle or RML2018 HDF5 data into canonical I/Q arrays."""

    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_rml_hdf5(path, max_samples=max_samples, seed=seed)
    with path.open("rb") as handle:
        dataset = pickle.load(handle, encoding="latin1")
    labels = sorted({modulation for modulation, _ in dataset})
    label_index = {label: index for index, label in enumerate(labels)}
    frames, targets, snrs = [], [], []
    for (modulation, snr), values in dataset.items():
        frames.append(np.asarray(values, dtype=np.float32))
        targets.extend([label_index[modulation]] * len(values))
        snrs.extend([int(snr)] * len(values))
    return np.concatenate(frames), np.asarray(targets, dtype=np.int64), np.asarray(snrs, dtype=np.int64), labels


def _load_rml_hdf5(path: Path, *, max_samples: int | None = None, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load common RML2018 HDF5 layouts without loading duplicate arrays."""

    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to load RML2018 HDF5 data") from error
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
        data_key = next((key for key in ("X", "x", "data", "signals") if key in keys), None)
        label_key = next((key for key in ("Y", "y", "labels", "label") if key in keys), None)
        snr_key = next((key for key in ("Z", "z", "snr", "snrs", "SNR") if key in keys), None)
        if data_key is None or label_key is None or snr_key is None:
            raise ValueError(f"RML HDF5 must contain data, labels, and SNR datasets; found {sorted(keys)}")
        raw_labels = np.asarray(handle[label_key])
        raw_snrs = np.asarray(handle[snr_key]).reshape(-1)
        data_shape = handle[data_key].shape
        class_names_attr = handle[label_key].attrs.get("class_names", handle.attrs.get("class_names", None))
        class_names = None if class_names_attr is None else [str(value.decode() if isinstance(value, bytes) else value) for value in np.asarray(class_names_attr).reshape(-1)]
        label_indices, inferred_names = _hdf5_label_indices(raw_labels)
        if class_names is None:
            class_names = _read_class_names(path, len(inferred_names)) or inferred_names
        selected = _stratified_indices(label_indices, raw_snrs, min(max_samples, len(raw_snrs)) if max_samples is not None else len(raw_snrs), seed) if max_samples is not None else np.arange(len(raw_snrs))
        raw_data = np.asarray(handle[data_key][selected])
    if raw_data.ndim not in {2, 3}:
        raise ValueError(f"RML HDF5 data must be 2- or 3-dimensional, got shape {raw_data.shape}")
    label_indices = label_indices[selected]
    raw_snrs = raw_snrs[selected]
    if np.iscomplexobj(raw_data):
        if raw_data.ndim == 2:
            frames = np.stack([raw_data.real, raw_data.imag], axis=1)
        elif raw_data.shape[1] == 1:
            frames = np.stack([raw_data[:, 0].real, raw_data[:, 0].imag], axis=1)
        else:
            frames = raw_data
    elif raw_data.shape[1] == 2:
        frames = raw_data
    elif raw_data.shape[-1] == 2:
        frames = np.moveaxis(raw_data, -1, 1)
    else:
        raise ValueError(f"RML HDF5 data must contain I/Q channels, got shape {raw_data.shape}")
    if len(frames) != len(label_indices) or len(frames) != len(raw_snrs):
        raise ValueError("RML HDF5 data, labels, and SNR arrays must have matching sample counts")
    return np.asarray(frames, dtype=np.float32), np.asarray(label_indices, dtype=np.int64), np.asarray(raw_snrs, dtype=np.int64), class_names


def _hdf5_label_indices(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if labels.ndim > 1 and labels.shape[-1] > 1:
        return np.argmax(labels, axis=-1).astype(np.int64), [str(index) for index in range(labels.shape[-1])]
    values = labels.reshape(-1)
    names = sorted({str(value.decode() if isinstance(value, bytes) else value) for value in values})
    mapping = {name: index for index, name in enumerate(names)}
    return np.asarray([mapping[str(value.decode() if isinstance(value, bytes) else value)] for value in values], dtype=np.int64), names


def _read_class_names(path: Path, count: int) -> list[str] | None:
    """Read the official sibling classes.txt when an HDF5 file lacks labels."""

    candidates = (path.with_name("classes.txt"), path.parent / "classes.txt")
    for candidate in candidates:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8")
        if "classes" not in text or "[" not in text:
            continue
        values = text.split("[", 1)[1].split("]", 1)[0]
        names = [item.strip().strip("'\"") for item in values.split(",") if item.strip()]
        if len(names) == count:
            return names
    return None


def train_modulation(
    data_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    max_samples: int | None = None,
    patience: int = 7,
    seed: int = 7,
    device: str | None = None,
    sample_length: int | None = 128,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, and save the modulation CNN and its metrics."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from radiofry.models.modulation_cnn import ModulationCNN

    torch.manual_seed(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    frames, targets, snrs, labels = load_rml_dataset(data_path, max_samples=max_samples, seed=seed)
    frames = _canonicalize_frames(frames, sample_length) if sample_length is not None else frames
    stratify_key = np.array([f"{target}:{snr}" for target, snr in zip(targets, snrs)])
    if max_samples is not None and Path(data_path).suffix.lower() not in {".h5", ".hdf5"}:
        if max_samples < len(labels) * 3:
            raise ValueError("max_samples is too small to represent every class")
        selected, _ = train_test_split(
            np.arange(len(targets)),
            train_size=min(max_samples, len(targets)),
            random_state=seed,
            stratify=stratify_key,
        )
        frames, targets, snrs = frames[selected], targets[selected], snrs[selected]
        stratify_key = stratify_key[selected]
    train_idx, remainder_idx = train_test_split(np.arange(len(targets)), test_size=0.30, random_state=seed, stratify=stratify_key)
    remainder_key = stratify_key[remainder_idx]
    val_idx, test_idx = train_test_split(remainder_idx, test_size=0.50, random_state=seed, stratify=remainder_key)
    make_loader = lambda indices, shuffle: DataLoader(TensorDataset(torch.from_numpy(frames[indices]), torch.from_numpy(targets[indices])), batch_size=batch_size, shuffle=shuffle, pin_memory=selected_device.type == "cuda")
    train_loader, val_loader = make_loader(train_idx, True), make_loader(val_idx, False)
    model = ModulationCNN(input_channels=frames.shape[1], num_classes=len(labels)).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, stale_epochs = float("inf"), 0
    best_state = None
    completed_epochs = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        training_loss = 0.0
        for batch, target in train_loader:
            batch, target = batch.to(selected_device, non_blocking=True), target.to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(target)
        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        with torch.inference_mode():
            for batch, target in val_loader:
                batch, target = batch.to(selected_device, non_blocking=True), target.to(selected_device, non_blocking=True)
                logits = model(batch)
                validation_loss += float(loss_fn(logits, target)) * len(target)
                validation_correct += int((torch.argmax(logits, dim=1) == target).sum())
        training_loss /= len(train_idx)
        validation_loss /= len(val_idx)
        validation_accuracy = validation_correct / len(val_idx)
        scheduler.step(validation_loss)
        completed_epochs = epoch
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            checkpoint = {"state_dict": best_state, "labels": labels, "input_channels": frames.shape[1], "sample_length": frames.shape[2], "dataset": str(data_path), "seed": seed, "best_epoch": epoch, "best_validation_loss": best_loss}
            temporary_output = output.with_suffix(output.suffix + ".tmp")
            torch.save(checkpoint, temporary_output)
            temporary_output.replace(output)
            best_marker = " | saved best checkpoint"
        else:
            stale_epochs += 1
            best_marker = ""
        if verbose:
            elapsed = time.perf_counter() - training_started
            epoch_time = time.perf_counter() - epoch_started
            eta = (elapsed / epoch) * max(0, epochs - epoch)
            memory = f" | GPU {torch.cuda.memory_allocated(selected_device) / 1024**3:.2f} GiB" if selected_device.type == "cuda" else ""
            print(f"epoch {epoch:02d}/{epochs} | train_loss={training_loss:.4f} | val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.2%} | {epoch_time:.1f}s | ETA {eta:.0f}s{memory}{best_marker}", flush=True)
        if stale_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch, _ in make_loader(test_idx, False):
            predictions.extend(torch.argmax(model(batch.to(selected_device, non_blocking=True)), dim=1).cpu().numpy().tolist())
    predictions = np.asarray(predictions)
    test_targets, test_snrs = targets[test_idx], snrs[test_idx]
    buckets = {"low": test_snrs <= 0, "mid": (test_snrs > 0) & (test_snrs <= 10), "high": test_snrs > 10}
    confusion: dict[str, list[list[int]]] = {}
    for name, mask in buckets.items():
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for actual, predicted in zip(test_targets[mask], predictions[mask]):
            matrix[actual, predicted] += 1
        confusion[name] = matrix.tolist()
    accuracy_by_snr = {str(int(snr)): float(np.mean(predictions[test_snrs == snr] == test_targets[test_snrs == snr])) for snr in sorted(set(test_snrs))}
    metrics_path = output.with_name(f"{output.stem}_metrics.json")
    metrics = {
        "samples": int(len(targets)), "classes": labels, "epochs_completed": completed_epochs, "device": str(selected_device), "dataset": str(data_path), "sample_length": int(frames.shape[2]), "seed": seed, "best_validation_loss": best_loss,
        "test_accuracy": float(np.mean(predictions == test_targets)),
        "accuracy_by_snr": accuracy_by_snr, "confusion_matrices": confusion,
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    from .evaluation_plots import generate_evaluation_plots
    report_dir = output.parent.parent / "reports" if output.parent.name == "models_saved" else output.parent / "reports"
    generate_evaluation_plots(metrics_path, report_dir)
    return metrics


if __name__ == "__main__":
    main()
