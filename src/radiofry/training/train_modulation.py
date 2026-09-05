"""Train and evaluate the modulation CNN on RML2016.10a."""

from __future__ import annotations

import json
import pickle
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
    args = parser.parse_args()
    metrics = train_modulation(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size, max_samples=args.max_samples, patience=args.patience, device=args.device)
    print(json.dumps(metrics, indent=2))


def load_rml_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the documented RML dictionary into arrays without importing Torch."""

    with Path(path).open("rb") as handle:
        dataset = pickle.load(handle, encoding="latin1")
    labels = sorted({modulation for modulation, _ in dataset})
    label_index = {label: index for index, label in enumerate(labels)}
    frames, targets, snrs = [], [], []
    for (modulation, snr), values in dataset.items():
        frames.append(np.asarray(values, dtype=np.float32))
        targets.extend([label_index[modulation]] * len(values))
        snrs.extend([int(snr)] * len(values))
    return np.concatenate(frames), np.asarray(targets, dtype=np.int64), np.asarray(snrs, dtype=np.int64), labels


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
    frames, targets, snrs, labels = load_rml_dataset(data_path)
    if max_samples is not None:
        if max_samples < len(labels) * 3:
            raise ValueError("max_samples is too small to represent every class")
        generator = np.random.default_rng(seed)
        selected = generator.choice(len(targets), size=min(max_samples, len(targets)), replace=False)
        frames, targets, snrs = frames[selected], targets[selected], snrs[selected]
    stratify_key = np.array([f"{target}:{snr}" for target, snr in zip(targets, snrs)])
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
    for _ in range(epochs):
        model.train()
        for batch, target in train_loader:
            batch, target = batch.to(selected_device, non_blocking=True), target.to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(batch), target).backward()
            optimizer.step()
        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for batch, target in val_loader:
                batch, target = batch.to(selected_device, non_blocking=True), target.to(selected_device, non_blocking=True)
                validation_loss += float(loss_fn(model(batch), target)) * len(target)
        validation_loss /= len(val_idx)
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "input_channels": frames.shape[1]}, output)
    metrics = {
        "samples": int(len(targets)), "classes": labels, "epochs_completed": epochs - stale_epochs, "device": str(selected_device),
        "test_accuracy": float(np.mean(predictions == test_targets)),
        "accuracy_by_snr": accuracy_by_snr, "confusion_matrices": confusion,
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
    }
    output.with_name("modulation_cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    from .evaluation_plots import generate_evaluation_plots
    generate_evaluation_plots(output.with_name("modulation_cnn_metrics.json"), "reports")
    return metrics


if __name__ == "__main__":
    main()
