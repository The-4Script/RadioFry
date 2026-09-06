"""Checkpoint loading and inference for modulation classification."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radiofry.contracts import UnifiedSignalContainer
from radiofry.models.artifact_integrity import hash_torch_state_dict, metrics_path


@dataclass(frozen=True)
class ModulationPrediction:
    label: str
    confidence: float
    top_k: tuple[tuple[str, float], ...]
    available: bool = True
    message: str = ""


def _fixed_iq(signal: UnifiedSignalContainer, length: int) -> np.ndarray:
    if signal.iq.size == 0:
        raise ValueError("cannot classify an empty signal")
    positions = np.linspace(0, signal.iq.size - 1, length)
    source = np.arange(signal.iq.size)
    real = np.interp(positions, source, signal.iq.real)
    imag = np.interp(positions, source, signal.iq.imag)
    values = np.stack([real, imag]).astype(np.float32)
    power = np.sqrt(np.mean(values**2))
    return values / power if power > 0 else values


def predict_modulation(signal: UnifiedSignalContainer, checkpoint_path: str | Path, *, top_k: int = 3) -> ModulationPrediction:
    """Load a saved model lazily and return JSON-friendly top-k predictions."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return ModulationPrediction("Unclassified", 0.0, (), False, f"Checkpoint not found: {checkpoint}")
    try:
        import torch
        from radiofry.models.modulation_cnn import ModulationCNN
        from radiofry.models.signal_features import add_signal_features
        # This loader trusts self-produced files in models_saved; do not use external checkpoints.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_hash = payload.get("model_sha256")
        actual_hash = hash_torch_state_dict(payload["state_dict"])
        metrics_file = metrics_path(checkpoint)
        if not expected_hash or expected_hash != actual_hash:
            return ModulationPrediction("Unclassified", 0.0, (), False, "CNN artifact integrity check failed: checkpoint hash is missing or invalid.")
        if not metrics_file.is_file():
            return ModulationPrediction("Unclassified", 0.0, (), False, f"CNN metrics not found: {metrics_file}")
        import json
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if metrics.get("model_sha256") != actual_hash:
            return ModulationPrediction("Unclassified", 0.0, (), False, "CNN artifact integrity check failed: metrics do not match checkpoint.")
        labels = list(payload["labels"])
        model = ModulationCNN(int(payload.get("input_channels", 2)), len(labels))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        inputs = _fixed_iq(signal, int(payload.get("sample_length", 128)))
        inputs = add_signal_features(inputs, include_engineered=payload.get("features", "iq") == "iqap")
        inputs = torch.from_numpy(inputs).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(inputs), dim=1)[0].numpy()
        indices = np.argsort(probabilities)[::-1][:top_k]
        predictions = tuple((labels[int(index)], float(probabilities[index])) for index in indices)
        return ModulationPrediction(predictions[0][0], predictions[0][1], predictions)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        return ModulationPrediction("Unclassified", 0.0, (), False, f"CNN inference unavailable: {error}")
