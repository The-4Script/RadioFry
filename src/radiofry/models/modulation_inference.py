"""Checkpoint loading and inference for modulation classification."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radiofry.contracts import UnifiedSignalContainer


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
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        labels = list(payload["labels"])
        model = ModulationCNN(int(payload.get("input_channels", 2)), len(labels))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        inputs = torch.from_numpy(_fixed_iq(signal, int(payload.get("sample_length", 128)))).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(inputs), dim=1)[0].numpy()
        indices = np.argsort(probabilities)[::-1][:top_k]
        predictions = tuple((labels[int(index)], float(probabilities[index])) for index in indices)
        return ModulationPrediction(predictions[0][0], predictions[0][1], predictions)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        return ModulationPrediction("Unclassified", 0.0, (), False, f"CNN inference unavailable: {error}")
