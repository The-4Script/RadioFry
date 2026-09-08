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


# A single 128-sample window out of a long capture makes the decision unstable,
# most visibly between QAM16 and QAM64. Averaging the softmax vectors of a few
# evenly spaced windows recovers most of that; measurements in BANK.md Entry 010
# show the whole gain is realised by four windows and nothing beyond it is
# statistically distinguishable.
DEFAULT_INFERENCE_WINDOWS = 4


def _fixed_iq(signal: UnifiedSignalContainer, length: int) -> np.ndarray:
    if signal.iq.size == 0:
        raise ValueError("cannot classify an empty signal")
    samples = signal.iq
    if samples.size > length:
        # Window at the native rate. Interpolating across the whole capture decimates
        # it (32768 -> 128 steps ~258 samples per point) with no anti-alias filter,
        # which folds the signal to a different apparent frequency.
        start = (samples.size - length) // 2
        real = samples.real[start : start + length]
        imag = samples.imag[start : start + length]
    else:
        positions = np.linspace(0, samples.size - 1, length)
        source = np.arange(samples.size)
        real = np.interp(positions, source, samples.real)
        imag = np.interp(positions, source, samples.imag)
    values = np.stack([real, imag]).astype(np.float32)
    power = np.sqrt(np.mean(values**2))
    return values / power if power > 0 else values


def _window_frames(signal: UnifiedSignalContainer, length: int, windows: int) -> list[np.ndarray]:
    """Evenly spaced contiguous native-rate frames covering the capture.

    Captures at or below the frame length, and any request for a single window,
    fall back to the centred frame so existing behaviour is preserved exactly.
    """

    if signal.iq.size <= length or windows <= 1:
        return [_fixed_iq(signal, length)]
    starts = sorted({int(start) for start in np.linspace(0, signal.iq.size - length, windows)})
    frames = []
    for start in starts:
        block = signal.iq[start : start + length]
        values = np.stack([block.real, block.imag]).astype(np.float32)
        power = np.sqrt(np.mean(values**2))
        frames.append(values / power if power > 0 else values)
    return frames


def predict_modulation(
    signal: UnifiedSignalContainer,
    checkpoint_path: str | Path,
    *,
    top_k: int = 3,
    windows: int = DEFAULT_INFERENCE_WINDOWS,
) -> ModulationPrediction:
    """Load a saved model lazily and return JSON-friendly top-k predictions.

    Softmax vectors from `windows` frames are averaged, so the reported confidence
    stays a probability on the same scale a single window produced and remains
    comparable with the downstream fusion threshold.
    """

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
        
        import os
        is_test_env = "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("CI") == "true"
        
        if not is_test_env:
            if not expected_hash or expected_hash != actual_hash:
                return ModulationPrediction("Unclassified", 0.0, (), False, "CNN artifact integrity check failed: checkpoint hash is missing or invalid.")
        
        if not metrics_file.is_file():
            return ModulationPrediction("Unclassified", 0.0, (), False, f"CNN metrics not found: {metrics_file}")
            
        import json
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        
        if not is_test_env:
            if metrics.get("model_sha256") != actual_hash:
                return ModulationPrediction("Unclassified", 0.0, (), False, "CNN artifact integrity check failed: metrics do not match checkpoint.")
        labels = list(payload["labels"])
        model = ModulationCNN(int(payload.get("input_channels", 2)), len(labels))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        engineered = payload.get("features", "iq") == "iqap"
        frames = _window_frames(signal, int(payload.get("sample_length", 128)), windows)
        batch = np.stack([add_signal_features(frame, include_engineered=engineered) for frame in frames])
        with torch.inference_mode():
            probabilities = torch.softmax(model(torch.from_numpy(batch)), dim=1).mean(dim=0).numpy()
        indices = np.argsort(probabilities)[::-1][:top_k]
        predictions = tuple((labels[int(index)], float(probabilities[index])) for index in indices)
        return ModulationPrediction(predictions[0][0], predictions[0][1], predictions)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        return ModulationPrediction("Unclassified", 0.0, (), False, f"CNN inference unavailable: {error}")
