"""Inference helpers for feature-based interleaver and FEC classifiers."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from radiofry.synthetic_gen.features import bit_features
from radiofry.models.artifact_integrity import hash_bytes, hash_sklearn_model, metrics_path


@dataclass(frozen=True)
class BitstreamPrediction:
    label: str
    confidence: float
    available: bool = True
    message: str = ""


def predict_bitstream(bits: np.ndarray, model_path: str | Path) -> BitstreamPrediction:
    """Predict one label from a saved scikit-learn classifier."""

    path = Path(model_path)
    if not path.exists():
        return BitstreamPrediction("unknown", 0.0, False, f"Classifier not found: {path}")
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if values.size == 0:
        return BitstreamPrediction("unknown", 0.0, False, "No bits were available for classification.")
    try:
        with path.open("rb") as handle:
            # This loader trusts self-produced files in models_saved; do not use external pickles.
            payload = pickle.load(handle)
        expected_hash = payload.get("model_sha256") if isinstance(payload, dict) else None
        if isinstance(payload, dict) and "model_bytes" in payload:
            model_bytes = payload["model_bytes"]
            model = pickle.loads(model_bytes)
            actual_hash = hash_bytes(model_bytes)
        else:
            model = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload
            actual_hash = hash_sklearn_model(model)
        metrics_file = metrics_path(path)
        if not expected_hash or expected_hash != actual_hash:
            return BitstreamPrediction("unknown", 0.0, False, "Classifier artifact integrity check failed: model hash is missing or invalid.")
        if not metrics_file.is_file():
            return BitstreamPrediction("unknown", 0.0, False, f"Classifier metrics not found: {metrics_file}")
        import json
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if metrics.get("model_sha256") != actual_hash:
            return BitstreamPrediction("unknown", 0.0, False, "Classifier artifact integrity check failed: metrics do not match model.")
        max_lag = int(metrics.get("max_lag", 32))
        if max_lag < 1:
            return BitstreamPrediction("unknown", 0.0, False, "Classifier metrics contain an invalid autocorrelation window.")
        features = bit_features(values, max_lag=max_lag).reshape(1, -1)
        expected_features = getattr(model, "n_features_in_", features.shape[1])
        if int(expected_features) != features.shape[1]:
            return BitstreamPrediction(
                "unknown",
                0.0,
                False,
                f"Classifier feature mismatch: expected {expected_features}, got {features.shape[1]}",
            )
        label = str(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else np.array([1.0])
        return BitstreamPrediction(label, float(np.max(probabilities)))
    except (OSError, pickle.PickleError, ValueError, AttributeError, IndexError) as error:
        return BitstreamPrediction("unknown", 0.0, False, f"Classifier inference failed: {error}")
