"""Pure helpers for modulation evidence presentation."""

import json
from pathlib import Path
from typing import Any

from radiofry.models.modulation_metrics import expected_accuracy_at_snr


# Placeholder for team sign-off before the judged demo; this threshold is intentionally not final.
LOW_SNR_BIAS_WARNING_THRESHOLD_DB = -5.0
LOW_SNR_BIAS_WARNING = "At this estimated SNR, the model's measured low-SNR confusion pattern shows a known class bias. Treat this result as low-confidence regardless of the reported softmax value."


def load_modulation_metrics(metrics_path: str | Path | None = None) -> dict[str, Any] | None:
    path = Path(metrics_path) if metrics_path is not None else Path(__file__).resolve().parents[1] / "models_saved" / "modulation_cnn_metrics.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _expected_cnn_accuracy(snr_db: float | None, metrics_path: str | Path | None = None) -> float | None:
    if snr_db is None:
        return None
    metrics = load_modulation_metrics(metrics_path)
    if metrics is None:
        return None
    try:
        return expected_accuracy_at_snr(metrics, snr_db)
    except (TypeError, ValueError):
        return None


def low_snr_bias_labels(metrics: dict[str, Any]) -> set[str]:
    classes = metrics.get("classes", [])
    matrix = metrics.get("confusion_matrices", {}).get("low", [])
    if not classes or not matrix:
        return set()
    column_totals = [sum(row[index] for row in matrix if index < len(row)) for index in range(len(classes))]
    mean_total = sum(column_totals) / len(column_totals)
    return {str(label) for label, total in zip(classes, column_totals) if total > mean_total}