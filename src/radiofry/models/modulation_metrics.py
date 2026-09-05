"""Helpers for presenting measured modulation-model validation metrics."""

from typing import Mapping

import numpy as np


def expected_accuracy_at_snr(metrics: Mapping[str, object], snr_db: float) -> float:
    """Interpolate the recorded CNN accuracy curve at an estimated SNR."""

    accuracy_by_snr = metrics.get("accuracy_by_snr", {})
    if not isinstance(accuracy_by_snr, Mapping) or not accuracy_by_snr:
        raise ValueError("metrics must contain a non-empty accuracy_by_snr mapping")
    points = sorted((float(snr), float(accuracy)) for snr, accuracy in accuracy_by_snr.items())
    return float(np.interp(snr_db, [point[0] for point in points], [point[1] for point in points]))