"""Evaluate the classical modulation-family estimator on an RML corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from radiofry.dsp.cyclostationary import CLASSICAL_THRESHOLDS, estimate_modulation_family
from radiofry.training.train_modulation import load_rml_dataset


CLASS_TO_FAMILY = {
    "BPSK": "PSK-like",
    "QPSK": "PSK-like",
    "8PSK": "PSK-like",
    "CPFSK": "FSK-like",
    "GFSK": "FSK-like",
    "QAM16": "QAM-like",
    "QAM64": "QAM-like",
    "PAM4": "QAM-like",
    "AM-DSB": "analog-like",
    "AM-SSB": "analog-like",
    "WBFM": "analog-like",
}


def evaluate_classical_samples(
    samples: Iterable[np.ndarray],
    labels: Iterable[str],
    snrs: Iterable[float],
) -> dict[str, object]:
    labels = list(labels)
    snrs = list(snrs)
    samples = list(samples)
    if not len(samples) == len(labels) == len(snrs):
        raise ValueError("samples, labels, and snrs must have equal lengths")
    expected = [CLASS_TO_FAMILY.get(label, "unknown") for label in labels]
    predicted = [estimate_modulation_family(sample).family for sample in samples]
    classes = sorted(set(expected) | set(predicted))
    confusion = {actual: {candidate: 0 for candidate in classes} for actual in classes}
    for actual, candidate in zip(expected, predicted):
        confusion[actual][candidate] += 1
    by_snr: dict[str, dict[str, float | int]] = {}
    for snr in sorted(set(snrs)):
        indices = [index for index, value in enumerate(snrs) if value == snr]
        by_snr[str(snr)] = {
            "samples": len(indices),
            "accuracy": float(np.mean([expected[index] == predicted[index] for index in indices])),
        }
    return {
        "samples": len(samples),
        "classes": classes,
        "test_accuracy": float(np.mean(np.asarray(expected) == np.asarray(predicted))) if samples else 0.0,
        "accuracy_by_snr": by_snr,
        "confusion_matrix": confusion,
        "thresholds": CLASSICAL_THRESHOLDS,
    }


def evaluate_rml(data_path: str | Path, output_path: str | Path, *, max_samples: int | None = None) -> dict[str, object]:
    frames, targets, snrs, labels = load_rml_dataset(data_path, max_samples=max_samples)
    samples = frames[:, 0].astype(np.complex64) + 1j * frames[:, 1].astype(np.complex64)
    result = evaluate_classical_samples(samples, (labels[int(target)] for target in targets), snrs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate classical modulation-family thresholds on RML data")
    parser.add_argument("--data", default="data/RML2016.10a_dict.pkl")
    parser.add_argument("--output", default="models_saved/classical_modulation_metrics.json")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate_rml(args.data, args.output, max_samples=args.max_samples), indent=2))


if __name__ == "__main__":
    main()