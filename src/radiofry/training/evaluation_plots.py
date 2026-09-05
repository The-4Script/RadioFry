"""Persist visual evaluation artifacts from modulation metrics."""

import json
from pathlib import Path


def generate_evaluation_plots(metrics_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Create the required SNR curve and bucketed confusion-matrix images."""

    import matplotlib.pyplot as plt
    import numpy as np

    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    snrs = sorted((int(snr), accuracy) for snr, accuracy in metrics["accuracy_by_snr"].items())
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([snr for snr, _ in snrs], [accuracy for _, accuracy in snrs], marker="o")
    axis.set(xlabel="SNR (dB)", ylabel="Accuracy", title="Modulation accuracy by SNR")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.25)
    curve = output / "accuracy_vs_snr.png"
    figure.tight_layout(); figure.savefig(curve, dpi=150); plt.close(figure)
    generated.append(curve)
    labels = metrics["classes"]
    for bucket, values in metrics["confusion_matrices"].items():
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(np.asarray(values), cmap="Blues")
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="Actual", title=f"{bucket.title()} SNR confusion matrix")
        figure.colorbar(image, ax=axis, fraction=0.046)
        figure.autofmt_xdate(rotation=45)
        matrix_path = output / f"confusion_{bucket}.png"
        figure.tight_layout(); figure.savefig(matrix_path, dpi=150); plt.close(figure)
        generated.append(matrix_path)
    return generated
