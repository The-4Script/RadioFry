"""Generate labeled bitstream examples for classical model training."""

import csv
from pathlib import Path

import numpy as np

from .interleavers import block_interleave, diagonal_interleave, pseudo_random_interleave


def generate_interleaver_dataset(output_dir: str | Path, *, examples_per_class: int = 100, length: int = 128, seed: int = 7) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    generator = np.random.default_rng(seed)
    labels = ["none", "block", "diagonal", "pseudo_random"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "interleaver_type", "parameter"])
        writer.writeheader()
        index = 0
        for label in labels:
            for _ in range(examples_per_class):
                frame = np.array([int(bit) for bit in "01111110000110111011001011001001"], dtype=np.uint8)
                bits = np.resize(frame, length).copy()
                noise = generator.random(length) < 0.05
                bits[noise] ^= 1
                parameter = ""
                if label == "block":
                    bits = block_interleave(bits, 8, length // 8); parameter = "8"
                elif label == "diagonal":
                    bits = diagonal_interleave(bits, 8); parameter = "8"
                elif label == "pseudo_random":
                    bits = pseudo_random_interleave(bits, index); parameter = str(index)
                filename = f"bits_{index:05d}.npy"
                np.save(output / filename, bits)
                writer.writerow({"filename": filename, "interleaver_type": label, "parameter": parameter})
                index += 1
    return manifest
