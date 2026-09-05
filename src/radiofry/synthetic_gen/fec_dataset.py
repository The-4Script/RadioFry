"""Synthetic labeled bitstreams for FEC-scheme classification."""

import csv
from pathlib import Path

import numpy as np


def convolutional_encode(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    state = 0
    encoded = []
    for bit in values:
        state = ((state << 1) | int(bit)) & 0b1111111
        encoded.extend([(state & 0o171).bit_count() & 1, (state & 0o133).bit_count() & 1])
    return np.asarray(encoded, dtype=np.uint8)


def reed_solomon_encode(bits: np.ndarray, nsym: int = 32) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    usable = (values.size // 8) * 8
    payload = np.packbits(values[:usable]).tobytes()
    try:
        from reedsolo import RSCodec
        encoded = RSCodec(nsym).encode(payload)
        return np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))
    except ImportError:
        return np.repeat(values, 2)


def generate_fec_dataset(output_dir: str | Path, *, examples_per_class: int = 100, length: int = 256, seed: int = 7) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "fec_manifest.csv"
    generator = np.random.default_rng(seed)
    labels = ["none", "convolutional", "reed_solomon", "concatenated", "ldpc"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "fec_type", "fec_params"])
        writer.writeheader()
        index = 0
        for label in labels:
            for _ in range(examples_per_class):
                bits = generator.integers(0, 2, length, dtype=np.uint8)
                if label == "convolutional":
                    bits = convolutional_encode(bits)
                elif label == "reed_solomon":
                    bits = reed_solomon_encode(bits)
                elif label == "concatenated":
                    bits = convolutional_encode(reed_solomon_encode(bits))
                elif label == "ldpc":
                    bits = np.repeat(bits, 2)
                filename = f"fec_{index:05d}.npy"
                np.save(output / filename, bits)
                writer.writerow({"filename": filename, "fec_type": label, "fec_params": "default"})
                index += 1
    return manifest
