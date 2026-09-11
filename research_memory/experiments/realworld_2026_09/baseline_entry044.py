"""Zero-training baseline: the frozen checkpoint a7b02533a7c7129c on real held-out data.

NO training, no fine-tuning, no threshold fitting. The TEST split only. The frozen
production inference configuration (128-sample frames, 4 windows, mean-softmax) is used
exactly as shipped.

Three of the seven dataset classes have an exact RadioFry counterpart. The other four are
reported as OUT OF LABEL SPACE rather than as errors: an 8-class digital model has no
output for OFDM or NBFM, so counting them wrong would understate it as badly as hiding
them would overstate it.
"""
import collections
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from radiofry.datasets.realworld_multipath import load_subset  # noqa: E402
from radiofry.models.artifact_integrity import hash_torch_state_dict  # noqa: E402
from radiofry.models.modulation_cnn import ModulationCNN  # noqa: E402
from radiofry.models.signal_features import add_signal_features  # noqa: E402
from radiofry import pipeline  # noqa: E402

ARCHIVE = (r"C:\Users\Kaustubh Bhoir\Downloads"
           r"\Real-World IQ Dataset for Automatic Radio Modulati.zip")
FRAME, WINDOWS = 128, 4
PER_CLASS = 2_000

payload = torch.load(pipeline.DEFAULT_MODULATION_MODEL, map_location="cpu",
                     weights_only=False)
sha = hash_torch_state_dict(payload["state_dict"])
labels = list(payload["labels"])
model = ModulationCNN(int(payload["input_channels"]), len(labels))
model.load_state_dict(payload["state_dict"])
model.eval()
print(f"checkpoint {pipeline.DEFAULT_MODULATION_MODEL}")
print(f"  sha256 {sha[:16]}  labels {labels}")
print(f"  frozen config: {payload['sample_length']}-sample frames, {WINDOWS} windows, "
      f"{payload['features']}\n")


def windows_of(frame: np.ndarray) -> np.ndarray:
    """The production framing, applied to a 1024-sample capture."""
    starts = sorted({int(s) for s in np.linspace(0, frame.size - FRAME, WINDOWS)})
    out = []
    for start in starts:
        block = frame[start:start + FRAME]
        values = np.stack([block.real, block.imag]).astype(np.float32)
        power = np.sqrt(np.mean(values ** 2))
        out.append(add_signal_features(values / power if power > 0 else values,
                                       include_engineered=True))
    return np.stack(out)


def predict(frames: np.ndarray, batch: int = 256):
    """Batched mean-softmax over the 4 production windows per frame."""
    results = []
    for start in range(0, frames.shape[0], batch):
        chunk = frames[start:start + batch]
        stacked = np.concatenate([windows_of(f) for f in chunk])
        with torch.inference_mode():
            logits = model(torch.from_numpy(stacked))
            probabilities = torch.softmax(logits, dim=1).numpy()
        probabilities = probabilities.reshape(len(chunk), -1, len(labels)).mean(axis=1)
        for row in probabilities:
            best = int(np.argmax(row))
            results.append((labels[best], float(row[best])))
    return results


print("=" * 92)
print("A. MAPPABLE CLASSES (BPSK, QPSK, QAM16) - real test split, zero training")
print("=" * 92)
subset = load_subset(ARCHIVE, "test", classes=("BPSK", "QPSK", "QAM"),
                     limit=PER_CLASS * 3, seed=7)
print(f"  {len(subset)} frames  "
      f"{dict(zip(*[list(x) for x in np.unique(subset.modulation, return_counts=True)]))}")

predictions = predict(subset.iq)
truth = subset.radiofry_label
correct = [p == t for (p, _), t in zip(predictions, truth)]
print(f"\n  overall top-1: {sum(correct)}/{len(correct)} = {np.mean(correct):.2%}")

print(f"\n  {'true':7s} {'n':>6s} {'top-1':>8s}   most common predictions")
for name in ("BPSK", "QPSK", "QAM16"):
    mask = truth == name
    if not mask.any():
        continue
    got = [predictions[i][0] for i in np.flatnonzero(mask)]
    hits = sum(1 for g in got if g == name)
    print(f"  {name:7s} {mask.sum():>6d} {hits/mask.sum():>8.1%}   "
          f"{collections.Counter(got).most_common(4)}")

print("\n  by channel condition:")
for value, tag in ((0, "clean"), (1, "multipath")):
    mask = subset.channel == value
    if mask.any():
        hits = [correct[i] for i in np.flatnonzero(mask)]
        print(f"    {tag:10s} {np.mean(hits):6.1%}  (n={mask.sum()})")

print("\n  by SNR:")
for snr in sorted(set(subset.snr_db.tolist())):
    mask = subset.snr_db == snr
    hits = [correct[i] for i in np.flatnonzero(mask)]
    print(f"    {snr:>2d} dB     {np.mean(hits):6.1%}  (n={mask.sum()})")

confidence = np.array([c for _, c in predictions])
wrong = confidence[~np.array(correct)]
right = confidence[np.array(correct)]
print("\n  calibration on real data:")
print(f"    correct  : median {np.median(right):.3f}" if right.size else "    correct  : none")
print(f"    incorrect: median {np.median(wrong):.3f} max {wrong.max():.3f}" if wrong.size
      else "    incorrect: none")
for gate in (0.9, 0.99):
    print(f"    wrong at confidence >= {gate}: {int((wrong >= gate).sum())} "
          f"of {wrong.size}")

print("\n" + "=" * 92)
print("B. OUT OF LABEL SPACE (GMSK, OFDM, NBFM, WBFM) - not errors, no counterpart")
print("=" * 92)
UNMAPPED_CLASSES = ("GMSK", "OFDM", "NBFM", "WBFM")
other = load_subset(ARCHIVE, "test", classes=UNMAPPED_CLASSES,
                    limit=PER_CLASS * 2, seed=7)
other_predictions = predict(other.iq)
print(f"  {len(other)} frames with no RadioFry counterpart")
print(f"\n  {'dataset class':14s} {'n':>6s}   what the 8-class model emits")
for name in UNMAPPED_CLASSES:
    mask = other.modulation == name
    if not mask.any():
        continue
    got = [other_predictions[i][0] for i in np.flatnonzero(mask)]
    print(f"  {name:14s} {mask.sum():>6d}   {collections.Counter(got).most_common(3)}")
other_confidence = np.array([c for _, c in other_predictions])
print(f"\n  confidence on out-of-label-space frames: median "
      f"{np.median(other_confidence):.3f}, >=0.9: "
      f"{int((other_confidence >= 0.9).sum())} of {other_confidence.size}")
print("  (a model with no 'none of these' output must pick something; these are not "
      "scored as errors)")
