"""Leakage battery for the real-world dataset.

The paper states the split protocol explicitly: continuous multi-minute SDR recordings were
segmented into 1024-sample frames, and the frames were then split train/val/test by
STRATIFIED RANDOM SAMPLING. That means frames from one physical recording are scattered
across all three splits. The structural pass already confirmed the consequence: all 84
(modulation, channel, SNR) configurations appear in all three splits.

That is recording-level leakage by construction. What is left to establish is whether it is
EXPLOITABLE, because a split can be nominally contaminated and still benign.

L1  exact duplicate frames across splits - definitive if positive
L2  near-duplicate waveform overlap between test and train
L3  the decisive test: can pure HARDWARE-ARTIFACT features, which carry no modulation
    information of their own, predict the modulation label across splits? DC offset is
    local-oscillator leakage, gain imbalance and quadrature error are receiver front-end
    imperfections. All four are zero for every one of these modulations in theory. If they
    identify the class, the benchmark is partly measuring which recording a frame came from.
L4  are the two channel labels separable at all?

READ ONLY.
"""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\Kaustubh Bhoir\Documents\RadioFry"
            r"\Real-World IQ Dataset for Automatic Radio Modulati\dataset")
SPLITS = ("train", "val", "test")
CLASSES = ["BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM"]


# --- L1: exact duplicates ------------------------------------------------------------------

print("=" * 96)
print("L1. EXACT DUPLICATE FRAMES ACROSS SPLITS")
print("=" * 96)
started = time.perf_counter()
digests: dict[str, dict] = {}
for split in SPLITS:
    with h5py.File(ROOT / f"subset_{split}.h5", "r") as handle:
        data = handle["X"]
        n = data.shape[0]
        table: dict[bytes, int] = {}
        for start in range(0, n, 20_000):
            block = data[start:start + 20_000]
            raw = np.ascontiguousarray(block)
            for offset in range(raw.shape[0]):
                table[hashlib.blake2b(raw[offset].tobytes(),
                                      digest_size=16).digest()] = start + offset
        digests[split] = table
        print(f"  {split:6s} {n:>7,} frames -> {len(table):>7,} distinct "
              f"({n - len(table):,} internal duplicates)")
print(f"  [{time.perf_counter() - started:.0f}s]")

for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
    shared = set(digests[a]) & set(digests[b])
    print(f"  {a} ∩ {b}: {len(shared):,} identical frames")
print()


# --- L2: near-duplicate waveform overlap ----------------------------------------------------

def load(split, *, limit, seed, classes=None, chan=None, snr=None, fields=False):
    with h5py.File(ROOT / f"subset_{split}.h5", "r") as handle:
        mapping = json.loads(handle.attrs["mod2id_json"])
        id_to_name = {int(v): k for k, v in mapping.items()}
        mods, chans, snrs = handle["y_mod"][:], handle["y_chan"][:], handle["y_snr"][:]
        keep = np.ones(mods.size, bool)
        if classes is not None:
            keep &= np.isin(mods, [mapping[c] for c in classes])
        if chan is not None:
            keep &= chans == chan
        if snr is not None:
            keep &= snrs == snr
        pool = np.flatnonzero(keep)
        index = np.sort(np.random.default_rng(seed).choice(
            pool, size=min(limit, pool.size), replace=False))
        raw = handle["X"][index].astype(np.float32)
        iq = (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)
        if not fields:
            return iq
        return (iq, np.array([id_to_name[int(v)] for v in mods[index]]),
                chans[index], snrs[index])


print("=" * 96)
print("L2. NEAR-DUPLICATE WAVEFORM OVERLAP (test frames vs train frames, same configuration)")
print("=" * 96)
print("  Normalised |<a,b>| at zero lag, and the peak of the full cross-correlation.")
print("  Unrelated 1024-sample frames sit at about 1/sqrt(1024) = 0.031.\n")
print(f"  {'config':22s} {'max zero-lag':>13s} {'max any-lag':>12s} {'chance':>8s}")
for name in ("BPSK", "QAM", "GMSK"):
    probe = load("test", limit=200, seed=11, classes=[name], chan=0, snr=30)
    bank = load("train", limit=3_000, seed=12, classes=[name], chan=0, snr=30)
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)
    bank_normalised = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    zero_lag = np.abs(probe @ bank_normalised.conj().T).max()

    # any-lag, via FFT, on a smaller sub-block (cost is quadratic in the bank size)
    size = 2048
    probe_f = np.fft.fft(probe[:60], n=size, axis=1)
    bank_f = np.fft.fft(bank_normalised[:800], n=size, axis=1)
    best = 0.0
    for row in probe_f:
        correlation = np.abs(np.fft.ifft(row[None, :] * bank_f.conj(), axis=1))
        best = max(best, float(correlation.max()))
    print(f"  {name + ' clean 30dB':22s} {zero_lag:>13.4f} {best:>12.4f} {0.031:>8.3f}")
print()


# --- L3: hardware-artifact probe ------------------------------------------------------------

def nuisance_features(iq: np.ndarray) -> np.ndarray:
    """Four receiver-artifact scalars. Every one is zero for an ideal signal of any of
    these modulations, so none of them carries legitimate modulation information."""
    scale = np.sqrt(np.mean(np.abs(iq) ** 2, axis=1, keepdims=True)) + 1e-12
    unit = iq / scale
    inphase, quadrature = unit.real, unit.imag
    dc_i, dc_q = inphase.mean(axis=1), quadrature.mean(axis=1)
    gain = inphase.std(axis=1) / (quadrature.std(axis=1) + 1e-12)
    quadrature_error = np.mean(inphase * quadrature, axis=1)
    return np.stack([dc_i, dc_q, np.log(gain + 1e-12), quadrature_error], axis=1)


def logistic(train_x, train_y, test_x, test_y, classes, *, epochs=400, seed=0):
    """Multinomial logistic regression, plain gradient descent - no sklearn dependency."""
    rng = np.random.default_rng(seed)
    mean, std = train_x.mean(0), train_x.std(0) + 1e-9
    a, b = (train_x - mean) / std, (test_x - mean) / std
    k = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    ya = np.array([index[v] for v in train_y])
    yb = np.array([index[v] for v in test_y])
    weights = rng.normal(0, 0.01, (a.shape[1], k))
    bias = np.zeros(k)
    onehot = np.eye(k)[ya]
    for _ in range(epochs):
        logits = a @ weights + bias
        logits -= logits.max(1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(1, keepdims=True)
        error = (probabilities - onehot) / len(a)
        weights -= 0.5 * (a.T @ error) + 1e-4 * weights
        bias -= 0.5 * error.sum(0)
    return float((np.argmax(b @ weights + bias, 1) == yb).mean())


print("=" * 96)
print("L3. CAN RECEIVER-ARTIFACT FEATURES ALONE PREDICT THE MODULATION ACROSS SPLITS?")
print("=" * 96)
train_iq, train_mod, train_chan, train_snr = load(
    "train", limit=28_000, seed=21, fields=True)
test_iq, test_mod, test_chan, test_snr = load(
    "test", limit=14_000, seed=22, fields=True)

artifacts_train = nuisance_features(train_iq)
artifacts_test = nuisance_features(test_iq)
power_train = np.log(np.mean(np.abs(train_iq) ** 2, axis=1) + 1e-12)[:, None]
power_test = np.log(np.mean(np.abs(test_iq) ** 2, axis=1) + 1e-12)[:, None]

print(f"  trained on {len(train_iq):,} train-split frames, scored on {len(test_iq):,} "
      f"test-split frames.  chance = {1/7:.1%}\n")

score = logistic(artifacts_train, train_mod, artifacts_test, test_mod, CLASSES)
print(f"  4 hardware-artifact features (DC offset I/Q, gain imbalance, quadrature error)")
print(f"    -> 7-class modulation accuracy: {score:.1%}")

score_power = logistic(power_train, train_mod, power_test, test_mod, CLASSES)
print(f"  1 feature, log frame power alone")
print(f"    -> 7-class modulation accuracy: {score_power:.1%}")

both = np.hstack([artifacts_train, power_train])
both_test = np.hstack([artifacts_test, power_test])
score_both = logistic(both, train_mod, both_test, test_mod, CLASSES)
print(f"  all 5 together")
print(f"    -> 7-class modulation accuracy: {score_both:.1%}")

config_train = np.array([f"{m}|{c}|{s}" for m, c, s in
                         zip(train_mod, train_chan, train_snr)])
config_test = np.array([f"{m}|{c}|{s}" for m, c, s in
                        zip(test_mod, test_chan, test_snr)])
configs = sorted(set(config_train.tolist()) | set(config_test.tolist()))
score_config = logistic(both, config_train, both_test, config_test, configs, epochs=600)
print(f"\n  same 5 features predicting the 84-way RECORDING CONFIGURATION")
print(f"    -> accuracy: {score_config:.1%}   (chance {1/len(configs):.2%})")
print()


# --- L4: are the channel labels separable? ---------------------------------------------------

print("=" * 96)
print("L4. ARE THE TWO CHANNEL LABELS SEPARABLE?")
print("=" * 96)
print("  Within one modulation and SNR, so only the channel condition differs.\n")
print(f"  {'class':6s} {'artifact feats':>15s} {'PSD shape (64 bins)':>21s}   chance 50%")
for name in ("BPSK", "QPSK", "QAM", "GMSK"):
    parts = {}
    for split, seed in (("train", 31), ("test", 32)):
        iq, mods, chans, snrs = load(split, limit=6_000, seed=seed,
                                     classes=[name], snr=30, fields=True)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq, axis=1), axes=1)) ** 2
        binned = spectrum.reshape(len(iq), 64, -1).mean(axis=2)
        binned = np.log(binned / binned.sum(1, keepdims=True) + 1e-12)
        parts[split] = (nuisance_features(iq), binned,
                        np.array([str(c) for c in chans]))
    artifact_score = logistic(parts["train"][0], parts["train"][2],
                              parts["test"][0], parts["test"][2], ["0", "1"])
    psd_score = logistic(parts["train"][1], parts["train"][2],
                         parts["test"][1], parts["test"][2], ["0", "1"], epochs=800)
    print(f"  {name:6s} {artifact_score:>14.1%} {psd_score:>20.1%}")
