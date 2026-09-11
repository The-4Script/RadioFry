"""Characterise the near-duplicate structure found by L2.

L2 measured 0.94-0.98 normalised correlation between test-split and train-split frames of
the same configuration, against a chance floor of 1/sqrt(1024) = 0.031. Before drawing any
conclusion from that, establish what it actually is:

1. Is it also present WITHIN a single split? (redundancy, not just cross-split leakage)
2. Does it survive across configurations - different SNR, different channel, different
   modulation? That separates "the transmitter repeated a fixed bit pattern" from
   "the splits overlap".
3. At what lag does the match occur?
4. How many genuinely distinct waveforms does the dataset contain?

A plausible mechanism is a GNU Radio source vector with repeat enabled, which cycles a
fixed symbol sequence forever. Every recording would then consist of the same few seconds
of content over and over, and a random frame-level split cannot separate anything.

READ ONLY.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\Kaustubh Bhoir\Documents\RadioFry"
            r"\Real-World IQ Dataset for Automatic Radio Modulati\dataset")


def load(split, *, limit, seed, classes=None, chan=None, snr=None):
    with h5py.File(ROOT / f"subset_{split}.h5", "r") as handle:
        mapping = json.loads(handle.attrs["mod2id_json"])
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
        return iq / (np.linalg.norm(iq, axis=1, keepdims=True) + 1e-12)


def peak_correlation(a: np.ndarray, b: np.ndarray, size: int = 2048) -> np.ndarray:
    """Max over all lags of |cross-correlation|, for every row of `a` against all of `b`."""
    b_spectrum = np.fft.fft(b, n=size, axis=1).conj()
    out = np.empty(len(a))
    for i, row in enumerate(a):
        spectrum = np.fft.fft(row, n=size)
        out[i] = np.abs(np.fft.ifft(spectrum[None, :] * b_spectrum, axis=1)).max()
    return out


CHANCE = 1 / np.sqrt(1024)
print(f"chance level for unrelated 1024-sample frames: {CHANCE:.4f}\n")

# --- 1. within-split redundancy --------------------------------------------------------------

print("=" * 96)
print("1. IS THE DUPLICATION ALSO PRESENT WITHIN A SINGLE SPLIT?")
print("=" * 96)
print(f"  {'config':26s} {'median off-diag':>16s} {'95th pct':>10s} {'max':>8s} "
      f"{'>0.9 partners/frame':>21s}")
for name, chan, snr in (("BPSK", 0, 30), ("QPSK", 0, 30), ("QAM", 0, 30),
                        ("GMSK", 0, 30), ("OFDM", 0, 30), ("WBFM", 0, 30)):
    frames = load("test", limit=700, seed=41, classes=[name], chan=chan, snr=snr)
    gram = np.abs(frames @ frames.conj().T)
    np.fill_diagonal(gram, 0.0)
    partners = (gram > 0.9).sum(axis=1)
    print(f"  {f'{name} chan{chan} {snr}dB':26s} {np.median(gram):>16.4f} "
          f"{np.percentile(gram, 95):>10.4f} {gram.max():>8.4f} "
          f"{partners.mean():>21.2f}")

# --- 2. does it cross configuration boundaries? -----------------------------------------------

print("\n" + "=" * 96)
print("2. DOES THE MATCH SURVIVE ACROSS CONFIGURATIONS?")
print("=" * 96)
print("  If a fixed transmit pattern was repeated, frames match across SNR and channel too.\n")
reference = load("test", limit=80, seed=51, classes=["BPSK"], chan=0, snr=30)
comparisons = [
    ("same config, train split", dict(split="train", classes=["BPSK"], chan=0, snr=30)),
    ("same mod, SNR 20 not 30", dict(split="train", classes=["BPSK"], chan=0, snr=20)),
    ("same mod, multipath",     dict(split="train", classes=["BPSK"], chan=1, snr=30)),
    ("different mod (QPSK)",    dict(split="train", classes=["QPSK"], chan=0, snr=30)),
    ("different mod (GMSK)",    dict(split="train", classes=["GMSK"], chan=0, snr=30)),
]
print(f"  {'comparison':28s} {'median peak':>12s} {'max peak':>10s}")
for label, kwargs in comparisons:
    split = kwargs.pop("split")
    bank = load(split, limit=1_200, seed=52, **kwargs)
    peaks = peak_correlation(reference, bank)
    print(f"  {label:28s} {np.median(peaks):>12.4f} {peaks.max():>10.4f}")

# --- 3. at what lag? ---------------------------------------------------------------------------

print("\n" + "=" * 96)
print("3. AT WHAT LAG DOES THE BEST MATCH OCCUR?")
print("=" * 96)
probe = load("test", limit=40, seed=61, classes=["BPSK"], chan=0, snr=30)
bank = load("train", limit=1_500, seed=62, classes=["BPSK"], chan=0, snr=30)
size = 2048
bank_spectrum = np.fft.fft(bank, n=size, axis=1).conj()
lags, values = [], []
for row in probe:
    correlation = np.abs(np.fft.ifft(np.fft.fft(row, n=size)[None, :] * bank_spectrum, axis=1))
    flat = int(np.argmax(correlation))
    lags.append(flat % size)
    values.append(float(correlation.max()))
lags = np.array(lags)
print(f"  best-match correlation: median {np.median(values):.4f}")
print(f"  best-match lag (samples, mod 2048): "
      f"min {lags.min()} max {lags.max()} distinct {len(set(lags.tolist()))} of {len(lags)}")
print(f"  lag 0 (aligned frame boundaries): {(lags == 0).sum()} of {len(lags)}")
print(f"  first 15 lags: {lags[:15].tolist()}")

# --- 4. how many distinct waveforms? -----------------------------------------------------------

print("\n" + "=" * 96)
print("4. HOW MANY DISTINCT WAVEFORMS DOES ONE CONFIGURATION ACTUALLY CONTAIN?")
print("=" * 96)
print("  Greedy clustering at |corr| > 0.9 over pooled train+val+test frames of one config.\n")
print(f"  {'config':24s} {'frames':>8s} {'clusters':>9s} {'largest':>8s} {'singletons':>11s}")
for name in ("BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM"):
    pooled = np.concatenate([
        load(split, limit=300, seed=70 + i, classes=[name], chan=0, snr=30)
        for i, split in enumerate(("train", "val", "test"))])
    gram = np.abs(pooled @ pooled.conj().T)
    unassigned = set(range(len(pooled)))
    clusters = []
    while unassigned:
        seed_index = unassigned.pop()
        members = {seed_index} | {j for j in unassigned if gram[seed_index, j] > 0.9}
        unassigned -= members
        clusters.append(members)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"  {f'{name} clean 30dB':24s} {len(pooled):>8d} {len(clusters):>9d} "
          f"{sizes[0]:>8d} {sum(1 for s in sizes if s == 1):>11d}")
