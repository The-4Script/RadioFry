"""Re-measure the duplication with a correct null. The 1/sqrt(1024) floor was wrong.

These frames are narrowband (about 270 kHz occupied in a 2 MHz span) and carry residual
local-oscillator leakage. Both inflate correlation between completely unrelated frames:

* a bandlimited frame has roughly 1024 * BW/fs independent degrees of freedom, not 1024;
* a shared DC / carrier component correlates every frame with every other frame, which is
  almost certainly why WBFM - an FM signal with a strong residual carrier - showed a median
  of 0.60 against unrelated frames of its own class.

The correct null is a PHASE-RANDOMISED SURROGATE: keep each frame's magnitude spectrum
exactly and randomise the phases. The surrogate has the identical power spectral density,
identical bandwidth, identical DC content and identical per-frame power, but its content is
unrelated by construction. Whatever correlation survives against surrogates is the floor.

Also reported with DC removed, which isolates the modulated content from LO leakage.

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
RNG = np.random.default_rng(12345)


def load(split, *, limit, seed, classes=None, chan=None, snr=None, remove_dc=True):
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
    if remove_dc:
        iq = iq - iq.mean(axis=1, keepdims=True)
    return iq / (np.linalg.norm(iq, axis=1, keepdims=True) + 1e-12)


def surrogate(frames: np.ndarray) -> np.ndarray:
    """Same magnitude spectrum, randomised phase: identical PSD, unrelated content."""
    spectrum = np.fft.fft(frames, axis=1)
    phases = RNG.uniform(0, 2 * np.pi, spectrum.shape)
    shuffled = np.abs(spectrum) * np.exp(1j * phases)
    out = np.fft.ifft(shuffled, axis=1).astype(np.complex64)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def peak_correlation(a: np.ndarray, b: np.ndarray, size: int = 2048) -> np.ndarray:
    b_spectrum = np.fft.fft(b, n=size, axis=1).conj()
    out = np.empty(len(a))
    for i, row in enumerate(a):
        out[i] = np.abs(np.fft.ifft(
            np.fft.fft(row, n=size)[None, :] * b_spectrum, axis=1)).max()
    return out


CLASSES = ("BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM")

print("=" * 100)
print("OBSERVED vs PHASE-RANDOMISED-SURROGATE NULL   (DC removed; peak over all lags)")
print("=" * 100)
print("  probe = 120 test-split frames.  bank = 1,500 frames.  clean channel, 30 dB.")
print("  'surrogate' has the same PSD as the real bank but unrelated content.\n")
print(f"  {'class':6s} {'vs TRAIN bank':>14s} {'vs SURROGATE':>13s} {'vs OTHER-MOD':>13s} "
      f"{'excess':>9s}   verdict")

summary = {}
for name in CLASSES:
    probe = load("test", limit=120, seed=81, classes=[name], chan=0, snr=30)
    bank = load("train", limit=1_500, seed=82, classes=[name], chan=0, snr=30)
    others = [c for c in CLASSES if c != name]
    other_bank = load("train", limit=1_500, seed=83, classes=others, chan=0, snr=30)

    observed = float(np.median(peak_correlation(probe, bank)))
    null = float(np.median(peak_correlation(probe, surrogate(bank))))
    cross = float(np.median(peak_correlation(probe, other_bank)))
    excess = observed - null
    verdict = ("DUPLICATION" if excess > 0.25 else
               "elevated" if excess > 0.08 else "no excess over null")
    print(f"  {name:6s} {observed:>14.4f} {null:>13.4f} {cross:>13.4f} "
          f"{excess:>+9.4f}   {verdict}")
    summary[name] = {"observed": observed, "null": null, "cross_mod": cross}

print("\n" + "=" * 100)
print("DISTRIBUTION OF THE MATCH, not just its median   (BPSK / QAM / GMSK, clean 30 dB)")
print("=" * 100)
for name in ("BPSK", "QAM", "GMSK"):
    probe = load("test", limit=300, seed=91, classes=[name], chan=0, snr=30)
    bank = load("train", limit=2_000, seed=92, classes=[name], chan=0, snr=30)
    observed = peak_correlation(probe, bank)
    null = peak_correlation(probe, surrogate(bank))
    print(f"  {name}:")
    for tag, values in (("real bank ", observed), ("surrogate ", null)):
        print(f"    {tag} median {np.median(values):.4f}  90th {np.percentile(values,90):.4f}  "
              f"max {values.max():.4f}  frac>0.9 {np.mean(values>0.9):.1%}")

print("\n" + "=" * 100)
print("CROSS-SPLIT vs WITHIN-SPLIT, with DC removed")
print("=" * 100)
print("  If the splits are contaminated relative to each other, a test frame matches a")
print("  TRAIN frame about as well as it matches another TEST frame. Equal values mean the")
print("  splits are interchangeable - which is what a random frame-level split produces.\n")
print(f"  {'class':6s} {'test vs train':>14s} {'test vs test':>13s} {'surrogate':>11s}")
for name in ("BPSK", "QPSK", "QAM", "GMSK", "OFDM"):
    probe = load("test", limit=120, seed=101, classes=[name], chan=0, snr=30)
    train_bank = load("train", limit=1_500, seed=102, classes=[name], chan=0, snr=30)
    test_bank = load("test", limit=1_500, seed=103, classes=[name], chan=0, snr=30)
    print(f"  {name:6s} {np.median(peak_correlation(probe, train_bank)):>14.4f} "
          f"{np.median(peak_correlation(probe, test_bank)):>13.4f} "
          f"{np.median(peak_correlation(probe, surrogate(train_bank))):>11.4f}")

json.dump(summary, open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stderr, indent=2)
