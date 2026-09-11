"""Does duplicated content cross RECORDING boundaries?

This single question decides the training protocol.

Each of the 84 (modulation, channel, SNR) combinations is a separate .dat recording, per the
paper's own file-naming convention. The released splits were drawn randomly from frames
pooled across those recordings, so train and test share every recording - which is why QAM
shows 40% near-duplicates across the split.

If duplicated content does NOT cross recording boundaries, then holding out entire SNR
levels yields a genuinely capture-disjoint split, and its accuracy is an honest
generalisation number. If content DOES repeat across recordings - a fixed transmit pattern
replayed for every capture - then no split of this dataset is clean and that has to be
stated rather than worked around.

Measured with DC removed and against the phase-randomised surrogate null established in
inv6. Self-matches are excluded by construction (disjoint index pools).

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
RNG = np.random.default_rng(2026)
CLASSES = ("BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM")


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
    iq = iq - iq.mean(axis=1, keepdims=True)
    return iq / (np.linalg.norm(iq, axis=1, keepdims=True) + 1e-12)


def surrogate(frames):
    spectrum = np.fft.fft(frames, axis=1)
    out = np.fft.ifft(np.abs(spectrum) *
                      np.exp(1j * RNG.uniform(0, 2 * np.pi, spectrum.shape)),
                      axis=1).astype(np.complex64)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def peaks(a, b, size=2048):
    b_spectrum = np.fft.fft(b, n=size, axis=1).conj()
    return np.array([np.abs(np.fft.ifft(
        np.fft.fft(row, n=size)[None, :] * b_spectrum, axis=1)).max() for row in a])


print("=" * 102)
print("DOES NEAR-DUPLICATE CONTENT CROSS RECORDING BOUNDARIES?")
print("=" * 102)
print("  probe: 150 TEST frames, clean, 30 dB. Each bank is 1,800 TRAIN frames.")
print("  'frac>0.9' is the fraction of probe frames having a near-identical partner.\n")
print(f"  {'class':6s} {'bank':28s} {'median':>8s} {'frac>0.9':>9s} {'null med':>9s} "
      f"{'excess':>8s}")

results = {}
for name in CLASSES:
    probe = load("test", limit=150, seed=201, classes=[name], chan=0, snr=30)
    banks = [
        ("SAME recording (clean,30dB)", dict(chan=0, snr=30)),
        ("other SNR   (clean,20dB)",    dict(chan=0, snr=20)),
        ("other SNR   (clean,24dB)",    dict(chan=0, snr=24)),
        ("other chan  (multipath,30dB)", dict(chan=1, snr=30)),
    ]
    row = {}
    for label, kwargs in banks:
        bank = load("train", limit=1_800, seed=202, classes=[name], **kwargs)
        observed = peaks(probe, bank)
        null = peaks(probe, surrogate(bank))
        row[label.split("(")[0].strip()] = {
            "median": float(np.median(observed)),
            "frac_above_0p9": float(np.mean(observed > 0.9)),
            "null_median": float(np.median(null)),
        }
        print(f"  {name:6s} {label:28s} {np.median(observed):>8.4f} "
              f"{np.mean(observed > 0.9):>8.1%} {np.median(null):>9.4f} "
              f"{np.median(observed) - np.median(null):>+8.4f}")
    results[name] = row
    print()

print("=" * 102)
print("SUMMARY: near-duplicate rate, same recording vs different recording")
print("=" * 102)
print(f"  {'class':6s} {'same recording':>16s} {'different SNR':>15s} "
      f"{'different channel':>18s}")
for name in CLASSES:
    row = results[name]
    print(f"  {name:6s} {row['SAME recording']['frac_above_0p9']:>15.1%} "
          f"{row['other SNR']['frac_above_0p9']:>14.1%} "
          f"{row['other chan']['frac_above_0p9']:>17.1%}")

same = np.mean([results[c]["SAME recording"]["frac_above_0p9"] for c in CLASSES])
other_snr = np.mean([results[c]["other SNR"]["frac_above_0p9"] for c in CLASSES])
print(f"\n  mean across classes: same recording {same:.1%}, different SNR {other_snr:.1%}")
print("  -> an SNR-disjoint split is capture-disjoint iff the second column is ~0."
      if other_snr < 0.01 else
      "  -> content DOES repeat across recordings; no split of this dataset is clean.")

json.dump(results, open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stderr, indent=2)
