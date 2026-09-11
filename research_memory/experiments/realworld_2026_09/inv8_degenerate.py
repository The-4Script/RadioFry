"""Why do 28.7% of OFDM and WBFM probe frames match EVERY bank equally well?

That figure was identical against the same recording, a different SNR and a different
channel condition - so it is a property of those probe frames, not of what they are being
compared against. The likely cause is degenerate frames: a frame that is essentially silent,
or dominated by a handful of samples, normalises to something that correlates highly with
anything. OFDM and WBFM also have the lowest frame power in the dataset by a factor of ~200,
which puts them closest to the float16 quantisation floor.

If such frames exist they are a data-quality defect: unlearnable, and corrupting to both
training and evaluation.

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
CLASSES = ("BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM")


def load_raw(split, *, limit, seed, classes=None, chan=None, snr=None):
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
        return (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)


def participation_ratio(frames: np.ndarray) -> np.ndarray:
    """Effective number of samples carrying the energy.

    (sum |x|^2)^2 / sum |x|^4. Equals N for a flat envelope and 1 for a single spike.
    """
    power = np.abs(frames) ** 2
    return (power.sum(axis=1) ** 2) / (np.maximum((power ** 2).sum(axis=1), 1e-30))


print("=" * 100)
print("FRAME DEGENERACY, by class   (clean, 30 dB, 1,500 test frames each)")
print("=" * 100)
print("  participation ratio: effective sample count out of 1024.")
print("  A healthy modulated frame sits in the hundreds; a spike-dominated one near 1.\n")
print(f"  {'class':6s} {'mean power':>12s} {'PR median':>10s} {'PR 5th':>8s} "
      f"{'PR<50':>7s} {'PR<10':>7s} {'exact zeros':>12s} {'distinct lvls':>13s}")

flagged = {}
for name in CLASSES:
    frames = load_raw("test", limit=1_500, seed=301, classes=[name], chan=0, snr=30)
    power = np.mean(np.abs(frames) ** 2, axis=1)
    ratio = participation_ratio(frames - frames.mean(axis=1, keepdims=True))
    zeros = np.mean(np.all(frames == 0, axis=1))
    # how many distinct float16 levels the real part uses - a quantisation health check
    levels = np.median([len(np.unique(f.real)) for f in frames[:200]])
    print(f"  {name:6s} {power.mean():>12.6f} {np.median(ratio):>10.1f} "
          f"{np.percentile(ratio, 5):>8.1f} {np.mean(ratio < 50):>6.1%} "
          f"{np.mean(ratio < 10):>6.1%} {zeros:>11.1%} {levels:>13.0f}")
    flagged[name] = {"frac_pr_below_50": float(np.mean(ratio < 50)),
                     "frac_pr_below_10": float(np.mean(ratio < 10)),
                     "median_pr": float(np.median(ratio))}

print("\n" + "=" * 100)
print("ARE THE 'MATCHES EVERYTHING' FRAMES THE DEGENERATE ONES?")
print("=" * 100)
RNG = np.random.default_rng(7)
for name in ("OFDM", "WBFM", "BPSK"):
    probe = load_raw("test", limit=150, seed=201, classes=[name], chan=0, snr=30)
    bank = load_raw("train", limit=1_200, seed=302, classes=[name], chan=0, snr=30)
    probe_dc = probe - probe.mean(axis=1, keepdims=True)
    bank_dc = bank - bank.mean(axis=1, keepdims=True)
    ratio = participation_ratio(probe_dc)
    probe_n = probe_dc / (np.linalg.norm(probe_dc, axis=1, keepdims=True) + 1e-12)
    bank_n = bank_dc / (np.linalg.norm(bank_dc, axis=1, keepdims=True) + 1e-12)

    size = 2048
    bank_spectrum = np.fft.fft(bank_n, n=size, axis=1).conj()
    best = np.array([np.abs(np.fft.ifft(
        np.fft.fft(row, n=size)[None, :] * bank_spectrum, axis=1)).max() for row in probe_n])

    high = best > 0.9
    print(f"  {name}: {high.sum()} of {len(best)} frames match at >0.9")
    if high.any():
        print(f"    participation ratio, matching frames    : "
              f"median {np.median(ratio[high]):8.1f}  max {ratio[high].max():8.1f}")
    if (~high).any():
        print(f"    participation ratio, non-matching frames: "
              f"median {np.median(ratio[~high]):8.1f}  min {ratio[~high].min():8.1f}")
    print(f"    frame power, matching {np.median(np.mean(np.abs(probe[high])**2,axis=1)) if high.any() else float('nan'):.6f}"
          f"  non-matching {np.median(np.mean(np.abs(probe[~high])**2,axis=1)) if (~high).any() else float('nan'):.6f}")

print("\n" + "=" * 100)
print("DEGENERATE-FRAME PREVALENCE ACROSS THE WHOLE TEST SPLIT")
print("=" * 100)
with h5py.File(ROOT / "subset_test.h5", "r") as handle:
    mapping = json.loads(handle.attrs["mod2id_json"])
    id_to_name = {int(v): k for k, v in mapping.items()}
    mods = handle["y_mod"][:]
    total = {name: 0 for name in CLASSES}
    bad = {name: 0 for name in CLASSES}
    for start in range(0, handle["X"].shape[0], 20_000):
        block = handle["X"][start:start + 20_000].astype(np.float32)
        iq = block[..., 0] + 1j * block[..., 1]
        iq = iq - iq.mean(axis=1, keepdims=True)
        ratio = participation_ratio(iq)
        names = np.array([id_to_name[int(v)] for v in mods[start:start + len(block)]])
        for name in CLASSES:
            mask = names == name
            total[name] += int(mask.sum())
            bad[name] += int((ratio[mask] < 50).sum())
    print(f"  {'class':6s} {'frames':>9s} {'PR<50':>9s} {'rate':>8s}")
    for name in CLASSES:
        print(f"  {name:6s} {total[name]:>9,} {bad[name]:>9,} "
              f"{bad[name]/max(total[name],1):>7.2%}")
    print(f"  {'TOTAL':6s} {sum(total.values()):>9,} {sum(bad.values()):>9,} "
          f"{sum(bad.values())/sum(total.values()):>7.2%}")

json.dump(flagged, open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stderr, indent=2)
