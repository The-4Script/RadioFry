"""Resolve the two questions the first measurement pass left open.

A. Is the 199 kHz line the symbol rate, or its second harmonic?
   Settled by looking at the whole cyclic spectrum instead of only its argmax: if the
   fundamental at ~100 kHz is present at all, the paper's 100 kSps is right and 199 kHz is
   the harmonic. Cross-checked against the project's own SCD estimator, which is the
   rigorous cyclostationary method rather than a magnitude-squared shortcut.

B. Which channel label is the multipath one?
   The PSD-ripple test was too blunt. An echo at delay D produces a peak at quefrency D in
   the cepstrum of the PSD; the documented delays are 100 and 200 us = 200 and 400 samples
   at 2 MSps. That is a direct, localised test.

READ ONLY.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from radiofry.contracts import UnifiedSignalContainer  # noqa: E402
from radiofry.dsp.spectral_correlation import (  # noqa: E402
    SCDConfig, compute_scd, dominant_cycle_frequencies,
)

ROOT = Path(r"C:\Users\Kaustubh Bhoir\Documents\RadioFry"
            r"\Real-World IQ Dataset for Automatic Radio Modulati\dataset")
FS = 2_000_000.0


def load(split, classes, chan=None, snr=30, per=400, seed=0):
    with h5py.File(ROOT / f"subset_{split}.h5", "r") as handle:
        mapping = json.loads(handle.attrs["mod2id_json"])
        mods, chans, snrs = handle["y_mod"][:], handle["y_chan"][:], handle["y_snr"][:]
        rng = np.random.default_rng(seed)
        keep = np.isin(mods, [mapping[c] for c in classes]) & (snrs == snr)
        if chan is not None:
            keep &= chans == chan
        pool = np.flatnonzero(keep)
        index = np.sort(rng.choice(pool, size=min(per, pool.size), replace=False))
        raw = handle["X"][index].astype(np.float32)
        return (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)


# --- A. fundamental or harmonic? -----------------------------------------------------------

print("=" * 96)
print("A. IS 199 kHz THE SYMBOL RATE OR ITS SECOND HARMONIC?")
print("=" * 96)
print("  Averaged |s|^2 spectrum. If the paper's 100 kSps is right there must be a line at")
print("  ~100 kHz; the question is whether one exists at all.\n")

for name in ("BPSK", "QPSK", "QAM"):
    frames = load("test", [name], chan=0)
    envelope = np.abs(frames) ** 2
    envelope -= envelope.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(envelope * np.hanning(frames.shape[1]), axis=1)) ** 2
    average = spectrum.mean(axis=0)
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / FS)

    # local noise floor: median away from DC
    floor = np.median(average[(freqs > 20_000)])
    decibels = 10 * np.log10(average / floor + 1e-30)

    print(f"  {name}:")
    for target in (100_000, 200_000, 300_000, 400_000):
        window = (freqs > target - 12_000) & (freqs < target + 12_000)
        peak = np.max(decibels[window])
        at = freqs[window][np.argmax(decibels[window])]
        flag = "LINE" if peak > 10 else "  --"
        print(f"    near {target/1000:>5.0f} kHz: peak {peak:6.1f} dB over floor "
              f"at {at:>8.0f} Hz   {flag}")
    top = np.argsort(average[freqs > 20_000])[-4:][::-1]
    valid_freqs = freqs[freqs > 20_000]
    print(f"    strongest lines: "
          f"{[f'{valid_freqs[i]:.0f} Hz' for i in top]}")

print("\n  Cross-check with the project's own SCD estimator (cyclostationary, not |s|^2):")
for name in ("BPSK", "QPSK", "QAM"):
    frames = load("test", [name], chan=0, per=64)
    stream = frames.reshape(-1)                       # concatenated frames -> longer record
    result = compute_scd(stream[:262_144], FS, SCDConfig(
        fft_size=256, alpha_max_hz=500_000.0, max_samples=262_144))
    features = dominant_cycle_frequencies(result, count=4)
    print(f"    {name:5s} dominant alpha: "
          f"{[f'{f.frequency_hz:.0f} Hz' for f in features]}")


# --- B. which channel label is multipath? ---------------------------------------------------

print("\n" + "=" * 96)
print("B. WHICH CHANNEL LABEL CARRIES THE MULTIPATH?")
print("=" * 96)
print("  Cepstrum of the PSD. Documented delays 100 and 200 us = 200 and 400 samples.\n")
print(f"  {'class':6s} {'chan':>5s} {'ceps@200':>10s} {'ceps@400':>10s} {'ceps floor':>11s} "
      f"{'ratio200':>9s} {'ratio400':>9s}")

polarity = {}
for name in ("BPSK", "QPSK", "QAM", "GMSK"):
    row = {}
    for chan in (0, 1):
        frames = load("test", [name], chan=chan, per=600)
        spectrum = np.abs(np.fft.fft(frames, axis=1)) ** 2
        log_psd = np.log(spectrum.mean(axis=0) + 1e-20)
        cepstrum = np.abs(np.fft.rfft(log_psd - log_psd.mean()))
        floor = float(np.median(cepstrum[20:500]))
        c200, c400 = float(cepstrum[200]), float(cepstrum[400])
        row[chan] = (c200, c400, floor)
        print(f"  {name:6s} {chan:>5d} {c200:>10.3f} {c400:>10.3f} {floor:>11.3f} "
              f"{c200/floor:>9.2f} {c400/floor:>9.2f}")
    polarity[name] = row

print("\n  verdict per class (higher cepstral echo energy = the multipath label):")
for name, row in polarity.items():
    score0 = row[0][0] / row[0][2] + row[0][1] / row[0][2]
    score1 = row[1][0] / row[1][2] + row[1][1] / row[1][2]
    print(f"    {name:6s} chan0 {score0:6.2f}  chan1 {score1:6.2f}  -> "
          f"{'chan=1' if score1 > score0 else 'chan=0'} looks like multipath")


# --- C. how separable are the two channel labels at all? ------------------------------------

print("\n" + "=" * 96)
print("C. ARE THE TWO CHANNEL LABELS DISTINGUISHABLE AT ALL?")
print("=" * 96)
for name in ("BPSK", "QAM"):
    a = load("test", [name], chan=0, per=800, seed=1)
    b = load("test", [name], chan=1, per=800, seed=1)
    pa = np.mean(np.abs(a) ** 2, axis=1)
    pb = np.mean(np.abs(b) ** 2, axis=1)
    print(f"  {name:6s} mean frame power  chan0 {pa.mean():.5f}  chan1 {pb.mean():.5f}  "
          f"({10*np.log10(pa.mean()/pb.mean()):+.1f} dB)")
    # spectral shape distance
    sa = np.fft.fftshift(np.abs(np.fft.fft(a, axis=1)) ** 2).mean(axis=0)
    sb = np.fft.fftshift(np.abs(np.fft.fft(b, axis=1)) ** 2).mean(axis=0)
    sa, sb = sa / sa.sum(), sb / sb.sum()
    print(f"         normalised-PSD total variation distance: "
          f"{0.5*np.abs(sa-sb).sum():.4f}")
