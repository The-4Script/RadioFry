"""Measure what the real dataset actually contains, rather than trusting the paper.

Three questions the paper leaves open or answers inconsistently:

1. SYMBOL RATE / SPS. Table 1 says 100 kSps at 4 samples/symbol, then says everything was
   upsampled to 2 MSps - which would be sps = 20, not 4. An earlier estimate from the
   pipeline's own estimator said ~199 kHz (sps = 10). Measured here three independent ways.

2. CHANNEL LABEL POLARITY. The paper says the `_ref` filename suffix means CLEAN, but the
   HDF5 attribute says `0=clean, 1=multipath(ref)` - which reads as if `ref` were the
   multipath one. Resolved physically: the documented 3-tap model has delays
   [0, 100, 200] us = [0, 200, 400] samples at 2 MSps, whose frequency response is a comb
   with nulls spaced fs/200 = 10 kHz apart. Frequency-selective ripple is measurable.

3. AMPLITUDE SCALE. The paper says each frame was normalised to unit average power; the
   structural pass measured 0.007-0.14 depending on class. Quantified here as a shortcut:
   how much modulation information is carried by frame power alone.

READ ONLY - opens the HDF5 files with mode 'r' and writes nothing to the dataset directory.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\Kaustubh Bhoir\Documents\RadioFry"
            r"\Real-World IQ Dataset for Automatic Radio Modulati\dataset")
FS = 2_000_000.0
PER_CLASS = 300


def load_stratified(split: str, per_class: int, seed: int = 0):
    """A sample spanning every class, taken at the highest SNR and both channels."""
    with h5py.File(ROOT / f"subset_{split}.h5", "r") as handle:
        mapping = json.loads(handle.attrs["mod2id_json"])
        id_to_name = {int(v): k for k, v in mapping.items()}
        mods = handle["y_mod"][:]
        chans = handle["y_chan"][:]
        snrs = handle["y_snr"][:]
        rng = np.random.default_rng(seed)
        picks = []
        for identifier in sorted(id_to_name):
            for chan in (0, 1):
                pool = np.flatnonzero((mods == identifier) & (chans == chan) & (snrs == 30))
                picks.append(rng.choice(pool, size=min(per_class, pool.size), replace=False))
        index = np.sort(np.concatenate(picks))
        raw = handle["X"][index].astype(np.float32)
        iq = (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)
        return (iq, np.array([id_to_name[int(v)] for v in mods[index]]),
                chans[index], snrs[index])


iq, names, chans, snrs = load_stratified("test", PER_CLASS)
print(f"sample: {len(iq)} frames, classes {sorted(set(names))}\n")

CLASSES = ["BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM"]


# --- 1. symbol rate, three independent estimators ------------------------------------------

def rate_from_magnitude(frames: np.ndarray) -> float:
    """Spectral line of |s|^2 - the classical cyclic-feature symbol-rate estimator."""
    envelope = np.abs(frames) ** 2
    envelope = envelope - envelope.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(envelope * np.hanning(frames.shape[1]), axis=1)) ** 2
    average = spectrum.mean(axis=0)
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / FS)
    valid = freqs > 20_000
    return float(freqs[valid][np.argmax(average[valid])])


def rate_from_fourth_power(frames: np.ndarray) -> float:
    """Nonlinear (4th power) cyclic feature; a different nonlinearity as a cross-check."""
    raised = frames ** 4
    raised = raised - raised.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.fft(raised * np.hanning(frames.shape[1]), axis=1)) ** 2
    average = spectrum.mean(axis=0)
    freqs = np.fft.fftfreq(frames.shape[1], 1 / FS)
    order = np.argsort(np.abs(freqs))
    freqs, average = freqs[order], average[order]
    valid = np.abs(freqs) > 20_000
    return float(abs(freqs[valid][np.argmax(average[valid])]))


def occupied_bandwidth(frames: np.ndarray, fraction: float = 0.99) -> float:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(
        frames * np.hanning(frames.shape[1]), axis=1), axes=1)) ** 2
    average = spectrum.mean(axis=0)
    cumulative = np.cumsum(average) / average.sum()
    lower = np.searchsorted(cumulative, (1 - fraction) / 2)
    upper = np.searchsorted(cumulative, 1 - (1 - fraction) / 2)
    return float((upper - lower) * FS / frames.shape[1])


print("=" * 96)
print("1. SYMBOL RATE / SAMPLES-PER-SYMBOL   (paper Table 1: 100 kSps, sps 4, upsampled to 2 MSps)")
print("=" * 96)
print(f"  {'class':6s} {'|s|^2 line':>12s} {'4th-power':>12s} {'BW99':>10s} "
      f"{'Rs=BW/(1+a)':>12s} {'sps@2MSps':>10s}")
measurements = {}
for name in CLASSES:
    mask = (names == name) & (chans == 0)
    frames = iq[mask]
    magnitude = rate_from_magnitude(frames)
    fourth = rate_from_fourth_power(frames)
    bandwidth = occupied_bandwidth(frames)
    implied = bandwidth / 1.35                       # RRC alpha = 0.35
    print(f"  {name:6s} {magnitude:>10.0f} Hz {fourth:>10.0f} Hz {bandwidth:>8.0f} Hz "
          f"{implied:>10.0f} Hz {FS/max(magnitude,1):>10.1f}")
    measurements[name] = {"magnitude_line_hz": magnitude, "fourth_power_hz": fourth,
                          "bw99_hz": bandwidth, "implied_rs_hz": implied}

print("\n  interpretation for the linearly-modulated classes (BPSK/QPSK/QAM):")
for name in ("BPSK", "QPSK", "QAM"):
    m = measurements[name]
    print(f"    {name:5s} |s|^2 line {m['magnitude_line_hz']:.0f} Hz -> "
          f"sps {FS/m['magnitude_line_hz']:.1f};  BW99 {m['bw99_hz']:.0f} Hz implies "
          f"Rs {m['implied_rs_hz']:.0f} Hz -> sps {FS/max(m['implied_rs_hz'],1):.1f}")


# --- 2. is the channel label polarity what the paper says? ----------------------------------

print("\n" + "=" * 96)
print("2. CHANNEL LABEL POLARITY   (attribute says '0=clean, 1=multipath(ref)')")
print("=" * 96)
print("  A 3-tap channel with delays [0,200,400] samples gives a comb with nulls every")
print("  fs/200 = 10 kHz. Measured as spectral ripple: std of the dB PSD after removing")
print("  the smooth shape, restricted to the occupied band.\n")
print(f"  {'class':6s} {'chan=0 ripple':>15s} {'chan=1 ripple':>15s}  verdict")
for name in CLASSES:
    ripples = {}
    for chan in (0, 1):
        frames = iq[(names == name) & (chans == chan)]
        if frames.size == 0:
            continue
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2
        average = 10 * np.log10(spectrum.mean(axis=0) + 1e-20)
        # smooth shape = 51-bin moving average; ripple = deviation from it, in-band only
        kernel = np.ones(51) / 51
        smooth = np.convolve(average, kernel, mode="same")
        centre = len(average) // 2
        half = int(0.15 * len(average))
        band = slice(centre - half, centre + half)
        ripples[chan] = float(np.std((average - smooth)[band]))
    verdict = ("chan=1 is the frequency-selective one"
               if ripples.get(1, 0) > ripples.get(0, 0) else
               "chan=0 is the frequency-selective one")
    print(f"  {name:6s} {ripples.get(0, float('nan')):>13.2f} dB "
          f"{ripples.get(1, float('nan')):>13.2f} dB  {verdict}")


# --- 3. amplitude shortcut ------------------------------------------------------------------

print("\n" + "=" * 96)
print("3. AMPLITUDE SHORTCUT   (paper: 'each frame individually normalized by average power')")
print("=" * 96)
power = np.mean(np.abs(iq) ** 2, axis=1)
print(f"  per-frame average power across the whole sample: "
      f"min {power.min():.5f} max {power.max():.5f} ratio {power.max()/power.min():.0f}x")
print("  -> the released frames are NOT unit-power normalised.\n")

# how much of the label is in the scalar power alone: nearest-class-median rule
order = np.argsort(power)
print(f"  {'class':6s} {'median power':>14s}  {'10*log10 vs quietest':>22s}")
medians = {}
for name in CLASSES:
    medians[name] = float(np.median(power[names == name]))
quietest = min(medians.values())
for name in CLASSES:
    print(f"  {name:6s} {medians[name]:>14.5f}  {10*np.log10(medians[name]/quietest):>20.1f} dB")

json.dump({"symbol_rate": measurements,
           "power_medians": medians},
          open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stderr, indent=2)
