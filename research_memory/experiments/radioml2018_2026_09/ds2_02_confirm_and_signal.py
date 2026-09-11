"""Dataset 2: confirm the class ordering orthogonally, then characterise the signals.

A. ORDERING CONFIRMATION, independent of the impropriety argument.
   For an M-ary PSK, raising the signal to the M-th power collapses the constellation to a
   single point, so |E[x^M]| peaks at exactly M. Under the FIXED ordering indices 3..7 are
   BPSK, QPSK, 8PSK, 16PSK, 32PSK, which must give a clean monotone 2, 4, 8, 16, 32. No
   arrangement of the ORIGINAL ordering produces that sequence. This shares no assumption
   with the impropriety test.

B. Symbol rate / samples-per-symbol, occupied bandwidth, amplitude normalisation.

C. Whether RadioFry's 4ASK <-> PAM4 mapping is scientifically valid, or whether it is the
   same kind of convenient-but-wrong mapping that GMSK -> GFSK would be.

READ ONLY.
"""
import json
import sys

import h5py
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATH = r"C:\Users\Kaustubh Bhoir\Documents\RadioFry\dataset 2\GOLD_XYZ_OSC.0001_1024.hdf5"
FIXED = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
         '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
         'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
BLOCK = 4096
SNRS = list(range(-20, 32, 2))


def block_start(class_index: int, snr: int) -> int:
    """The file is laid out as contiguous 4096-frame (class, SNR) blocks, verified in step 1."""
    return (class_index * len(SNRS) + SNRS.index(snr)) * BLOCK


def load(handle, class_index: int, snr: int, count: int = 256, offset: int = 0):
    start = block_start(class_index, snr) + offset
    raw = handle["X"][start:start + count]
    return (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)


handle = h5py.File(PATH, "r")

# verify the block formula against the stored labels before trusting it
Y = handle["Y"]
Z = handle["Z"]
ok = True
for index in (0, 5, 13, 23):
    for snr in (-20, 0, 30):
        start = block_start(index, snr)
        if int(np.argmax(Y[start])) != index or int(Z[start][0]) != snr:
            ok = False
print(f"block-index formula verified against stored labels: {ok}\n")

# --- A. PSK order -----------------------------------------------------------------------------

print("=" * 100)
print("A. INDEPENDENT ORDERING CONFIRMATION: M-th power moment peaks at the PSK order M")
print("=" * 100)
print("  FIXED predicts indices 3-7 are BPSK, QPSK, 8PSK, 16PSK, 32PSK -> peak M = 2,4,8,16,32\n")
print(f"  {'idx':>3s} {'FIXED':>8s} " + "".join(f"{f'M={m}':>9s}" for m in (2, 4, 8, 16, 32))
      + "   peak")

for index in range(3, 8):
    frames = load(handle, index, 30, 400)
    unit = frames / np.maximum(np.abs(frames), 1e-12)      # phase only
    moments = {}
    for m in (2, 4, 8, 16, 32):
        moments[m] = float(np.abs(np.mean(unit ** m, axis=1)).mean())
    peak = max(moments, key=moments.get)
    print(f"  {index:>3d} {FIXED[index]:>8s} "
          + "".join(f"{moments[m]:>9.4f}" for m in (2, 4, 8, 16, 32))
          + f"   M={peak}")

print("\n  (a constant-envelope M-PSK collapses to one point at its own order; the peak is")
print("   read directly off the row above)")

# --- B. symbol rate, bandwidth, amplitude ------------------------------------------------------

print("\n" + "=" * 100)
print("B. SYMBOL RATE / SPS / BANDWIDTH / AMPLITUDE   (SNR = 30 dB)")
print("=" * 100)
print(f"  {'idx':>3s} {'class':>10s} {'|s|^2 line':>11s} {'sps':>6s} {'BW99 (frac fs)':>15s} "
      f"{'mean power':>11s} {'power std':>10s}")

signal = {}
for index in range(24):
    frames = load(handle, index, 30, 400)
    x = frames - frames.mean(axis=1, keepdims=True)

    envelope = np.abs(x) ** 2
    envelope -= envelope.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(envelope * np.hanning(1024), axis=1)) ** 2
    average = spec.mean(axis=0)
    bins = np.fft.rfftfreq(1024, 1.0)          # cycles/sample
    valid = bins > 0.01
    line = float(bins[valid][np.argmax(average[valid])])

    full = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2
    mean_spectrum = full.mean(axis=0)
    cumulative = np.cumsum(mean_spectrum) / mean_spectrum.sum()
    low = int(np.searchsorted(cumulative, 0.005))
    high = int(np.searchsorted(cumulative, 0.995))
    bandwidth = (high - low) / 1024.0

    power = np.mean(np.abs(frames) ** 2, axis=1)
    signal[index] = {"line_cycles_per_sample": line, "sps": 1.0 / line if line else float("nan"),
                     "bw99_fraction_fs": bandwidth, "mean_power": float(power.mean()),
                     "power_std": float(power.std())}
    print(f"  {index:>3d} {FIXED[index]:>10s} {line:>11.4f} {1.0/line if line else float('nan'):>6.2f} "
          f"{bandwidth:>15.4f} {power.mean():>11.5f} {power.std():>10.5f}")

print("\n  NOTE: the |s|^2 line is only a valid symbol-rate estimator for linearly modulated,")
print("  non-constant-envelope classes. Rows for FM / GMSK / the PSK family are NOT symbol")
print("  rates and must not be quoted as such.")

# --- amplitude normalisation across the whole set ---------------------------------------------

print("\n  amplitude normalisation check, pooled over all 24 classes at 30 dB:")
powers = np.array([signal[i]["mean_power"] for i in range(24)])
print(f"    per-class mean frame power: min {powers.min():.5f}  max {powers.max():.5f}  "
      f"spread {10*np.log10(powers.max()/powers.min()):.2f} dB")
print("    (Dataset 1 had a 23.3 dB spread, which was a usable amplitude shortcut)")

# --- C. is 4ASK the same thing as RadioFry's PAM4? ---------------------------------------------

print("\n" + "=" * 100)
print("C. IS RadioML '4ASK' THE SAME MODULATION AS RadioFry's 'PAM4'?")
print("=" * 100)
for index in (0, 1, 2, 3):
    frames = load(handle, index, 30, 200)
    raw_mean = np.abs(frames.mean(axis=1))
    rms = np.sqrt(np.mean(np.abs(frames) ** 2, axis=1))
    # project onto the dominant real axis, then look at the level structure
    rotated = frames * np.exp(-1j * np.angle(np.mean(frames ** 2, axis=1, keepdims=True)) / 2)
    real = rotated.real.ravel()
    real = real / np.std(real)
    print(f"  index {index} ({FIXED[index]}): "
          f"|mean|/rms = {float(np.median(raw_mean / rms)):.3f}  "
          f"(0 = bipolar/zero-mean, >0 = unipolar)")
    quantiles = np.percentile(real, [5, 25, 50, 75, 95])
    print(f"      real-axis quantiles (5/25/50/75/95): "
          f"{'  '.join(f'{q:+.2f}' for q in quantiles)}")

handle.close()

print("\n  RadioFry's PAM4 must be checked against this before any mapping is made.")

if len(sys.argv) > 1:
    json.dump(signal, open(sys.argv[1], "w"), indent=2)
