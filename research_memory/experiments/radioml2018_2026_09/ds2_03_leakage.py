"""Dataset 2 (RadioML 2018.01A) - leakage and split validity.

This dataset ships as ONE monolithic array with no train/test split and no recording or
capture identifier of any kind. So the split is ours to design, and the only question that
matters is whether any split can be made genuinely independent.

Three questions, in order of how much damage a wrong answer does:

L1  Do the 26 SNR levels of a class share the SAME underlying waveform, differing only in
    added noise? If so, frame k of (class, 30 dB) and frame k of (class, 20 dB) are the same
    transmission, and ANY split that puts them on opposite sides leaks - including a split by
    SNR, which is exactly what worked for Dataset 1. Tested by comparing frames at equal
    offsets within their blocks.

L2  Are consecutive frames inside a block contiguous slices of one continuous recording? If
    they are, neighbouring frames are near-duplicates and a random frame split leaks.

L3  Near-duplicate rate within a block, measured against a phase-randomised surrogate null.
    The naive 1/sqrt(N) floor is wrong for bandlimited frames - that error cost real time on
    Dataset 1 and is not repeated here.

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
RNG = np.random.default_rng(4242)


def start_of(class_index, snr):
    return (class_index * len(SNRS) + SNRS.index(snr)) * BLOCK


def load(handle, class_index, snr, count=200, offset=0, normalise=True):
    s = start_of(class_index, snr) + offset
    raw = handle["X"][s:s + count]
    iq = (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)
    if not normalise:
        return iq
    iq = iq - iq.mean(axis=1, keepdims=True)
    return iq / (np.linalg.norm(iq, axis=1, keepdims=True) + 1e-12)


def surrogate(frames):
    spec = np.fft.fft(frames, axis=1)
    out = np.fft.ifft(np.abs(spec) * np.exp(1j * RNG.uniform(0, 2*np.pi, spec.shape)),
                      axis=1).astype(np.complex64)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def peak_corr(a, b, size=2048):
    bs = np.fft.fft(b, n=size, axis=1).conj()
    return np.array([np.abs(np.fft.ifft(np.fft.fft(r, n=size)[None, :] * bs, axis=1)).max()
                     for r in a])


handle = h5py.File(PATH, "r")
DIGITAL = [3, 4, 5, 12, 14, 0, 1, 22, 23]      # the classes RadioFry cares about most

# --- L1: is the same waveform reused across SNR levels? ---------------------------------------

print("=" * 100)
print("L1. DO THE SNR LEVELS SHARE THE SAME UNDERLYING WAVEFORM?")
print("=" * 100)
print("  Frame k of (class, SNR=a) vs frame k of (class, SNR=b), SAME offset in the block.")
print("  If the dataset re-noised one waveform per SNR, this is ~1.0. Matched-offset compared")
print("  against SHUFFLED-offset tells the two apart.\n")
print(f"  {'class':>8s} {'30 vs 28':>10s} {'30 vs 20':>10s} {'30 vs 0':>10s} "
      f"{'shuffled':>10s} {'surrogate':>10s}")

l1 = {}
for index in [3, 4, 12, 14, 22]:
    reference = load(handle, index, 30, 150)
    row = {}
    for other in (28, 20, 0):
        partner = load(handle, index, other, 150)
        # matched offset: element-wise, frame k vs frame k
        matched = np.abs(np.sum(reference * partner.conj(), axis=1))
        row[other] = float(np.median(matched))
    shuffled = load(handle, index, 20, 150)[RNG.permutation(150)]
    row["shuffled"] = float(np.median(np.abs(np.sum(reference * shuffled.conj(), axis=1))))
    row["surrogate"] = float(np.median(
        np.abs(np.sum(reference * surrogate(load(handle, index, 20, 150)).conj(), axis=1))))
    l1[FIXED[index]] = row
    print(f"  {FIXED[index]:>8s} {row[28]:>10.4f} {row[20]:>10.4f} {row[0]:>10.4f} "
          f"{row['shuffled']:>10.4f} {row['surrogate']:>10.4f}")

verdict_l1 = max(l1[c][20] for c in l1) > 0.5
print(f"\n  => {'SNR levels SHARE waveforms - an SNR-disjoint split would LEAK'if verdict_l1 else 'SNR levels are INDEPENDENT realisations - matched offset is no better than shuffled'}")

# --- L2: are consecutive frames contiguous in time? -------------------------------------------

print("\n" + "=" * 100)
print("L2. ARE CONSECUTIVE FRAMES SLICES OF ONE CONTINUOUS RECORDING?")
print("=" * 100)
print("  If frame k+1 continues frame k, the join correlates. Compared against frames taken")
print("  far apart in the same block.\n")
print(f"  {'class':>8s} {'adjacent k,k+1':>15s} {'far apart':>11s} {'surrogate':>10s}")

for index in [3, 4, 12, 22]:
    frames = load(handle, index, 30, 240)
    adjacent = np.abs(np.sum(frames[:-1] * frames[1:].conj(), axis=1))
    far = np.abs(np.sum(frames[:100] * frames[120:220].conj(), axis=1))
    null = np.abs(np.sum(frames[:100] * surrogate(frames[100:200]).conj(), axis=1))
    print(f"  {FIXED[index]:>8s} {np.median(adjacent):>15.4f} {np.median(far):>11.4f} "
          f"{np.median(null):>10.4f}")

# --- L3: near-duplicate rate within a block ---------------------------------------------------

print("\n" + "=" * 100)
print("L3. NEAR-DUPLICATE RATE WITHIN ONE (class, SNR) BLOCK")
print("=" * 100)
print("  peak |corr| over all lags, DC removed, against a phase-randomised surrogate null.\n")
print(f"  {'class':>8s} {'observed med':>13s} {'null med':>10s} {'excess':>8s} {'frac>0.9':>9s}")

l3 = {}
for index in [3, 4, 12, 14, 0, 22]:
    probe = load(handle, index, 30, 120, offset=0)
    bank = load(handle, index, 30, 1200, offset=1500)
    observed = peak_corr(probe, bank)
    null = peak_corr(probe, surrogate(bank))
    l3[FIXED[index]] = {"observed": float(np.median(observed)),
                        "null": float(np.median(null)),
                        "frac_above_0p9": float(np.mean(observed > 0.9))}
    print(f"  {FIXED[index]:>8s} {np.median(observed):>13.4f} {np.median(null):>10.4f} "
          f"{np.median(observed)-np.median(null):>+8.4f} {np.mean(observed>0.9):>8.1%}")

# --- L4: exact duplicates across a sample of the file ------------------------------------------

print("\n" + "=" * 100)
print("L4. EXACT DUPLICATE FRAMES (sampled)")
print("=" * 100)
import hashlib
seen = {}
collisions = 0
checked = 0
for index in range(0, 24, 4):
    for snr in (-20, 0, 30):
        raw = handle["X"][start_of(index, snr):start_of(index, snr) + 500]
        for j in range(raw.shape[0]):
            d = hashlib.blake2b(np.ascontiguousarray(raw[j]).tobytes(), digest_size=16).digest()
            checked += 1
            if d in seen:
                collisions += 1
            seen[d] = (index, snr, j)
print(f"  checked {checked:,} frames across {len(range(0,24,4))*3} blocks: "
      f"{collisions} exact duplicates")

handle.close()

if len(sys.argv) > 1:
    json.dump({"l1": l1, "l3": l3}, open(sys.argv[1], "w"), indent=2)
