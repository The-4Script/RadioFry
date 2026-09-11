"""Dataset 2 (RadioML 2018.01A) - structure, distributions, and the class-ordering question.

The file ships with TWO contradictory class orderings and carries NO class-name metadata of
its own, so the mapping cannot be read from the data the way Dataset 1's `mod2id_json` could.
It has to be settled by measurement, and getting it wrong mislabels every frame.

Decisive discriminators, chosen because they are physical and unambiguous at high SNR:

  impropriety |E[x^2]|/E[|x|^2]   ~1 for a real-valued constellation (OOK, ASK, BPSK, AM),
                                  ~0 for a proper complex one (QAM, PSK, APSK)
  zero fraction                   OOK alone parks ~half its symbols at zero amplitude
  amplitude CV                    ~0 for constant-envelope (PSK, FM, GMSK, OQPSK)
  spectral asymmetry              only the SSB classes suppress one sideband
  DC fraction                     only the "with carrier" (WC) variants carry a tone at 0 Hz

Under the FIXED ordering indices 0-3 are OOK, 4ASK, 8ASK, BPSK - all improper.
Under the ORIGINAL ordering indices 0-3 are 32PSK, 16APSK, 32QAM, FM - all proper.
One measurement separates them.

READ ONLY.
"""
import collections
import json
import sys

import h5py
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATH = r"C:\Users\Kaustubh Bhoir\Documents\RadioFry\dataset 2\GOLD_XYZ_OSC.0001_1024.hdf5"

ORIGINAL = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK',
            '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK',
            'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
FIXED = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
         '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
         'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

handle = h5py.File(PATH, "r")
Y = handle["Y"][:]
Z = handle["Z"][:].ravel()
labels = np.argmax(Y, axis=1)
n = labels.size

print("=" * 100)
print("1. LABEL AND SNR STRUCTURE")
print("=" * 100)
print(f"  frames: {n:,}   one-hot rows all sum to 1: {bool((Y.sum(axis=1) == 1).all())}")
snrs = sorted(set(Z.tolist()))
print(f"  classes present: {len(set(labels.tolist()))}   SNR levels: {len(snrs)}  {snrs}")
print(f"  24 x 26 x 4096 = {24*26*4096:,}  matches frame count: {24*26*4096 == n}")

counts = collections.Counter(labels.tolist())
per_pair = collections.Counter(zip(labels.tolist(), Z.tolist()))
sizes = set(per_pair.values())
print(f"  frames per class: {set(counts.values())}")
print(f"  frames per (class, SNR) pair: {sizes}   distinct pairs: {len(per_pair)} / {24*26}")
print("  => PERFECTLY BALANCED" if len(sizes) == 1 else "  => NOT balanced")

# Is the file ordered in contiguous (class, SNR) blocks?
changes = np.flatnonzero(np.diff(labels) != 0) + 1
print(f"\n  label changes at {len(changes)} positions; first few: {changes[:4].tolist()}")
block = 4096
ordered = all(len(set(labels[i:i + block].tolist())) == 1 and len(set(Z[i:i + block].tolist())) == 1
              for i in range(0, min(n, 200 * block), block))
print(f"  every 4096-frame block is a single (class, SNR): {ordered}")
print(f"  => the file is laid out as contiguous per-configuration blocks"
      if ordered else "  => layout is NOT block-contiguous")

# index of the first frame of each (class, snr) block
first = {}
for i in range(0, n, block):
    first[(int(labels[i]), int(Z[i]))] = i
print(f"  block index recovered for {len(first)} configurations")


# --- 2. the ordering test -------------------------------------------------------------------

def features(frames: np.ndarray) -> dict:
    """Physical discriminators, measured on complex baseband frames."""
    x = frames - frames.mean(axis=1, keepdims=True)      # DC removed for envelope stats
    power = np.mean(np.abs(x) ** 2, axis=1)
    amplitude = np.abs(x)
    rms = np.sqrt(power)[:, None]

    impropriety = np.abs(np.mean(x ** 2, axis=1)) / np.maximum(power, 1e-20)
    zero_fraction = np.mean(amplitude < 0.2 * rms, axis=1)
    amplitude_cv = amplitude.std(axis=1) / np.maximum(amplitude.mean(axis=1), 1e-20)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2
    half = spectrum.shape[1] // 2
    lower, upper = spectrum[:, :half].sum(axis=1), spectrum[:, half:].sum(axis=1)
    asymmetry = np.abs(upper - lower) / np.maximum(upper + lower, 1e-20)

    raw_power = np.mean(np.abs(frames) ** 2, axis=1)
    dc_fraction = np.abs(frames.mean(axis=1)) ** 2 / np.maximum(raw_power, 1e-20)

    return {"impropriety": float(np.median(impropriety)),
            "zero_fraction": float(np.median(zero_fraction)),
            "amplitude_cv": float(np.median(amplitude_cv)),
            "asymmetry": float(np.median(asymmetry)),
            "dc_fraction": float(np.median(dc_fraction))}


print("\n" + "=" * 100)
print("2. MEASURED SIGNAL CHARACTER PER LABEL INDEX   (SNR = 30 dB, 256 frames each)")
print("=" * 100)
print(f"  {'idx':>3s} {'improper':>9s} {'zero%':>7s} {'ampCV':>7s} {'asym':>6s} {'DC%':>6s}   "
      f"{'ORIGINAL says':>14s}  {'FIXED says':>12s}")

measured = {}
for index in range(24):
    start = first[(index, 30)]
    frames = handle["X"][start:start + 256]
    iq = (frames[..., 0] + 1j * frames[..., 1]).astype(np.complex64)
    f = features(iq)
    measured[index] = f
    print(f"  {index:>3d} {f['impropriety']:>9.3f} {f['zero_fraction']*100:>6.1f}% "
          f"{f['amplitude_cv']:>7.3f} {f['asymmetry']:>6.3f} {f['dc_fraction']*100:>5.1f}%   "
          f"{ORIGINAL[index]:>14s}  {FIXED[index]:>12s}")

handle.close()

# --- 3. verdict ------------------------------------------------------------------------------

print("\n" + "=" * 100)
print("3. WHICH ORDERING IS CONSISTENT WITH THE MEASUREMENTS?")
print("=" * 100)

REAL_VALUED = {"OOK", "4ASK", "8ASK", "BPSK", "AM-SSB-SC", "AM-SSB-WC",
               "AM-DSB-SC", "AM-DSB-WC"}
CONSTANT_ENVELOPE = {"BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "FM", "GMSK", "OQPSK"}

for name, ordering in (("ORIGINAL (classes.txt)", ORIGINAL), ("FIXED (classes-fixed)", FIXED)):
    hits = 0
    total = 0
    detail = []
    for index in range(24):
        cls = ordering[index]
        f = measured[index]
        # a real-valued constellation must show impropriety well above zero
        expected_improper = cls in REAL_VALUED
        got_improper = f["impropriety"] > 0.3
        total += 1
        if expected_improper == got_improper:
            hits += 1
        else:
            detail.append(f"{index}:{cls}(improper={f['impropriety']:.2f})")
    print(f"\n  {name}")
    print(f"    impropriety agrees for {hits}/{total} classes")
    if detail:
        print(f"    disagreements: {', '.join(detail[:8])}"
              + (" ..." if len(detail) > 8 else ""))

print("\n  single-class checks (each is unique in the label set):")
ook_index = max(measured, key=lambda i: measured[i]["zero_fraction"])
print(f"    highest zero-amplitude fraction  -> index {ook_index} "
      f"({measured[ook_index]['zero_fraction']*100:.1f}%)  "
      f"ORIGINAL={ORIGINAL[ook_index]}  FIXED={FIXED[ook_index]}   [expect OOK]")
asym_index = sorted(measured, key=lambda i: -measured[i]["asymmetry"])[:2]
print(f"    strongest spectral asymmetry     -> indices {asym_index}  "
      f"ORIGINAL={[ORIGINAL[i] for i in asym_index]}  "
      f"FIXED={[FIXED[i] for i in asym_index]}   [expect the two AM-SSB]")
dc_index = sorted(measured, key=lambda i: -measured[i]["dc_fraction"])[:2]
print(f"    strongest DC / residual carrier  -> indices {dc_index}  "
      f"ORIGINAL={[ORIGINAL[i] for i in dc_index]}  "
      f"FIXED={[FIXED[i] for i in dc_index]}   [expect the two *-WC]")

if len(sys.argv) > 1:
    json.dump({"measured": measured,
               "per_class": {str(k): v for k, v in counts.items()},
               "snrs": snrs, "frames": int(n)},
              open(sys.argv[1], "w"), indent=2)
