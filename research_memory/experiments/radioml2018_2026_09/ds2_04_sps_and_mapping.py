"""Dataset 2: per-frame samples-per-symbol, and whether 4ASK may be mapped to RadioFry PAM4.

A. SPS. The pooled |s|^2 estimate in the previous pass gave incoherent values (16 to 93),
   which is what a POOLED estimate looks like when sps varies frame to frame. RadioML 2018.01A
   is documented as applying random resampling / clock offset per example. Measured per frame
   here, and reported as a distribution rather than a single number.

B. 4ASK vs PAM4. The earlier attempt was broken: it rotated each frame by
   angle(E[x^2])/2, which carries a +-pi ambiguity, so half the frames flipped sign and the
   pooled histogram looked symmetric no matter what. Redone with a sign-resolved projection
   and compared against RadioFry's OWN PAM4 generator rather than against theory.

READ ONLY with respect to the dataset.
"""
import sys

import h5py
import numpy as np

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATH = r"C:\Users\Kaustubh Bhoir\Documents\RadioFry\dataset 2\GOLD_XYZ_OSC.0001_1024.hdf5"
FIXED = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
         '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
         'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
BLOCK, SNRS = 4096, list(range(-20, 32, 2))


def start_of(c, s):
    return (c * len(SNRS) + SNRS.index(s)) * BLOCK


def load(handle, c, s, count=300, offset=0):
    st = start_of(c, s) + offset
    raw = handle["X"][st:st + count]
    return (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)


handle = h5py.File(PATH, "r")

print("=" * 100)
print("A. SAMPLES-PER-SYMBOL, MEASURED PER FRAME (SNR = 30 dB)")
print("=" * 100)
print("  |s|^2 cyclic line found independently in each frame, then summarised.")
print("  A single fixed sps gives a tight distribution; per-example resampling gives a wide one.\n")
print(f"  {'class':>8s} {'median sps':>11s} {'5th':>7s} {'95th':>7s} {'IQR':>7s} "
      f"{'line SNR dB':>12s}")

for index in (3, 4, 12, 14, 0, 1):
    frames = load(handle, index, 30, 400)
    x = frames - frames.mean(axis=1, keepdims=True)
    envelope = np.abs(x) ** 2
    envelope -= envelope.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(envelope * np.hanning(1024), axis=1)) ** 2
    bins = np.fft.rfftfreq(1024, 1.0)
    valid = bins > 0.02                       # exclude DC region; sps < 50
    sub = spec[:, valid]
    subbins = bins[valid]
    peak_at = subbins[np.argmax(sub, axis=1)]
    floor = np.median(sub, axis=1)
    strength = 10 * np.log10(sub.max(axis=1) / np.maximum(floor, 1e-30))
    sps = 1.0 / peak_at
    print(f"  {FIXED[index]:>8s} {np.median(sps):>11.2f} {np.percentile(sps,5):>7.2f} "
          f"{np.percentile(sps,95):>7.2f} "
          f"{np.percentile(sps,75)-np.percentile(sps,25):>7.2f} "
          f"{np.median(strength):>12.1f}")

print("\n  Occupied bandwidth per frame is the more robust handle when the cyclic line is weak:")
print(f"  {'class':>8s} {'BW99 median':>12s} {'5th':>8s} {'95th':>8s}  "
      f"{'implied sps if RRC a=0.35':>26s}")
for index in (3, 4, 12, 14):
    frames = load(handle, index, 30, 400)
    x = frames - frames.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2
    cumulative = np.cumsum(spec, axis=1) / spec.sum(axis=1, keepdims=True)
    low = np.argmax(cumulative > 0.005, axis=1)
    high = np.argmax(cumulative > 0.995, axis=1)
    bw = (high - low) / 1024.0
    print(f"  {FIXED[index]:>8s} {np.median(bw):>12.4f} {np.percentile(bw,5):>8.4f} "
          f"{np.percentile(bw,95):>8.4f}  {1.35/np.median(bw):>26.2f}")

# --- B. 4ASK vs PAM4 --------------------------------------------------------------------------

print("\n" + "=" * 100)
print("B. IS RadioML 4ASK THE SAME MODULATION AS RadioFry PAM4?")
print("=" * 100)


def level_profile(iq: np.ndarray) -> np.ndarray:
    """Sign-resolved projection onto the constellation axis, pooled over frames.

    The axis is fixed per frame by the dominant eigenvector of the real 2x2 covariance, and
    the sign ambiguity is resolved by forcing positive skewness -- so unipolar and bipolar
    constellations stay distinguishable instead of being symmetrised by a random flip.
    """
    out = []
    for frame in iq:
        pts = np.stack([frame.real, frame.imag])
        pts = pts - pts.mean(axis=1, keepdims=True)
        cov = pts @ pts.T / pts.shape[1]
        values, vectors = np.linalg.eigh(cov)
        axis = vectors[:, np.argmax(values)]
        projected = axis @ pts
        if np.mean(projected ** 3) < 0:          # resolve the +-1 ambiguity consistently
            projected = -projected
        out.append(projected / (projected.std() + 1e-12))
    return np.concatenate(out)


print("  RadioML classes, pooled constellation-axis histogram (10 bins over +-2.5 sigma):")
for index in (0, 1, 2, 3):
    profile = level_profile(load(handle, index, 30, 60))
    hist, edges = np.histogram(profile, bins=10, range=(-2.5, 2.5), density=True)
    bar = "".join("#" if v > 0.25 else ("+" if v > 0.12 else "."), ) if False else \
        "".join("#" if v > 0.25 else ("+" if v > 0.12 else ".") for v in hist)
    print(f"    {FIXED[index]:>6s}  {bar}   raw mean/std = "
          f"{float(np.mean(profile)):+.3f}")

# RadioFry's own PAM4, for a like-for-like comparison
try:
    from radiofry.synthetic_gen.v1 import SampleSpec, generate_source_bits, modulate
    print("\n  RadioFry's own generator, same measurement:")
    for name in ("PAM4", "BPSK"):
        spec = SampleSpec(modulation=name, samples_per_symbol=8, num_symbols=128,
                          snr_db=30.0, seed=5)
        bits = generate_source_bits(spec)
        wave = np.asarray(modulate(spec, bits), dtype=np.complex64)
        chunks = wave[: (wave.size // 1024) * 1024].reshape(-1, 1024)
        if chunks.shape[0] == 0:
            chunks = np.resize(wave, (1, 1024)).astype(np.complex64)
        profile = level_profile(chunks)
        hist, _ = np.histogram(profile, bins=10, range=(-2.5, 2.5), density=True)
        bar = "".join("#" if v > 0.25 else ("+" if v > 0.12 else ".") for v in hist)
        print(f"    {name:>6s}  {bar}   raw mean/std = {float(np.mean(profile)):+.3f}")
except Exception as error:                                    # noqa: BLE001
    print(f"\n  (could not run RadioFry's PAM4 generator: {error})")

handle.close()
print("\n  A 4-level bipolar PAM4 shows four roughly equal humps symmetric about zero.")
print("  A unipolar ASK piles up on one side. Read the bars, not the label.")
