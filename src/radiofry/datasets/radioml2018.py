"""Read-only adapter for RadioML 2018.01A (DeepSig, `GOLD_XYZ_OSC.0001_1024.hdf5`).

Opened with `h5py` mode `'r'`. Nothing here writes to the dataset.

Structure, measured rather than assumed
---------------------------------------
* `X` `(2_555_904, 1024, 2)` float32, contiguous, uncompressed (20.9 GB).
* `Y` `(N, 24)` int64 one-hot, `Z` `(N, 1)` int64 SNR in dB.
* 24 classes x 26 SNR levels (-20..+30 dB, 2 dB steps) x **exactly 4096 frames** = perfectly
  balanced, all 624 configurations present.
* The file is laid out as contiguous 4096-frame (class, SNR) blocks, in class-major then
  SNR-ascending order. `block_start` encodes that and it is asserted against the stored
  labels before use.
* There is **no train/val/test split and no recording or capture identifier of any kind**.

The class ordering
------------------
The download ships TWO contradictory orderings - `classes.txt` and `classes-fixed.*` - and
the HDF5 carries no class-name metadata to arbitrate. It was settled by measurement
(BANK Entry 047):

* **impropriety** `|E[x^2]|/E[|x|^2]` agrees with the FIXED ordering for **24 of 24** classes
  and with the original for 10 of 24. Indices 0-3 measure 0.99, 0.99, 0.99, 1.00 - real-valued
  constellations, i.e. OOK/4ASK/8ASK/BPSK, not the original's 32PSK/16APSK/32QAM/FM.
* index 21 has amplitude CV **0.007**, a perfectly constant envelope: FM, and impossible for
  the original's AM-DSB-WC.
* index 22 has CV 0.107: GMSK, and impossible for the original's OOK.
* indices 19/20 have spectral asymmetry exactly 0.000, as DSB must.
* independent confirmation: the M-th power moment peaks at M = 2, 4, 8 for indices 3, 4, 5,
  i.e. BPSK, QPSK, 8PSK. (16PSK/32PSK are inconclusive, not contradictory - ~100 symbols per
  frame and RadioML's deliberate carrier offset destroy 16th/32nd-order moments.)

`CLASSES` below is therefore the FIXED ordering. **DeepSig's shipped `classes.txt` is wrong
and must not be used.**

Label overlap with RadioFry
---------------------------
RadioFry's production checkpoint is 8-class digital. Five have an exact counterpart here:

    BPSK -> BPSK, QPSK -> QPSK, 8PSK -> 8PSK, 16QAM -> QAM16, 64QAM -> QAM64

Deliberately NOT mapped:

* `GMSK` -> `GFSK`. Both use a Gaussian frequency pulse, but GMSK is CPM with h = 0.5 and
  RadioFry's GFSK is a separate generator configuration. Same decision as for Dataset 1.
* `4ASK` -> `PAM4`. RadioML's 4ASK measures a DC fraction of 68%, i.e. unipolar levels;
  RadioFry's PAM4 is bipolar. A direct generator-to-generator comparison was attempted and
  the probe was broken (it subtracted the per-frame mean, destroying the very DC under test),
  so this is **unresolved** and left unmapped rather than assumed.
* `CPFSK` has no counterpart in this label set.

Leakage
-------
Three mechanisms that would make a random split leak were each measured and found absent
(BANK Entry 047):

* SNR levels are independent realisations - frames at matched block offsets across SNR
  correlate no better than shuffled ones or than a phase-randomised surrogate;
* consecutive frames are not contiguous slices of a recording - adjacent (k, k+1) correlation
  equals far-apart equals the null;
* near-duplicates above 0.9 correlation: **0.0% of frames in every class tested**, and zero
  exact duplicates in 9,000 sampled frames.

Because frames within a block carry no temporal order, the split below partitions each block
**contiguously**. That is equivalent to a random partition here and reads sequentially from a
21 GB file instead of seeking per frame.

**Caveat that must travel with any result**: the absence of recording identifiers means
recording-level independence cannot be *verified*, only inferred from the three measurements
above. That is far stronger evidence than Dataset 1 had, but it is not a metadata guarantee.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radiofry.contracts import UnifiedSignalContainer

FRAME_SAMPLES = 1024
FRAMES_PER_BLOCK = 4096
SNR_LEVELS: tuple[int, ...] = tuple(range(-20, 32, 2))

# The FIXED ordering. See the module docstring for the measurements that establish it.
CLASSES: tuple[str, ...] = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
)

# The ordering DeepSig ships, kept only so a caller can assert it is NOT being used.
SHIPPED_WRONG_ORDER: tuple[str, ...] = (
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK", "BPSK", "8PSK",
    "AM-SSB-SC", "4ASK", "16PSK", "64APSK", "128QAM", "128APSK", "AM-DSB-SC", "AM-SSB-WC",
    "64QAM", "QPSK", "256QAM", "AM-DSB-WC", "OOK", "16QAM",
)

LABEL_MAP = {"BPSK": "BPSK", "QPSK": "QPSK", "8PSK": "8PSK",
             "16QAM": "QAM16", "64QAM": "QAM64"}

# Split fractions applied inside every (class, SNR) block, so the split is stratified over
# class and SNR by construction.
TRAIN_FRACTION, VALIDATION_FRACTION = 0.70, 0.15
SEALED_SPLIT = "test"

_SPLITS = ("train", "val", "test")


def block_start(class_index: int, snr_db: int) -> int:
    """First frame of one (class, SNR) block. Verified against the stored labels."""

    if not 0 <= class_index < len(CLASSES):
        raise ValueError(f"class_index must be in 0..{len(CLASSES)-1}")
    if snr_db not in SNR_LEVELS:
        raise ValueError(f"snr_db must be one of {SNR_LEVELS}")
    return (class_index * len(SNR_LEVELS) + SNR_LEVELS.index(snr_db)) * FRAMES_PER_BLOCK


def block_bounds(split: str) -> tuple[int, int]:
    """Offsets within a 4096-frame block belonging to one split.

    Contiguous rather than random: the frames inside a block carry no temporal order
    (measured), so a contiguous partition is equivalent and reads sequentially.
    """

    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}")
    train_end = int(FRAMES_PER_BLOCK * TRAIN_FRACTION)
    validation_end = train_end + int(FRAMES_PER_BLOCK * VALIDATION_FRACTION)
    return {"train": (0, train_end),
            "val": (train_end, validation_end),
            "test": (validation_end, FRAMES_PER_BLOCK)}[split]


@dataclass(frozen=True)
class RadioMLSubset:
    """Frames held as float32 I/Q pairs, with their labels and provenance."""

    split: str
    raw: np.ndarray                 # (N, 1024, 2) float32 or float16
    modulation: np.ndarray          # (N,) dataset class names
    radiofry_label: np.ndarray      # (N,) production label, or "" when unmapped
    snr_db: np.ndarray              # (N,)
    indices: np.ndarray             # positions in the source file
    source: str = "radioml2018.01a"

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    @property
    def mapped(self) -> np.ndarray:
        return self.radiofry_label != ""

    def complex_frames(self, rows: np.ndarray | None = None) -> np.ndarray:
        block = self.raw if rows is None else self.raw[rows]
        block = block.astype(np.float32, copy=False)
        return (block[..., 0] + 1j * block[..., 1]).astype(np.complex64)

    def container(self, index: int, sample_rate_hz: float = 1.0) -> UnifiedSignalContainer:
        """One frame as the pipeline's signal type.

        RadioML publishes no absolute sample rate; the default of 1.0 makes every derived
        frequency a fraction of fs, which is honest rather than inventing a rate.
        """
        return UnifiedSignalContainer(
            self.complex_frames(np.array([index]))[0], sample_rate_hz, "iq",
            {"source": self.source, "snr_db": float(self.snr_db[index]),
             "modulation": str(self.modulation[index])})


@contextlib.contextmanager
def open_dataset(path: str | Path):
    """Read-only handle to the HDF5 file, or to the directory containing it."""

    import h5py

    target = Path(path)
    if target.is_dir():
        candidates = sorted(target.glob("*.hdf5")) + sorted(target.glob("*.h5"))
        if not candidates:
            raise FileNotFoundError(f"no HDF5 file under {target}")
        target = candidates[0]
    with h5py.File(target, "r") as handle:
        yield handle


def verify_layout(path: str | Path, *, samples: int = 24) -> bool:
    """Assert the block arithmetic against the stored labels before anything relies on it."""

    rng = np.random.default_rng(0)
    with open_dataset(path) as handle:
        labels, snrs = handle["Y"], handle["Z"]
        for _ in range(samples):
            index = int(rng.integers(0, len(CLASSES)))
            snr = int(rng.choice(SNR_LEVELS))
            start = block_start(index, snr)
            if int(np.argmax(labels[start])) != index or int(snrs[start][0]) != snr:
                return False
    return True


def select_indices(path: str | Path, split: str, *,
                   classes: tuple[str, ...] | None = None,
                   snr_db: tuple[int, ...] | None = None,
                   per_config: int | None = None,
                   seed: int = 0) -> np.ndarray:
    """Frame positions for one split, stratified over (class, SNR) by construction.

    `per_config` caps how many frames each (class, SNR) contributes, which keeps a pool
    balanced and bounded. Selection within a block is seeded and reproducible.
    """

    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}")
    wanted = classes or CLASSES
    unknown = set(wanted) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown classes: {sorted(unknown)}")
    levels = snr_db or SNR_LEVELS

    low, high = block_bounds(split)
    available = high - low
    take = available if per_config is None else min(per_config, available)
    rng = np.random.default_rng(seed)

    chosen = []
    for name in wanted:
        index = CLASSES.index(name)
        for snr in levels:
            start = block_start(index, snr) + low
            offsets = (np.arange(take) if take == available
                       else np.sort(rng.choice(available, size=take, replace=False)))
            chosen.append(start + offsets)
    return np.sort(np.concatenate(chosen)).astype(np.int64)


READ_CHUNK = 20_000


def load_indices(path: str | Path, split: str, indices: np.ndarray, *,
                 dtype: str = "float16") -> RadioMLSubset:
    """Load exactly these frame positions. `float16` halves the memory of a large pool.

    Read in chunks and cast as we go. `X` is stored float32, so the obvious
    `handle["X"][indices].astype("float16")` would first materialise the whole selection at
    float32 - twice the memory of the result, on top of the result itself. For a 374k-frame
    pool that is a 3.1 GB temporary beside a 1.5 GB destination, which is enough to exhaust
    memory on a 16 GB machine and have the run killed with no traceback.
    """

    indices = np.sort(np.asarray(indices, dtype=np.int64))
    with open_dataset(path) as handle:
        source = handle["X"]
        raw = np.empty((indices.size, source.shape[1], source.shape[2]), dtype=dtype)
        for start in range(0, indices.size, READ_CHUNK):
            block = indices[start:start + READ_CHUNK]
            raw[start:start + block.size] = source[block].astype(dtype, copy=False)
        label_index = np.argmax(handle["Y"][indices], axis=1)
        snr = np.asarray(handle["Z"][indices]).ravel()

    modulation = np.array([CLASSES[int(i)] for i in label_index])
    mapped = np.array([LABEL_MAP.get(name, "") for name in modulation])
    return RadioMLSubset(split, raw, modulation, mapped, snr, indices)


def inventory(path: str | Path) -> dict:
    """Counts and ranges, without loading the sample array."""

    with open_dataset(path) as handle:
        labels = np.argmax(handle["Y"][:], axis=1)
        snrs = np.asarray(handle["Z"][:]).ravel()
        frames = int(handle["X"].shape[0])
        shape = tuple(int(v) for v in handle["X"].shape)
        dtype = str(handle["X"].dtype)

    per_class = {CLASSES[i]: int((labels == i).sum()) for i in range(len(CLASSES))}
    per_snr = {int(s): int((snrs == s).sum()) for s in sorted(set(snrs.tolist()))}
    pairs = {(int(a), int(b)) for a, b in zip(labels.tolist(), snrs.tolist())}
    return {
        "frames": frames, "shape": shape, "dtype": dtype,
        "classes": list(CLASSES), "class_counts": per_class,
        "snr_levels": sorted(per_snr), "snr_counts": per_snr,
        "configurations": len(pairs),
        "balanced": len(set(per_class.values())) == 1,
        "splits": {name: block_bounds(name) for name in _SPLITS},
        "mapped_to_radiofry": dict(LABEL_MAP),
    }


def index_digest(indices: np.ndarray) -> str:
    """Content address for a selection, so a run is reproducible frame for frame."""

    import hashlib

    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def split_manifest(path: str | Path) -> str:
    """A stable description of the split protocol, for the metrics record."""

    return json.dumps({
        "protocol": "contiguous partition within each (class, SNR) block",
        "fractions": {"train": TRAIN_FRACTION, "val": VALIDATION_FRACTION,
                      "test": round(1 - TRAIN_FRACTION - VALIDATION_FRACTION, 4)},
        "bounds": {name: block_bounds(name) for name in _SPLITS},
        "sealed": SEALED_SPLIT,
        "justification": "frames within a block carry no temporal order (measured)",
    }, sort_keys=True)
