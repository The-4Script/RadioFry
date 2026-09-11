"""Read-only adapter for the real-world multipath IQ dataset (Belousov & Ronkin, 2026).

The raw data is never extracted, modified or copied. A zip archive is read from inside into
memory; an extracted directory is opened with h5py in mode 'r'. Nothing here writes anything.

Dataset facts, taken from the archive itself rather than from the paper
----------------------------------------------------------------------
* `X` is `(N, 1024, 2)` float16 - I and Q as the last axis.
* `y_mod` int16, `y_chan` int8, `y_snr` int16.
* The label mapping is an HDF5 file attribute, `mod2id_json`:
  `{"BPSK": 0, "QPSK": 1, "QAM": 2, "GMSK": 3, "OFDM": 4, "NBFM": 5, "WBFM": 6}`.
  It is read from the file, never hard-coded, so a differently-encoded release cannot be
  silently mis-labelled.
* Sampling rate 2 MSps, carrier 2.4 GHz, frames of 1024 samples, SNR 20-30 dB in 2 dB steps,
  RRC pulse shaping with alpha = 0.35, and `QAM` means specifically 16-QAM.
* Released as train / val / test of 400k / 80k / 80k frames.

Where the released data differs from its own paper (all measured, see BANK Entry 045)
-------------------------------------------------------------------------------------
* Symbol rate is ~199.2 kHz, i.e. **10 samples per symbol** at 2 MSps. Table 1 of the paper
  says 100 kSps at 4 samples/symbol. The measured cyclic line sits 33 dB over the noise
  floor at the same FFT bin for BPSK, QPSK and QAM alike, and the occupied bandwidth agrees
  with it, so the measurement is used.
* The frames are **not** power-normalised, contrary to the paper's preprocessing section.
  Median frame power ranges over 23 dB between classes, which is a usable shortcut for any
  model that sees absolute amplitude. RadioFry's production contract normalises per window,
  so it does not and cannot use that shortcut.
* Released frame counts are 560,000, not the 840,000 the paper describes, and the per
  configuration counts vary from 753 to 1,179 rather than being balanced.

Label-space overlap with RadioFry
---------------------------------
RadioFry's production checkpoint is 8-class and digital-only. Only three dataset classes
have an exact counterpart:

    BPSK -> BPSK,  QPSK -> QPSK,  QAM (16-QAM) -> QAM16

`GMSK` is deliberately **not** mapped to `GFSK`: both use a Gaussian frequency pulse at
BT = 0.3, but GMSK is a specific CPM with h = 0.5 and RadioFry's GFSK is a separate
generator configuration. Calling them the same class would manufacture agreement.
`OFDM`, `NBFM` and `WBFM` have no counterpart in an 8-class digital label set.

Frames of unmapped classes are therefore **out of label space**, not errors: the model
cannot be right about a class it has no output for, and scoring them as mistakes would
understate it just as badly as hiding them would overstate it.
"""

from __future__ import annotations

import contextlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radiofry.contracts import UnifiedSignalContainer

SAMPLE_RATE_HZ = 2_000_000.0
CARRIER_HZ = 2.4e9
FRAME_SAMPLES = 1024
SUBSETS = ("train", "val", "test")

# The seven classes the dataset itself labels, in its own `mod2id_json` order.
DATASET_CLASSES: tuple[str, ...] = ("BPSK", "QPSK", "QAM", "GMSK", "OFDM", "NBFM", "WBFM")

# Dataset class -> RadioFry production label. Only exact counterparts appear here.
LABEL_MAP = {"BPSK": "BPSK", "QPSK": "QPSK", "QAM": "QAM16"}
UNMAPPED = ("GMSK", "OFDM", "NBFM", "WBFM")

# Measured, not assumed: the |s|^2 cyclic line sits at ~199.2 kHz for every linearly
# modulated class, giving 10 samples per symbol at 2 MSps.
MEASURED_SYMBOL_RATE_HZ = 199_219.0
MEASURED_SAMPLES_PER_SYMBOL = 10

# --- split protocol -------------------------------------------------------------------------
#
# Each of the 84 (modulation, channel, SNR) combinations is a separate .dat recording, and the
# released splits were drawn by random frame-level sampling across all of them. Every
# configuration therefore appears in all three splits, and near-duplicate frames straddle
# them: measured at 48% of QAM test frames, 10% BPSK, 8% QPSK, 4% GMSK having a partner in
# train above 0.9 correlation (DC removed, peak over lags, against a phase-randomised
# surrogate null of ~0.30).
#
# The same measurement across recording boundaries is 0.0% for every digital class. Holding
# out whole SNR levels therefore yields a genuinely capture-disjoint split, which is the only
# honest way to measure generalisation on this dataset.
TRAIN_SNR_DB: tuple[int, ...] = (20, 24, 28)
HELDOUT_SNR_DB: tuple[int, ...] = (22, 26, 30)

# The split that must never be read during training or model selection.
SEALED_SPLIT = "test"


@dataclass(frozen=True)
class RealWorldSubset:
    """One split, with its labels decoded and its provenance carried."""

    name: str
    iq: np.ndarray                  # (N, 1024) complex64
    modulation: np.ndarray          # (N,) dataset class names
    radiofry_label: np.ndarray      # (N,) production label, or "" when unmapped
    channel: np.ndarray             # (N,) 0 clean, 1 multipath
    snr_db: np.ndarray              # (N,)
    sample_rate_hz: float = SAMPLE_RATE_HZ
    source: str = "realworld_multipath"
    indices: np.ndarray | None = None   # positions in the source file, for reproducibility

    def __len__(self) -> int:
        return int(self.iq.shape[0])

    @property
    def mapped(self) -> np.ndarray:
        """Boolean mask of frames whose class RadioFry can actually predict."""
        return self.radiofry_label != ""

    def container(self, index: int) -> UnifiedSignalContainer:
        """One frame as the pipeline's own signal type, at the dataset's sample rate."""
        return UnifiedSignalContainer(
            self.iq[index], SAMPLE_RATE_HZ, "iq",
            {"source": self.source, "center_frequency_hz": CARRIER_HZ,
             "channel": int(self.channel[index]), "snr_db": float(self.snr_db[index])})


def _member(archive: zipfile.ZipFile, subset: str) -> str:
    wanted = f"subset_{subset}.h5"
    for name in archive.namelist():
        if name.endswith(wanted):
            return name
    raise FileNotFoundError(f"{wanted} is not in the archive")


@contextlib.contextmanager
def open_split(source: str | Path, subset: str):
    """Yield a read-only h5py handle for one split, from a directory or a zip archive.

    A directory is opened in place with mode 'r'. A zip is read into memory and opened
    through a file-like object; it is never extracted to disk.
    """

    import h5py

    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    path = Path(source)

    if path.is_dir():
        candidates = [path / f"subset_{subset}.h5",
                      path / "dataset" / f"subset_{subset}.h5"]
        for candidate in candidates:
            if candidate.is_file():
                with h5py.File(candidate, "r") as handle:
                    yield handle
                return
        raise FileNotFoundError(f"subset_{subset}.h5 is not under {path}")

    with zipfile.ZipFile(path) as archive:
        payload = archive.read(_member(archive, subset))
    with h5py.File(io.BytesIO(payload), "r") as handle:
        yield handle


def label_mapping(source: str | Path, subset: str = "test") -> dict[str, int]:
    """The dataset's own class -> id mapping, read from the file attribute."""

    with open_split(source, subset) as handle:
        return json.loads(handle.attrs["mod2id_json"])


def select_indices(source: str | Path, subset: str, *,
                   classes: tuple[str, ...] | None = None,
                   channels: tuple[int, ...] | None = None,
                   snr_db: tuple[int, ...] | None = None,
                   limit: int | None = None,
                   per_class_limit: int | None = None,
                   per_config_limit: int | None = None,
                   seed: int = 0) -> np.ndarray:
    """Positions of frames matching the filters, sampled reproducibly.

    Sampling is uniform at random with a fixed seed rather than a prefix, because the file is
    ordered by configuration and a prefix would be one modulation at one SNR.

    `per_class_limit` takes that many frames from each dataset class. `per_config_limit`
    takes that many from each (class, channel, SNR) recording, which additionally stratifies
    over channel condition and SNR. Either keeps a training set balanced despite the released
    data not being: measured per-configuration counts range from 753 to 1,179.
    """

    with open_split(source, subset) as handle:
        mapping = json.loads(handle.attrs["mod2id_json"])
        identifier_to_name = {int(v): k for k, v in mapping.items()}
        names = np.array([identifier_to_name[int(v)] for v in handle["y_mod"][:]])
        channel_values = np.asarray(handle["y_chan"][:])
        snr_values = np.asarray(handle["y_snr"][:])

    keep = np.ones(names.size, dtype=bool)
    if classes is not None:
        keep &= np.isin(names, np.asarray(classes))
    if channels is not None:
        keep &= np.isin(channel_values, np.asarray(channels))
    if snr_db is not None:
        keep &= np.isin(snr_values, np.asarray(snr_db))

    rng = np.random.default_rng(seed)

    if per_config_limit is not None:
        configuration = np.array([f"{m}|{c}|{s}" for m, c, s in
                                  zip(names, channel_values.tolist(), snr_values.tolist())])
        chosen = []
        for key in sorted(set(configuration[keep].tolist())):
            pool = np.flatnonzero(keep & (configuration == key))
            chosen.append(rng.choice(pool, size=min(per_config_limit, pool.size),
                                     replace=False))
        return np.sort(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)

    if per_class_limit is not None:
        chosen = []
        for name in sorted(set(names[keep].tolist())):
            pool = np.flatnonzero(keep & (names == name))
            chosen.append(rng.choice(pool, size=min(per_class_limit, pool.size),
                                     replace=False))
        return np.sort(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)

    indices = np.flatnonzero(keep)
    if limit is not None and indices.size > limit:
        indices = rng.choice(indices, size=limit, replace=False)
    return np.sort(indices)


def load_indices(source: str | Path, subset: str, indices: np.ndarray) -> RealWorldSubset:
    """Load exactly the given frame positions."""

    indices = np.sort(np.asarray(indices, dtype=np.int64))
    with open_split(source, subset) as handle:
        mapping = json.loads(handle.attrs["mod2id_json"])
        identifier_to_name = {int(v): k for k, v in mapping.items()}
        raw = handle["X"][indices]                       # (n, 1024, 2) float16
        names = np.array([identifier_to_name[int(v)] for v in handle["y_mod"][indices]])
        channel = np.asarray(handle["y_chan"][indices])
        snr = np.asarray(handle["y_snr"][indices])

    # float16 -> complex64. The dataset's own amplitude scale is left alone; the production
    # contract normalises per window downstream.
    iq = (raw[..., 0].astype(np.float32)
          + 1j * raw[..., 1].astype(np.float32)).astype(np.complex64)
    labels = np.array([LABEL_MAP.get(name, "") for name in names])
    return RealWorldSubset(subset, iq, names, labels, channel, snr, indices=indices)


def load_subset(archive_path: str | Path, subset: str = "test", *,
                limit: int | None = None,
                classes: tuple[str, ...] | None = None,
                channels: tuple[int, ...] | None = None,
                snr_db: tuple[int, ...] | None = None,
                per_class_limit: int | None = None,
                per_config_limit: int | None = None,
                seed: int = 0) -> RealWorldSubset:
    """Load one split read-only from a directory or a zip archive."""

    indices = select_indices(archive_path, subset, classes=classes, channels=channels,
                             snr_db=snr_db, limit=limit, per_class_limit=per_class_limit,
                             per_config_limit=per_config_limit, seed=seed)
    return load_indices(archive_path, subset, indices)


def inventory(archive_path: str | Path) -> dict:
    """Counts and ranges for every split, without loading the sample arrays."""

    summary: dict[str, dict] = {}
    for subset in SUBSETS:
        try:
            with open_split(archive_path, subset) as handle:
                mapping = json.loads(handle.attrs["mod2id_json"])
                identifier_to_name = {int(v): k for k, v in mapping.items()}
                modulation_ids = handle["y_mod"][:]
                names = [identifier_to_name[int(value)] for value in modulation_ids]
                channels = handle["y_chan"][:]
                snrs = handle["y_snr"][:]
                configurations = {
                    f"{m}|{c}|{s}" for m, c, s in zip(names, channels.tolist(), snrs.tolist())}
                summary[subset] = {
                    "frames": int(handle["X"].shape[0]),
                    "frame_samples": int(handle["X"].shape[1]),
                    "dtype": str(handle["X"].dtype),
                    "mapping": mapping,
                    "modulations": {name: int(names.count(name)) for name in mapping},
                    "channels": {int(v): int((channels == v).sum()) for v in (0, 1)},
                    "snr_values": sorted({int(v) for v in snrs}),
                    "configurations": len(configurations),
                }
        except FileNotFoundError:
            continue
    return summary
