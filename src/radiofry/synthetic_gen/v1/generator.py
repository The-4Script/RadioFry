"""Capture and dataset orchestration for Synthetic Dataset V1.

The pipeline is deliberately linear and impairment-free:

    known bits -> modulation -> clean baseband -> AWGN -> IQ/WAV -> ground truth

Every capture is written alongside a JSON ground-truth record and the exact bit
arrays that produced it, so a later harness can score the existing RadioFry
pipeline against known values instead of against its own estimates.
"""

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radiofry.ingestion.iq_parser import IQFormat

from .channel import add_awgn
from .config import (
    DEFAULT_FSK_MODULATION_INDICES,
    DEFAULT_SNR_SWEEP_DB,
    KNOWN_HARD_FSK_MODULATION_INDICES,
    KNOWN_HARD_FSK_REASON,
    MODULATIONS,
    V1_IMPAIRMENTS,
    SampleSpec,
)
from .modulation import generate_source_bits, modulate
from .writers import DEFAULT_FULL_SCALE_FRACTION, write_iq_file, write_wav_file

GENERATOR_NAME = "radiofry.synthetic_gen.v1"
GENERATOR_VERSION = "1.0.0"
CAPTURE_SCHEMA = "radiofry.synthetic.v1"
DATASET_SCHEMA = "radiofry.synthetic.v1.dataset"
SUPPORTED_FORMATS = ("iq", "wav")

MANIFEST_FIELDS = [
    "capture_id",
    "modulation",
    "radiofry_label",
    "order",
    "bits_per_symbol",
    "sample_rate_hz",
    "symbol_rate_hz",
    "samples_per_symbol",
    "fsk_modulation_index",
    "fsk_deviation_hz",
    "known_hard",
    "num_symbols",
    "num_samples",
    "noise_type",
    "target_snr_db",
    "realized_snr_db",
    "seed",
    "bits_seed",
    "iq_file",
    "wav_file",
    "ground_truth_file",
    "source_bits_file",
]


def capture_id_for(
    modulation: str,
    snr_db: float | None,
    replicate: int,
    *,
    fsk_modulation_index: float | None = None,
) -> str:
    """Stable, human-readable identifier for one capture.

    Frequency modulations carry their modulation index so the swept arms are
    distinguishable on disk; every other modulation keeps its original id.
    """

    tag = "noiseless" if snr_db is None else f"snr{snr_db:g}dB"
    index = "" if fsk_modulation_index is None else f"_h{round(float(fsk_modulation_index), 6)}"
    return f"{modulation}{index}_{tag}_r{replicate:03d}"


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a reproducible child seed that does not depend on iteration order."""

    key = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**31)


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def generate_sample(
    spec: SampleSpec,
    output_dir: str | Path,
    capture_id: str,
    *,
    formats: tuple[str, ...] = SUPPORTED_FORMATS,
    iq_format: IQFormat | None = None,
    wav_dtype: str = "int16",
    full_scale_fraction: float = DEFAULT_FULL_SCALE_FRACTION,
) -> dict[str, Any]:
    """Generate one capture plus its ground-truth record; returns the record."""

    unsupported = set(formats) - set(SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(f"unsupported output formats: {sorted(unsupported)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_bits = generate_source_bits(spec)
    transmitted_bits = source_bits  # V1 applies no FEC and no interleaving
    clean = modulate(transmitted_bits, spec)
    noisy, noise = add_awgn(clean, spec.snr_db, np.random.default_rng([spec.seed, 2]))

    source_name = f"{capture_id}.source_bits.npy"
    transmitted_name = f"{capture_id}.transmitted_bits.npy"
    np.save(output / source_name, source_bits)
    np.save(output / transmitted_name, transmitted_bits)

    files: list[dict[str, Any]] = []
    if "iq" in formats:
        report = write_iq_file(
            output / f"{capture_id}.iq",
            noisy,
            fmt=iq_format or IQFormat(),
            full_scale_fraction=full_scale_fraction,
        )
        files.append(_file_entry(report))
    if "wav" in formats:
        report = write_wav_file(
            output / f"{capture_id}.wav",
            noisy,
            spec.sample_rate_hz,
            dtype=wav_dtype,
            full_scale_fraction=full_scale_fraction,
        )
        files.append(_file_entry(report))

    # SNR is defined over the full sampled band, so oversampling raises the
    # per-symbol energy ratio by 10*log10(samples_per_symbol).
    es_n0_db = (
        None
        if noise.target_snr_db is None
        else float(noise.target_snr_db + 10 * np.log10(spec.samples_per_symbol))
    )
    is_known_hard = spec.fsk_modulation_index is not None and any(
        abs(spec.fsk_modulation_index - value) < 1e-9 for value in KNOWN_HARD_FSK_MODULATION_INDICES
    )
    truth = {
        "schema": CAPTURE_SCHEMA,
        "schema_version": 1,
        "capture_id": capture_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "modulation": {
            "name": spec.modulation,
            "family": spec.modulation_spec.family,
            "order": spec.modulation_spec.order,
            "bits_per_symbol": spec.bits_per_symbol,
            "radiofry_label": spec.modulation_spec.radiofry_label,
            "bit_mapping": "natural_binary_msb_first",
            "constellation_normalization": "unit_average_symbol_power",
            "pulse_shape": spec.pulse_shape,
        },
        "signal": {
            "sample_rate_hz": spec.sample_rate_hz,
            "symbol_rate_hz": spec.symbol_rate_hz,
            "samples_per_symbol": spec.samples_per_symbol,
            "num_symbols": spec.num_symbols,
            "num_samples": spec.num_samples,
            "duration_sec": spec.duration_sec,
            "fsk_deviation_hz": spec.fsk_deviation_hz if spec.modulation_spec.family == "fsk" else None,
            "fsk_modulation_index": spec.fsk_modulation_index,
            "center_frequency_hz": 0.0,
        },
        "noise": {
            "noise_type": noise.noise_type,
            "target_snr_db": noise.target_snr_db,
            "realized_snr_db": noise.realized_snr_db,
            "snr_definition": noise.snr_definition,
            "es_n0_db": es_n0_db,
            "eb_n0_db": None if es_n0_db is None else es_n0_db - 10 * np.log10(spec.bits_per_symbol),
            "signal_power": noise.signal_power,
            "target_noise_power": noise.target_noise_power,
            "realized_noise_power": noise.realized_noise_power,
        },
        "known_hard": is_known_hard,
        "known_hard_reason": KNOWN_HARD_FSK_REASON if is_known_hard else "",
        "impairments": dict(V1_IMPAIRMENTS),
        "bits": {
            "num_source_bits": int(source_bits.size),
            "num_transmitted_bits": int(transmitted_bits.size),
            "source_bits_file": source_name,
            "transmitted_bits_file": transmitted_name,
            "source_bits_sha256": _sha256(source_bits),
            "transmitted_bits_sha256": _sha256(transmitted_bits),
            "source_equals_transmitted": True,
        },
        "seeds": {
            "seed": spec.seed,
            "bits_seed": spec.effective_bits_seed,
            "bits_stream": "numpy.random.default_rng([bits_seed, 1])",
            "noise_stream": "numpy.random.default_rng([seed, 2])",
        },
        "files": files,
    }
    (output / f"{capture_id}.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def _file_entry(report: Any) -> dict[str, Any]:
    entry = asdict(report)
    entry["filename"] = Path(entry.pop("path")).name
    return entry


def load_ground_truth(json_path: str | Path) -> dict[str, Any]:
    """Load a ground-truth record together with its bit arrays."""

    path = Path(json_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    return {
        "metadata": metadata,
        "source_bits": np.load(base / metadata["bits"]["source_bits_file"]),
        "transmitted_bits": np.load(base / metadata["bits"]["transmitted_bits_file"]),
    }


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for a full V1 sweep."""

    output_dir: str | Path
    modulations: tuple[str, ...] = tuple(MODULATIONS)
    snr_sweep_db: tuple[float | None, ...] = DEFAULT_SNR_SWEEP_DB
    fsk_modulation_indices: tuple[float, ...] = DEFAULT_FSK_MODULATION_INDICES
    captures_per_condition: int = 1
    num_symbols: int = 4096
    samples_per_symbol: int | None = 8
    sample_rate_hz: float | None = 200_000.0
    symbol_rate_hz: float | None = None
    seed: int = 2026
    formats: tuple[str, ...] = SUPPORTED_FORMATS
    iq_dtype: str = "int16"
    iq_byte_order: str = "little"
    wav_dtype: str = "int16"
    full_scale_fraction: float = DEFAULT_FULL_SCALE_FRACTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "modulations", tuple(self.modulations))
        object.__setattr__(self, "snr_sweep_db", tuple(self.snr_sweep_db))
        object.__setattr__(self, "formats", tuple(self.formats))
        object.__setattr__(self, "fsk_modulation_indices", tuple(self.fsk_modulation_indices))
        if not self.fsk_modulation_indices:
            raise ValueError("at least one FSK modulation index is required")
        if any(value <= 0 for value in self.fsk_modulation_indices):
            raise ValueError("FSK modulation indices must be positive")
        unknown = [name for name in self.modulations if name not in MODULATIONS]
        if unknown:
            raise ValueError(f"unsupported modulation(s) {unknown}; V1 supports {sorted(MODULATIONS)}")
        if not self.modulations:
            raise ValueError("at least one modulation is required")
        if not self.snr_sweep_db:
            raise ValueError("at least one SNR point is required")
        if self.captures_per_condition < 1:
            raise ValueError("captures_per_condition must be positive")
        unsupported = set(self.formats) - set(SUPPORTED_FORMATS)
        if unsupported:
            raise ValueError(f"unsupported output formats: {sorted(unsupported)}")


def generate_dataset(spec: DatasetSpec) -> Path:
    """Generate the full sweep and return the path of the written manifest."""

    output = Path(spec.output_dir)
    captures_dir = output / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    iq_format = IQFormat(spec.iq_dtype, spec.iq_byte_order)
    rows: list[dict[str, Any]] = []

    for modulation in spec.modulations:
        is_fsk = MODULATIONS[modulation].family == "fsk"
        indices: tuple[float | None, ...] = spec.fsk_modulation_indices if is_fsk else (None,)
        for replicate in range(spec.captures_per_condition):
            bits_seed = derive_seed(spec.seed, modulation, "bits", replicate)
            for index in indices:
                for snr_db in spec.snr_sweep_db:
                    sample_spec = SampleSpec(
                        modulation=modulation,
                        num_symbols=spec.num_symbols,
                        samples_per_symbol=spec.samples_per_symbol,
                        sample_rate_hz=spec.sample_rate_hz,
                        symbol_rate_hz=spec.symbol_rate_hz,
                        snr_db=snr_db,
                        # The seed deliberately excludes the index, so the swept FSK arms
                        # share one payload and one noise realisation and differ only in h.
                        seed=derive_seed(spec.seed, modulation, snr_db, replicate),
                        bits_seed=bits_seed,
                        fsk_modulation_index=index,
                    )
                    capture_id = capture_id_for(
                        modulation, snr_db, replicate, fsk_modulation_index=index
                    )
                    truth = generate_sample(
                        sample_spec,
                        captures_dir,
                        capture_id,
                        formats=spec.formats,
                        iq_format=iq_format,
                        wav_dtype=spec.wav_dtype,
                        full_scale_fraction=spec.full_scale_fraction,
                    )
                    rows.append(_manifest_row(truth))

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": DATASET_SCHEMA,
        "schema_version": 1,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "capture_count": len(rows),
        "impairments": dict(V1_IMPAIRMENTS),
        "config": {
            "modulations": list(spec.modulations),
            "snr_sweep_db": list(spec.snr_sweep_db),
            "fsk_modulation_indices": list(spec.fsk_modulation_indices),
            "captures_per_condition": spec.captures_per_condition,
            "num_symbols": spec.num_symbols,
            "samples_per_symbol": spec.samples_per_symbol,
            "sample_rate_hz": spec.sample_rate_hz,
            "symbol_rate_hz": spec.symbol_rate_hz,
            "seed": spec.seed,
            "formats": list(spec.formats),
            "iq_dtype": spec.iq_dtype,
            "iq_byte_order": spec.iq_byte_order,
            "wav_dtype": spec.wav_dtype,
            "full_scale_fraction": spec.full_scale_fraction,
        },
        "capture_ids": [row["capture_id"] for row in rows],
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path


def _manifest_row(truth: dict[str, Any]) -> dict[str, Any]:
    by_format = {entry["file_format"]: entry["filename"] for entry in truth["files"]}
    capture_id = truth["capture_id"]
    return {
        "capture_id": capture_id,
        "modulation": truth["modulation"]["name"],
        "radiofry_label": truth["modulation"]["radiofry_label"],
        "order": truth["modulation"]["order"],
        "bits_per_symbol": truth["modulation"]["bits_per_symbol"],
        "sample_rate_hz": truth["signal"]["sample_rate_hz"],
        "symbol_rate_hz": truth["signal"]["symbol_rate_hz"],
        "samples_per_symbol": truth["signal"]["samples_per_symbol"],
        "fsk_modulation_index": "" if truth["signal"]["fsk_modulation_index"] is None
        else truth["signal"]["fsk_modulation_index"],
        "fsk_deviation_hz": "" if truth["signal"]["fsk_deviation_hz"] is None
        else truth["signal"]["fsk_deviation_hz"],
        "known_hard": truth["known_hard"],
        "num_symbols": truth["signal"]["num_symbols"],
        "num_samples": truth["signal"]["num_samples"],
        "noise_type": truth["noise"]["noise_type"],
        "target_snr_db": "" if truth["noise"]["target_snr_db"] is None else truth["noise"]["target_snr_db"],
        "realized_snr_db": "" if truth["noise"]["realized_snr_db"] is None else truth["noise"]["realized_snr_db"],
        "seed": truth["seeds"]["seed"],
        "bits_seed": truth["seeds"]["bits_seed"],
        "iq_file": f"captures/{by_format['iq']}" if "iq" in by_format else "",
        "wav_file": f"captures/{by_format['wav']}" if "wav" in by_format else "",
        "ground_truth_file": f"captures/{capture_id}.json",
        "source_bits_file": f"captures/{truth['bits']['source_bits_file']}",
    }
