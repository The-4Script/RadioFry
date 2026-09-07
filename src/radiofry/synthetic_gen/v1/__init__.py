"""Synthetic Dataset V1: controlled, impairment-free captures with ground truth."""

from .channel import NoiseReport, add_awgn
from .config import (
    DEFAULT_FSK_MODULATION_INDEX,
    DEFAULT_FSK_MODULATION_INDICES,
    DEFAULT_SNR_SWEEP_DB,
    KNOWN_HARD_FSK_MODULATION_INDICES,
    MODULATIONS,
    V1_IMPAIRMENTS,
    ModulationSpec,
    SampleSpec,
)
from .generator import (
    DatasetSpec,
    capture_id_for,
    generate_dataset,
    generate_sample,
    load_ground_truth,
)
from .modulation import bits_to_symbol_indices, constellation_for, generate_source_bits, modulate
from .writers import WriteReport, write_iq_file, write_wav_file

__all__ = [
    "DEFAULT_FSK_MODULATION_INDEX",
    "DEFAULT_FSK_MODULATION_INDICES",
    "DEFAULT_SNR_SWEEP_DB",
    "KNOWN_HARD_FSK_MODULATION_INDICES",
    "MODULATIONS",
    "DatasetSpec",
    "ModulationSpec",
    "NoiseReport",
    "SampleSpec",
    "V1_IMPAIRMENTS",
    "WriteReport",
    "add_awgn",
    "bits_to_symbol_indices",
    "capture_id_for",
    "constellation_for",
    "generate_dataset",
    "generate_sample",
    "generate_source_bits",
    "load_ground_truth",
    "modulate",
    "write_iq_file",
    "write_wav_file",
]
