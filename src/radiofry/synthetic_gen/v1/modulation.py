"""Bit generation and baseband modulation for Synthetic Dataset V1.

Symbol indices use natural binary, MSB first, which is the convention the
existing RadioFry de-mappers assume (``decoding/demodulators/psk_demod.py`` and
``qam_demod.py``). Constellations are scaled to unit average symbol power.
"""

import numpy as np

from .config import SampleSpec


def constellation_for(modulation: str) -> np.ndarray:
    """Return the unit-average-power constellation indexed by symbol index."""

    from .config import MODULATIONS

    if modulation not in MODULATIONS:
        raise ValueError(f"unsupported modulation {modulation!r}")
    spec = MODULATIONS[modulation]
    if spec.family == "psk":
        return np.exp(2j * np.pi * np.arange(spec.order) / spec.order)
    if spec.family == "qam":
        side = int(round(np.sqrt(spec.order)))
        levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
        points = levels[:, None] + 1j * levels[None, :]
        points = points.ravel()
        return points / np.sqrt(np.mean(np.abs(points) ** 2))
    raise ValueError(f"{modulation} is not a memoryless constellation modulation")


def generate_source_bits(spec: SampleSpec) -> np.ndarray:
    """Draw the reproducible payload bits for one capture."""

    rng = np.random.default_rng([spec.effective_bits_seed, 1])
    return rng.integers(0, 2, size=spec.num_bits, dtype=np.uint8)


def bits_to_symbol_indices(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Pack bits into natural-binary, MSB-first symbol indices."""

    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if values.size % bits_per_symbol:
        raise ValueError("bit count must be a whole number of symbols")
    grouped = values.reshape(-1, bits_per_symbol).astype(np.int64)
    weights = 1 << np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    return grouped @ weights


def modulate(bits: np.ndarray, spec: SampleSpec) -> np.ndarray:
    """Map bits to a rectangular-pulse complex baseband waveform."""

    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if values.size != spec.num_bits:
        raise ValueError(f"expected {spec.num_bits} bits for {spec.num_symbols} symbols, got {values.size}")
    indices = bits_to_symbol_indices(values, spec.bits_per_symbol)
    if spec.modulation_spec.family == "fsk":
        waveform = _modulate_cpfsk(indices, spec)
    else:
        symbols = constellation_for(spec.modulation)[indices]
        waveform = np.repeat(symbols, spec.samples_per_symbol)
    return waveform.astype(np.complex64)


def _modulate_cpfsk(indices: np.ndarray, spec: SampleSpec) -> np.ndarray:
    """Continuous-phase FSK: index k maps to tone (2k - (M-1)) * deviation."""

    order = spec.modulation_spec.order
    tones = (2 * indices - (order - 1)) * spec.fsk_deviation_hz
    per_sample = np.repeat(tones, spec.samples_per_symbol)
    phase = 2 * np.pi * np.cumsum(per_sample) / spec.sample_rate_hz
    return np.exp(1j * phase)
