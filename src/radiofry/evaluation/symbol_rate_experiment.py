"""Controlled comparison of symbol-rate spectral features on rect and RRC signals.

Resolves the open caveat from BANK.md Entry 003: the phase-step feature was only
validated on V1's rectangular pulses, while the production `|x|^2` feature is the
textbook method for band-limited (RRC) signals. This module builds equivalent
RRC-shaped signals and scores all three candidate features on both pulse shapes.

Diagnostic only. It imports RadioFry's own `_select_symbol_rate` unchanged and
modifies nothing in the pipeline, the V1 generator, or the V1 dataset. RRC shaping
is applied here, not in the generator.
"""

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.signal import find_peaks, welch

from radiofry.dsp.parameter_estimation import _select_symbol_rate
from radiofry.synthetic_gen.v1 import (
    SampleSpec,
    add_awgn,
    bits_to_symbol_indices,
    constellation_for,
    generate_source_bits,
    modulate,
)

FS = 200_000.0
SPS = 8
NUM_SYMBOLS = 2_048
TRUE_SYMBOL_RATE = FS / SPS
RRC_SPAN_SYMBOLS = 10

MODULATIONS = ["BPSK", "QPSK", "8PSK", "BFSK", "16QAM", "64QAM"]
SNR_SWEEP_DB: list[float | None] = [None, 20.0, 15.0, 10.0, 5.0, 0.0]
SEEDS = [3, 17, 91]
BETAS = [0.20, 0.35]

# The demodulator needs round(fs/est) == 8, i.e. the estimate within about +/-6.7%.
DEMOD_TOLERANCE = 1.0 / 15.0


def _spec(modulation: str, seed: int, snr_db: float | None = None) -> SampleSpec:
    return SampleSpec(
        modulation=modulation,
        num_symbols=NUM_SYMBOLS,
        samples_per_symbol=SPS,
        sample_rate_hz=FS,
        snr_db=snr_db,
        seed=seed,
    )


def source_symbols(modulation: str, *, seed: int) -> np.ndarray:
    """Ideal transmit symbols for a V1 capture (linear modulations only)."""

    spec = _spec(modulation, seed)
    indices = bits_to_symbol_indices(generate_source_bits(spec), spec.bits_per_symbol)
    return constellation_for(modulation)[indices]


def rrc_taps(beta: float, span_symbols: int, samples_per_symbol: int) -> np.ndarray:
    """Unit-energy root-raised-cosine filter, odd length so it has no net delay."""

    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must be in (0, 1]")
    n = np.arange(-span_symbols * samples_per_symbol // 2, span_symbols * samples_per_symbol // 2 + 1)
    t = n / samples_per_symbol
    taps = np.empty(t.size, dtype=np.float64)
    for index, value in enumerate(t):
        if np.isclose(value, 0.0):
            taps[index] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif np.isclose(abs(value), 1.0 / (4.0 * beta)):
            taps[index] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = np.sin(np.pi * value * (1.0 - beta)) + 4.0 * beta * value * np.cos(
                np.pi * value * (1.0 + beta)
            )
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[index] = numerator / denominator
    return taps / np.sqrt(np.sum(taps**2))


def _rrc_linear(modulation: str, beta: float, seed: int) -> np.ndarray:
    symbols = source_symbols(modulation, seed=seed)
    upsampled = np.zeros(symbols.size * SPS, dtype=np.complex128)
    upsampled[::SPS] = symbols
    taps = rrc_taps(beta, RRC_SPAN_SYMBOLS, SPS)
    shaped = np.convolve(upsampled, taps, mode="same")
    return (shaped / np.sqrt(np.mean(np.abs(shaped) ** 2))).astype(np.complex64)


def _rrc_cpfsk(beta: float, seed: int) -> np.ndarray:
    """CPFSK with an RRC-shaped frequency pulse; still constant modulus by construction."""

    spec = _spec("BFSK", seed)
    indices = bits_to_symbol_indices(generate_source_bits(spec), spec.bits_per_symbol)
    tones = (2 * indices - 1) * spec.fsk_deviation_hz
    upsampled = np.zeros(tones.size * SPS, dtype=np.float64)
    upsampled[::SPS] = tones * SPS
    taps = rrc_taps(beta, RRC_SPAN_SYMBOLS, SPS)
    frequency = np.convolve(upsampled, taps / np.sum(taps), mode="same")
    phase = 2.0 * np.pi * np.cumsum(frequency) / FS
    return np.exp(1j * phase).astype(np.complex64)


def build_signal(
    modulation: str,
    *,
    pulse: str,
    beta: float = 0.35,
    snr_db: float | None = None,
    seed: int = 3,
) -> np.ndarray:
    """Build one diagnostic capture; `pulse` is 'rect' (V1 generator) or 'rrc'."""

    if pulse == "rect":
        spec = _spec(modulation, seed)
        clean = modulate(generate_source_bits(spec), spec)
    elif pulse == "rrc":
        clean = _rrc_cpfsk(beta, seed) if modulation == "BFSK" else _rrc_linear(modulation, beta, seed)
    else:
        raise ValueError("pulse must be 'rect' or 'rrc'")
    if snr_db is None:
        return clean
    return add_awgn(clean, snr_db, np.random.default_rng([seed, 2]))[0]


def occupied_bandwidth(iq: np.ndarray, sample_rate: float, fraction: float = 0.99) -> float:
    frequencies, psd = welch(iq, fs=sample_rate, nperseg=min(1024, iq.size), return_onesided=False)
    order = np.argsort(frequencies)
    frequencies, psd = frequencies[order], np.real(psd[order])
    cumulative = np.cumsum(np.maximum(psd, 0.0)) / np.sum(np.maximum(psd, 0.0))
    lower = frequencies[np.searchsorted(cumulative, (1 - fraction) / 2)]
    upper = frequencies[np.searchsorted(cumulative, 1 - (1 - fraction) / 2)]
    return float(abs(upper - lower))


def _instantaneous_frequency(x: np.ndarray) -> np.ndarray:
    return np.diff(np.unwrap(np.angle(x)), prepend=0.0)


FEATURES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "abs_x_squared": lambda x: np.abs(x) ** 2,
    "abs_diff_x_squared": lambda x: np.abs(np.diff(x, prepend=x[0])) ** 2,
    "phase_second_difference_squared": lambda x: np.diff(_instantaneous_frequency(x), prepend=0.0) ** 2,
}


def estimate_with_feature(
    iq: np.ndarray, feature: Callable[[np.ndarray], np.ndarray], sample_rate: float = FS
) -> tuple[float | None, float | None]:
    """parameter_estimation.py:91-102 with only the pre-FFT nonlinearity swapped."""

    centered = iq - np.mean(iq)
    powered = feature(centered)
    spectrum = np.abs(np.fft.rfft(powered - np.mean(powered)))
    rates = np.fft.rfftfreq(powered.size, d=1 / sample_rate)
    spectrum[0] = 0
    peaks, properties = find_peaks(spectrum, prominence=max(float(np.median(spectrum)) * 2.0, 1e-12))
    return _select_symbol_rate(rates, spectrum, peaks, properties.get("prominences", np.array([], float)))


@dataclass(frozen=True)
class Case:
    modulation: str
    pulse: str
    beta: float | None
    snr_db: float | None
    seed: int


def run_experiment(
    *,
    modulations: Sequence[str] = tuple(MODULATIONS),
    snr_sweep_db: Sequence[float | None] = tuple(SNR_SWEEP_DB),
    seeds: Sequence[int] = tuple(SEEDS),
    betas: Sequence[float] = tuple(BETAS),
) -> list[dict]:
    """Score every feature on every (modulation, pulse shape, SNR, seed) combination."""

    shapes: list[tuple[str, float | None]] = [("rect", None)] + [("rrc", beta) for beta in betas]
    records: list[dict] = []
    for modulation in modulations:
        for pulse, beta in shapes:
            for snr_db in snr_sweep_db:
                for seed in seeds:
                    iq = build_signal(
                        modulation, pulse=pulse, beta=beta or 0.35, snr_db=snr_db, seed=seed
                    )
                    row = {
                        "modulation": modulation,
                        "pulse": pulse if beta is None else f"rrc{beta:.2f}",
                        "beta": beta,
                        "snr_db": snr_db,
                        "seed": seed,
                        "true_symbol_rate_hz": TRUE_SYMBOL_RATE,
                    }
                    for name, feature in FEATURES.items():
                        rate, confidence = estimate_with_feature(iq, feature)
                        error = None if rate is None else (rate - TRUE_SYMBOL_RATE) / TRUE_SYMBOL_RATE
                        row[f"{name}__rate_hz"] = rate
                        row[f"{name}__rel_error"] = error
                        row[f"{name}__confidence"] = confidence
                        row[f"{name}__within_1pct"] = error is not None and abs(error) <= 0.01
                        row[f"{name}__demod_usable"] = (
                            error is not None and abs(error) <= DEMOD_TOLERANCE
                        )
                    records.append(row)
    return records


def summarise(records: Sequence[dict]) -> dict:
    """Aggregate hit-rates overall and by pulse shape, modulation and SNR."""

    def rates(subset: Sequence[dict]) -> dict:
        if not subset:
            return {}
        return {
            name: {
                "within_1pct": sum(r[f"{name}__within_1pct"] for r in subset),
                "demod_usable": sum(r[f"{name}__demod_usable"] for r in subset),
                "n": len(subset),
            }
            for name in FEATURES
        }

    def grouped(key: str) -> dict:
        buckets: dict[str, list[dict]] = {}
        for record in records:
            buckets.setdefault(str(record[key]), []).append(record)
        return {name: rates(rows) for name, rows in sorted(buckets.items())}

    return {
        "overall": rates(records),
        "by_pulse": grouped("pulse"),
        "by_modulation": grouped("modulation"),
        "by_snr": grouped("snr_db"),
        "by_pulse_and_modulation": {
            f"{pulse}|{modulation}": rates(
                [r for r in records if r["pulse"] == pulse and r["modulation"] == modulation]
            )
            for pulse in sorted({r["pulse"] for r in records})
            for modulation in MODULATIONS
        },
    }


def write_outputs(records: Sequence[dict], summary: dict, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "feature_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if v is None else v) for k, v in record.items()})
    json_path = output / "feature_comparison.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "true_symbol_rate_hz": TRUE_SYMBOL_RATE,
                "sample_rate_hz": FS,
                "samples_per_symbol": SPS,
                "num_symbols": NUM_SYMBOLS,
                "demod_tolerance": DEMOD_TOLERANCE,
                "summary": summary,
                "records": list(records),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path}
