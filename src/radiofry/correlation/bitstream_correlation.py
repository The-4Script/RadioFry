"""Known-sync and periodicity detection for decoded bitstreams."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationResult:
    sync_pattern: str | None
    sync_positions: tuple[int, ...]
    period: int | None
    header_bits: np.ndarray
    payload_bits: np.ndarray
    autocorrelation_peak: float = 0.0
    sync_match_score: float = 0.0


# CCSDS TM Synchronization and Channel Coding ASM, 0x1ACFFC1D.
SYNC_LIBRARY = {"hdlc": "01111110", "ccsds": "00011010110011111111110000011101"}


def _autocorrelation(values: np.ndarray) -> tuple[int | None, float]:
    if values.size < 4:
        return None, 0.0
    signed = values.astype(np.float64) * 2.0 - 1.0
    limit = min(values.size // 2, 512)
    scores = np.array([np.mean(signed[:-lag] * signed[lag:]) for lag in range(1, limit + 1)])
    index = int(np.argmax(scores))
    score = float(scores[index])
    return index + 1, score


def correlate_bitstream(bits: np.ndarray, *, sync_library: dict[str, str] | None = None) -> CorrelationResult:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    library = sync_library or SYNC_LIBRARY
    best_name, best_positions, best_score = None, [], 0.0
    for name, pattern in library.items():
        encoded = np.fromiter((int(bit) for bit in pattern), dtype=np.uint8)
        if values.size < encoded.size:
            continue
        matches = []
        for index in range(values.size - encoded.size + 1):
            score = float(np.mean(values[index:index + encoded.size] == encoded))
            if score >= 0.75:
                matches.append((index, score))
        selected = []
        for index, score in sorted(matches, key=lambda item: (-item[1], item[0])):
            if all(abs(index - previous) >= encoded.size for previous in selected):
                selected.append(index)
        positions = sorted(selected)
        score = max((match_score for _, match_score in matches), default=0.0)
        if (len(positions), score) > (len(best_positions), best_score):
            best_name, best_positions, best_score = name, positions, score
    autocorrelation_period, autocorrelation_peak = _autocorrelation(values)
    period = None
    if len(best_positions) > 1:
        period = int(np.median(np.diff(best_positions)))
    elif autocorrelation_peak >= 0.5:
        period = autocorrelation_period
    split = best_positions[0] if best_positions else 0
    header_size = split + len(library[best_name]) if best_name else 0
    return CorrelationResult(best_name, tuple(best_positions), period, values[:header_size], values[header_size:], autocorrelation_peak, best_score)
