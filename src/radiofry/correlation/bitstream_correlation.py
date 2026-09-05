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


SYNC_LIBRARY = {"hdlc": "01111110", "ccsds": "0001101"}


def correlate_bitstream(bits: np.ndarray, *, sync_library: dict[str, str] | None = None) -> CorrelationResult:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    library = sync_library or SYNC_LIBRARY
    best_name, best_positions = None, []
    for name, pattern in library.items():
        encoded = np.fromiter((int(bit) for bit in pattern), dtype=np.uint8)
        positions = [index for index in range(values.size - encoded.size + 1) if np.array_equal(values[index:index + encoded.size], encoded)]
        if len(positions) > len(best_positions):
            best_name, best_positions = name, positions
    period = None
    if len(best_positions) > 1:
        period = int(np.median(np.diff(best_positions)))
    split = best_positions[0] if best_positions else 0
    header_size = split + len(library[best_name]) if best_name else 0
    return CorrelationResult(best_name, tuple(best_positions), period, values[:header_size], values[header_size:])
