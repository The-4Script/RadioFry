"""Bounded candidate search for block and diagonal interleavers."""

from dataclasses import dataclass

import numpy as np

from .deinterleavers.block import block_deinterleave
from .deinterleavers.diagonal import diagonal_deinterleave


@dataclass(frozen=True)
class DeinterleaveResult:
    bits: np.ndarray
    interleaver_type: str
    parameters: dict[str, int]
    score: float
    limitation: str | None = None


def _entropy(bits: np.ndarray) -> float:
    if bits.size == 0:
        return 0.0
    counts = np.bincount(bits.astype(np.uint8), minlength=2) / bits.size
    return float(-sum(prob * np.log2(prob) for prob in counts if prob > 0))


def search_deinterleave(bits: np.ndarray, interleaver_type: str, candidates: tuple[int, ...] = (2, 4, 8, 16)) -> DeinterleaveResult:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if interleaver_type == "pseudo_random":
        return DeinterleaveResult(values, interleaver_type, {}, 0.0, "Exact de-interleaving requires the generator seed or permutation.")
    best = (values, {}, -_entropy(values))
    for first in candidates:
        try:
            candidate = block_deinterleave(values, first, values.size // first) if interleaver_type == "block" and values.size % first == 0 else diagonal_deinterleave(values, first) if interleaver_type == "diagonal" and values.size % first == 0 else None
        except ValueError:
            candidate = None
        if candidate is not None and -_entropy(candidate) > best[2]:
            best = (candidate, {"rows": first, "columns": values.size // first} if interleaver_type == "block" else {"width": first}, -_entropy(candidate))
    return DeinterleaveResult(best[0], interleaver_type, best[1], best[2])
