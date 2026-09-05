"""Shared output contract for demodulators."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DemodulationResult:
    symbols: np.ndarray
    bits: np.ndarray
    modulation: str
    metadata: dict[str, float | int | str] = field(default_factory=dict)
