"""Shared typed contracts used across the analysis pipeline."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class UnifiedSignalContainer:
    """Canonical representation of a captured signal."""

    iq: np.ndarray
    sample_rate: float | None = None
    source_format: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = np.asarray(self.iq)
        if samples.ndim != 1:
            raise ValueError("iq must be a one-dimensional sample array")
        if not np.issubdtype(samples.dtype, np.complexfloating):
            samples = samples.astype(np.complex64)
        self.iq = samples.astype(np.complex64, copy=False)
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive when supplied")

    @property
    def duration_sec(self) -> float | None:
        if self.sample_rate is None:
            return None
        return self.iq.size / self.sample_rate
