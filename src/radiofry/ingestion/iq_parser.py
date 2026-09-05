"""Parser for common headerless interleaved IQ files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radiofry.contracts import UnifiedSignalContainer


@dataclass(frozen=True)
class IQFormat:
    """Raw IQ interpretation supplied by a sensor or user override."""

    dtype: str = "int16"
    byte_order: str = "little"

    def numpy_dtype(self) -> np.dtype:
        if self.dtype not in {"int16", "float32"}:
            raise ValueError("dtype must be 'int16' or 'float32'")
        if self.byte_order not in {"little", "big"}:
            raise ValueError("byte_order must be 'little' or 'big'")
        kind = {"int16": "i2", "float32": "f4"}[self.dtype]
        return np.dtype(("<" if self.byte_order == "little" else ">") + kind)


def read_iq(
    path: str | Path,
    *,
    sample_rate: float | None = None,
    fmt: IQFormat | None = None,
    max_bytes: int | None = None,
    max_samples: int | None = None,
) -> UnifiedSignalContainer:
    """Read interleaved I,Q values from a headerless binary file."""

    fmt = fmt or IQFormat()
    file_path = Path(path)
    if max_bytes is not None and file_path.stat().st_size > max_bytes:
        raise ValueError(f"IQ file exceeds the {max_bytes:,}-byte limit")
    values = np.fromfile(path, dtype=fmt.numpy_dtype())
    if values.size == 0:
        raise ValueError("IQ file contains no samples")
    if values.size % 2:
        raise ValueError("interleaved IQ file must contain an even number of values")
    if max_samples is not None and values.size // 2 > max_samples:
        raise ValueError(f"IQ file exceeds the {max_samples:,}-sample limit")
    pairs = values.reshape(-1, 2)
    iq = pairs[:, 0].astype(np.float32) + 1j * pairs[:, 1].astype(np.float32)
    return UnifiedSignalContainer(
        iq=iq,
        sample_rate=sample_rate,
        source_format="iq",
        metadata={"dtype": fmt.dtype, "byte_order": fmt.byte_order, "path": str(path)},
    )
