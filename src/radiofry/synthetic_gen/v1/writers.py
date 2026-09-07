"""Capture writers for Synthetic Dataset V1.

Both writers target the formats the existing RadioFry ingestion layer already
reads: headerless interleaved I/Q (``ingestion/iq_parser.py``) and two-channel
WAV interpreted as I/Q (``ingestion/wav_parser.py``). The ``IQFormat`` contract
is reused directly rather than duplicated.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from radiofry.ingestion.iq_parser import IQFormat

INT16_FULL_SCALE = 32767.0
DEFAULT_FULL_SCALE_FRACTION = 0.95


@dataclass(frozen=True)
class WriteReport:
    """How a capture was serialized, and how to invert that serialization."""

    path: str
    file_format: str
    dtype: str
    byte_order: str
    channel_layout: str
    scale_factor: float
    full_scale_fraction: float
    clipped_samples: int
    num_samples: int
    bytes_written: int


def _integer_scale(iq: np.ndarray, full_scale_fraction: float) -> float:
    peak = float(np.max(np.abs(np.concatenate((iq.real, iq.imag))))) if iq.size else 0.0
    if peak <= 0:
        return 1.0
    return INT16_FULL_SCALE * full_scale_fraction / peak


def _to_int16(iq: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, int]:
    real = np.round(iq.real * scale)
    imag = np.round(iq.imag * scale)
    clipped = int(np.count_nonzero(np.abs(real) > INT16_FULL_SCALE) + np.count_nonzero(np.abs(imag) > INT16_FULL_SCALE))
    return np.clip(real, -INT16_FULL_SCALE, INT16_FULL_SCALE), np.clip(imag, -INT16_FULL_SCALE, INT16_FULL_SCALE), clipped


def write_iq_file(
    path: str | Path,
    iq: np.ndarray,
    *,
    fmt: IQFormat | None = None,
    full_scale_fraction: float = DEFAULT_FULL_SCALE_FRACTION,
) -> WriteReport:
    """Write a headerless interleaved I,Q capture."""

    fmt = fmt or IQFormat()
    target = Path(path)
    samples = np.asarray(iq, dtype=np.complex64)
    if fmt.dtype == "int16":
        scale = _integer_scale(samples, full_scale_fraction)
        real, imag, clipped = _to_int16(samples, scale)
    elif fmt.dtype == "float32":
        scale, clipped = 1.0, 0
        real, imag = samples.real, samples.imag
    else:
        raise ValueError("IQ dtype must be 'int16' or 'float32'")
    interleaved = np.empty(samples.size * 2, dtype=fmt.numpy_dtype())
    interleaved[0::2] = real
    interleaved[1::2] = imag
    target.parent.mkdir(parents=True, exist_ok=True)
    interleaved.tofile(target)
    return WriteReport(
        path=str(target),
        file_format="iq",
        dtype=fmt.dtype,
        byte_order=fmt.byte_order,
        channel_layout="interleaved_iq",
        scale_factor=float(scale),
        full_scale_fraction=float(full_scale_fraction),
        clipped_samples=clipped,
        num_samples=int(samples.size),
        bytes_written=target.stat().st_size,
    )


def write_wav_file(
    path: str | Path,
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    dtype: str = "int16",
    full_scale_fraction: float = DEFAULT_FULL_SCALE_FRACTION,
) -> WriteReport:
    """Write a two-channel WAV whose left/right channels carry I and Q."""

    if dtype not in {"int16", "float32"}:
        raise ValueError("WAV dtype must be 'int16' or 'float32'")
    if abs(sample_rate_hz - round(sample_rate_hz)) > 1e-9:
        raise ValueError("WAV headers store an integer sample rate; use an integer sample_rate_hz")
    target = Path(path)
    samples = np.asarray(iq, dtype=np.complex64)
    if dtype == "int16":
        scale = _integer_scale(samples, full_scale_fraction)
        real, imag, clipped = _to_int16(samples, scale)
        frames = np.stack((real, imag), axis=1).astype(np.int16)
    else:
        scale, clipped = 1.0, 0
        frames = np.stack((samples.real, samples.imag), axis=1).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(target, int(round(sample_rate_hz)), frames)
    return WriteReport(
        path=str(target),
        file_format="wav",
        dtype=dtype,
        byte_order="little",
        channel_layout="stereo_iq",
        scale_factor=float(scale),
        full_scale_fraction=float(full_scale_fraction),
        clipped_samples=clipped,
        num_samples=int(samples.size),
        bytes_written=target.stat().st_size,
    )
