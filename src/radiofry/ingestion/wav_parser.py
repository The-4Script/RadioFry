"""WAV parser with mono and stereo-IQ support."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

from radiofry.contracts import UnifiedSignalContainer


def _scale_audio(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = max(abs(info.min), info.max)
        return samples.astype(np.float32) / scale
    return samples.astype(np.float32, copy=False)


def read_wav(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    max_samples: int | None = None,
) -> UnifiedSignalContainer:
    """Read a WAV file, treating stereo channels as I/Q and mono as analytic IQ."""

    file_path = Path(path)
    if max_bytes is not None and file_path.stat().st_size > max_bytes:
        raise ValueError(f"WAV file exceeds the {max_bytes:,}-byte limit")
    sample_rate, raw = wavfile.read(path)
    samples = _scale_audio(np.asarray(raw))
    if max_samples is not None and samples.shape[0] > max_samples:
        raise ValueError(f"WAV file exceeds the {max_samples:,}-sample limit")
    if samples.ndim == 2 and samples.shape[1] == 2:
        iq = samples[:, 0] + 1j * samples[:, 1]
        channel_mode = "stereo_iq"
    elif samples.ndim == 1:
        iq = hilbert(samples).astype(np.complex64)
        channel_mode = "mono_analytic"
    else:
        raise ValueError("WAV input must be mono or two-channel stereo")
    return UnifiedSignalContainer(
        iq=iq,
        sample_rate=float(sample_rate),
        source_format="wav",
        metadata={"channel_mode": channel_mode, "path": str(path)},
    )
