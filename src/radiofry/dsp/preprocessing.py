"""Format-independent preprocessing for captured IQ samples."""

import numpy as np
from scipy.signal import resample_poly

from radiofry.contracts import UnifiedSignalContainer


def preprocess(
    signal: UnifiedSignalContainer,
    *,
    target_sample_rate: float | None = None,
) -> UnifiedSignalContainer:
    """Remove DC, normalize average power, and optionally resample IQ."""

    iq = signal.iq.astype(np.complex64, copy=True)
    iq -= np.mean(iq, dtype=np.complex64)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) if iq.size else 0.0
    if rms > 0:
        iq /= rms

    sample_rate = signal.sample_rate
    if target_sample_rate is not None:
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if sample_rate is None:
            raise ValueError("cannot resample a signal with unknown sample rate")
        if not np.isclose(sample_rate, target_sample_rate):
            gcd = np.gcd(round(sample_rate), round(target_sample_rate))
            up = round(target_sample_rate) // gcd
            down = round(sample_rate) // gcd
            iq = resample_poly(iq, up, down).astype(np.complex64)
            sample_rate = float(target_sample_rate)

    metadata = dict(signal.metadata)
    metadata["preprocessed"] = True
    return UnifiedSignalContainer(
        iq=iq,
        sample_rate=sample_rate,
        source_format=signal.source_format,
        metadata=metadata,
    )
