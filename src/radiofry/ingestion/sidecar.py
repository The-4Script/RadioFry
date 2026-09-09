"""Minimal SigMF sidecar reader for real capture metadata (BANK.md Entry 030).

Deliberately **not** a SigMF implementation. RadioFry needs exactly two things a real
receiver records and a raw IQ file cannot carry: the tuner's centre frequency and the
sample rate. Everything else in the standard is ignored.

Why this exists: Entry 029 established that the suppressed AM-SSB carrier cannot be
recovered blind to the ~1-2 Hz the demodulator needs, while `estimate_parameters`
already prefers `metadata["center_frequency_hz"]`. Nothing ever populated it. A
`.sigmf-meta` sidecar is legitimate capture metadata - what the radio was tuned to - not
an estimate and not ground truth.

The synthetic generator does not write these files, so this path can never turn synthetic
ground truth into an estimator shortcut.
"""

import json
import math
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".sigmf-meta"
SIGMF_FREQUENCY_KEY = "core:frequency"
SIGMF_SAMPLE_RATE_KEY = "core:sample_rate"
# A tuner frequency is a non-negative, finite number of Hz. 0 is legitimate (baseband).
_MAX_PLAUSIBLE_HZ = 1e15


def _finite_non_negative(value: Any) -> float | None:
    """Accept a real number in [0, 1e15]; reject anything else rather than guessing."""

    # bool is an int subclass and is never a frequency.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > _MAX_PLAUSIBLE_HZ:
        return None
    return number


def sidecar_path_for(capture_path: str | Path) -> Path:
    """`cap.iq` / `cap.sigmf-data` -> `cap.sigmf-meta`, per the SigMF layout."""

    path = Path(capture_path)
    return path.with_suffix(SIDECAR_SUFFIX)


def read_sigmf_sidecar(capture_path: str | Path) -> dict[str, float]:
    """Read centre frequency and sample rate from a capture's SigMF sidecar.

    Returns an empty dict when the sidecar is missing, unreadable, malformed, or carries
    no usable value. A missing value is never replaced with a default - an absent centre
    frequency has to stay absent so the estimator falls back to its own analysis.
    """

    path = sidecar_path_for(capture_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}

    found: dict[str, float] = {}

    captures = document.get("captures")
    if isinstance(captures, list):
        # The first capture segment describes how the recording starts, which is the
        # only segment a single-segment RadioFry capture has.
        for segment in captures:
            if not isinstance(segment, dict):
                continue
            frequency = _finite_non_negative(segment.get(SIGMF_FREQUENCY_KEY))
            if frequency is not None:
                found["center_frequency_hz"] = frequency
            break

    global_block = document.get("global")
    if isinstance(global_block, dict):
        sample_rate = _finite_non_negative(global_block.get(SIGMF_SAMPLE_RATE_KEY))
        if sample_rate is not None and sample_rate > 0.0:
            found["sample_rate_hz"] = sample_rate

    return found
