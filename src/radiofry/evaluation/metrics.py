"""Scoring primitives for the Synthetic V1 evaluation harness.

Every metric distinguishes "measured" from "unavailable"; nothing here invents a
value when the pipeline did not expose one.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Generator modulation family -> the vocabulary dsp/cyclostationary.py emits and
# fusion/confidence_fusion.py maps labels onto. PAM shares the QAM-like family, matching
# fusion's own mapping for PAM4; analog is present so a bit-less capture cannot crash the
# harness before its ingestion try/except. See BANK.md Entry 019.
_FAMILY_BY_V1_FAMILY = {
    "psk": "PSK-like",
    "qam": "QAM-like",
    "pam": "QAM-like",
    "fsk": "FSK-like",
    "analog": "analog-like",
}

DEFAULT_MAX_ALIGNMENT_OFFSET = 64


@dataclass(frozen=True)
class BerResult:
    """Bit-error scoring outcome, including why it may be unavailable."""

    status: str
    reason: str
    strict: float | None
    aligned: float | None
    alignment_offset: int | None
    compared_bits: int
    expected_bits: int
    recovered_bits: int


def expected_family_for(v1_family: str) -> str:
    """Map a V1 modulation family onto the classical detector's vocabulary."""

    return _FAMILY_BY_V1_FAMILY[v1_family]


def signed_error(estimate: float | None, truth: float | None) -> float | None:
    if estimate is None or truth is None:
        return None
    return float(estimate) - float(truth)


def relative_error(estimate: float | None, truth: float | None) -> float | None:
    if estimate is None or truth is None or float(truth) == 0.0:
        return None
    return (float(estimate) - float(truth)) / float(truth)


def bit_error_rate(truth: Sequence[int] | np.ndarray, recovered: Sequence[int] | np.ndarray) -> float | None:
    """Fraction of differing bits over the overlapping prefix of both streams."""

    reference = np.asarray(truth, dtype=np.uint8).ravel()
    measured = np.asarray(recovered, dtype=np.uint8).ravel()
    length = min(reference.size, measured.size)
    if length == 0:
        return None
    return float(np.mean(reference[:length] != measured[:length]))


def best_aligned_ber(
    truth: Sequence[int] | np.ndarray,
    recovered: Sequence[int] | np.ndarray,
    *,
    max_offset: int = DEFAULT_MAX_ALIGNMENT_OFFSET,
) -> tuple[float | None, int | None]:
    """Lowest BER over a bounded bit-shift search.

    A positive offset means the recovered stream started that many bits late
    (leading truth bits were dropped); a negative offset means it started early.
    This is a diagnostic only - the strict, unshifted rate stays the headline
    number so a timing failure is never hidden.
    """

    reference = np.asarray(truth, dtype=np.uint8).ravel()
    measured = np.asarray(recovered, dtype=np.uint8).ravel()
    if reference.size == 0 or measured.size == 0:
        return None, None
    limit = int(max(0, min(max_offset, reference.size - 1, measured.size - 1)))
    best_ber, best_offset = None, None
    for offset in range(-limit, limit + 1):
        if offset >= 0:
            candidate = bit_error_rate(reference[offset:], measured)
        else:
            candidate = bit_error_rate(reference, measured[-offset:])
        if candidate is None:
            continue
        if best_ber is None or candidate < best_ber:
            best_ber, best_offset = candidate, offset
    return best_ber, best_offset


def score_bits(
    truth: Sequence[int] | np.ndarray,
    recovered: Sequence[int] | np.ndarray | None,
    *,
    available: bool,
    reason: str = "",
    max_offset: int = DEFAULT_MAX_ALIGNMENT_OFFSET,
) -> BerResult:
    """Score recovered bits against ground truth, or record why it cannot be done."""

    reference = np.asarray(truth, dtype=np.uint8).ravel()
    if not available or recovered is None:
        return BerResult(
            "unavailable", reason or "demodulation_unavailable", None, None, None, 0, int(reference.size), 0
        )
    measured = np.asarray(recovered, dtype=np.uint8).ravel()
    if measured.size == 0:
        return BerResult("unavailable", "no_recovered_bits", None, None, None, 0, int(reference.size), 0)
    strict = bit_error_rate(reference, measured)
    aligned, offset = best_aligned_ber(reference, measured, max_offset=max_offset)
    return BerResult(
        "ok",
        "",
        strict,
        aligned,
        offset,
        int(min(reference.size, measured.size)),
        int(reference.size),
        int(measured.size),
    )


def confusion_matrix(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Nested truth -> prediction -> count table."""

    matrix: dict[str, dict[str, int]] = {}
    for truth_label, predicted_label in pairs:
        row = matrix.setdefault(str(truth_label), {})
        row[str(predicted_label)] = row.get(str(predicted_label), 0) + 1
    return matrix


def accuracy(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Fraction of pairs whose prediction equals the truth label."""

    if not pairs:
        return None
    return float(np.mean([truth == prediction for truth, prediction in pairs]))
