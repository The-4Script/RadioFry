"""RS outer code plus convolutional inner code adapter."""

import numpy as np

from .rs_wrapper import decode_reed_solomon
from .viterbi_wrapper import FECResult, decode_convolutional


def decode_concatenated(bits: np.ndarray) -> FECResult:
    inner = decode_convolutional(bits)
    if not inner.success:
        return FECResult(inner.bits, "concatenated", False, inner.message)
    # commpy returns the six traceback/tail symbols after the decoded payload for
    # the frozen constraint-length-7 code. They are decoder framing, not RS input.
    tail_bits = 6
    if inner.bits.size <= tail_bits or (inner.bits.size - tail_bits) % 8:
        return FECResult(
            inner.bits,
            "concatenated",
            False,
            "Concatenated decoder produced an inner stream that is not byte-aligned "
            "after its known Viterbi traceback tail; no padding or truncation was applied.",
        )
    aligned = inner.bits[:-tail_bits]
    outer = decode_reed_solomon(bytes(np.packbits(aligned)))
    return FECResult(
        outer.bits,
        "concatenated",
        outer.success,
        f"{outer.message} Viterbi traceback tail: {tail_bits} bits excluded.",
        discarded_bits=tail_bits + outer.discarded_bits,
    )
