"""RS outer code plus convolutional inner code adapter."""

import numpy as np

from .rs_wrapper import decode_reed_solomon
from .viterbi_wrapper import FECResult, decode_convolutional


def decode_concatenated(bits: np.ndarray) -> FECResult:
    inner = decode_convolutional(bits)
    if not inner.success:
        return FECResult(inner.bits, "concatenated", False, inner.message)
    outer = decode_reed_solomon(bytes(np.packbits(inner.bits)))
    return FECResult(outer.bits, "concatenated", outer.success, outer.message)
