"""Optional scikit-commpy Viterbi adapter."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FECResult:
    bits: np.ndarray
    scheme: str
    success: bool
    message: str = ""


def decode_convolutional(bits: np.ndarray, *, constraint_length: int = 7) -> FECResult:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    try:
        from commpy.channelcoding import Trellis, convcode
        memory = np.array([constraint_length - 1])
        generators = np.array([[0o171, 0o133]])
        trellis = Trellis(memory, generators)
        decoded = convcode.viterbi_decode(values.astype(float), trellis, tb_depth=5 * constraint_length)
        return FECResult(np.asarray(decoded, dtype=np.uint8), "convolutional", True)
    except (ImportError, ValueError, IndexError) as error:
        return FECResult(values, "convolutional", False, f"Viterbi unavailable: {error}")
