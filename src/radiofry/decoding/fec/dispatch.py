"""Dispatch decoded bits to the selected FEC adapter."""

import numpy as np

from .concatenated_wrapper import decode_concatenated
from .ldpc_wrapper import decode_ldpc
from .rs_wrapper import decode_reed_solomon
from .viterbi_wrapper import FECResult, decode_convolutional


def decode_fec(bits: np.ndarray, scheme: str) -> FECResult:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if scheme in {"none", "unknown", ""}:
        return FECResult(values, "none", True, "No FEC decoding applied.")
    if scheme == "convolutional":
        return decode_convolutional(values)
    if scheme == "reed_solomon":
        usable = (values.size // 8) * 8
        return decode_reed_solomon(np.packbits(values[:usable]).tobytes())
    if scheme == "concatenated":
        return decode_concatenated(values)
    if scheme == "ldpc":
        return decode_ldpc(values)
    return FECResult(values, scheme, False, f"No FEC adapter is registered for {scheme}.")
