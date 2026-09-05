"""Forward-error-correction adapters."""

from .ldpc_wrapper import decode_ldpc
from .rs_wrapper import decode_reed_solomon
from .viterbi_wrapper import decode_convolutional
from .dispatch import decode_fec

__all__ = ["decode_convolutional", "decode_reed_solomon", "decode_ldpc", "decode_fec"]
