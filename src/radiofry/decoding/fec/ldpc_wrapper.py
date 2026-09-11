"""LDPC decoding for a KNOWN code, and an honest refusal for an unknown one.

An LDPC code is defined by its parity-check matrix `H`. Without `H` there is nothing to
decode against: the received bits are indistinguishable from any other bitstream, and no
amount of processing recovers the message. Blind recovery of `H` from a captured
bitstream is a research problem in its own right and is not attempted here.

So this adapter has two modes, and says which one it is in:

* **No `H` supplied** - classification only. The bits are returned unchanged with
  `success=False` and a message saying the code parameters are missing. This is the
  default and preserves the previous behaviour exactly.
* **`H` supplied** - Gallager bit-flipping decoding, which genuinely corrects errors.

Why bit-flipping rather than sum-product: the input here is *hard* bits from a
demodulator, not log-likelihood ratios. Sum-product is the stronger algorithm but needs
soft information the pipeline does not currently carry; running it on hard bits would add
complexity without adding capability. If the demodulators ever expose LLRs, sum-product
becomes the right upgrade and this is the place for it.
"""

import numpy as np

from .viterbi_wrapper import FECResult

DEFAULT_MAX_ITERATIONS = 50


def _bit_flipping(codeword: np.ndarray, parity_check: np.ndarray,
                  max_iterations: int) -> tuple[np.ndarray, bool, int]:
    """Gallager's hard-decision algorithm.

    Each round: compute the syndrome, count for every bit how many of the checks it
    participates in are currently unsatisfied, and flip the bits that fail the most
    checks. Terminates as soon as the syndrome is zero, which is a *proof* that the
    result is a codeword - so success here is verified, not assumed.
    """

    bits = codeword.copy().astype(np.uint8)
    # How many checks each bit participates in at all. Used to normalise the vote, so a
    # bit in one check that is failing counts as worse than a bit in three checks of
    # which one is failing. Without that normalisation an irregular code ties constantly:
    # for a (7,4) Hamming code a single error leaves four bits on the same absolute vote,
    # and flipping all of them diverges instead of correcting.
    degree = np.maximum(parity_check.sum(axis=0).astype(np.float64), 1.0)

    for iteration in range(max_iterations):
        syndrome = (parity_check @ bits) % 2
        if not syndrome.any():
            return bits, True, iteration
        votes = (parity_check.T @ syndrome).astype(np.float64)
        if votes.max() == 0:
            break
        # Rank by the fraction of a bit's checks that are unsatisfied, then by the raw
        # count. One bit is flipped per round: on a small or irregular code, flipping
        # every tied bit overshoots, and a single flip is guaranteed to change the
        # syndrome rather than permute it.
        order = np.lexsort((votes, votes / degree))
        bits = bits.copy()
        bits[order[-1]] ^= 1
    return bits, not ((parity_check @ bits) % 2).any(), max_iterations


def decode_ldpc(
    bits: np.ndarray,
    parity_check: np.ndarray | None = None,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    systematic_length: int | None = None,
) -> FECResult:
    """Decode `bits` against `parity_check`, or refuse when the code is unknown.

    `systematic_length` selects the leading message bits from the corrected codeword. Left
    as None the whole corrected codeword is returned, since which bits are systematic is a
    property of the code and is not inferable from `H` alone.
    """

    values = np.asarray(bits, dtype=np.uint8).ravel() & 1

    if parity_check is None:
        return FECResult(
            values, "ldpc", False,
            "LDPC was classified, but decoding is disabled until code parameters are "
            "supplied.")

    matrix = np.asarray(parity_check, dtype=np.uint8)
    if matrix.ndim != 2:
        return FECResult(values, "ldpc", False,
                         "LDPC parity-check matrix must be two-dimensional.")
    checks, length = matrix.shape
    if length == 0 or checks == 0:
        return FECResult(values, "ldpc", False,
                         "LDPC parity-check matrix is empty.")
    if values.size < length:
        return FECResult(
            values, "ldpc", False,
            f"LDPC needs at least {length} bits for this code; got {values.size}.")

    # Decode whole codewords; a trailing partial block is carried through untouched
    # rather than silently dropped or padded into a fabricated codeword.
    blocks = values.size // length
    remainder = values.size - blocks * length
    decoded: list[np.ndarray] = []
    corrected = failed = 0
    for index in range(blocks):
        block = values[index * length:(index + 1) * length].astype(np.uint8)
        result, ok, _ = _bit_flipping(block, matrix, max_iterations)
        if ok:
            corrected += int(np.count_nonzero(result != block))
        else:
            failed += 1
        decoded.append(result[:systematic_length] if systematic_length else result)

    output = np.concatenate(decoded) if decoded else values[:0]
    if remainder:
        output = np.concatenate([output, values[blocks * length:]])

    if failed:
        return FECResult(
            output, "ldpc", False,
            f"LDPC decoding did not converge for {failed} of {blocks} block(s); the "
            "syndrome is still non-zero, so those blocks are not valid codewords.")
    note = f"LDPC decoded {blocks} block(s), correcting {corrected} bit(s)."
    if remainder:
        note += f" {remainder} trailing bit(s) were not a whole codeword and are unchanged."
    return FECResult(output, "ldpc", True, note)
