"""Pseudo-random de-interleaving for a KNOWN seed.

A pseudo-random interleaver applies a permutation drawn from a seeded generator. Without
the seed the permutation is one of `n!` possibilities and the bitstream carries no
information about which - so blind inversion is not merely unimplemented here, it is not
recoverable. `search_deinterleave` continues to refuse when no seed is supplied.

With the seed it is exact, and it is exact because this reproduces the *same* generator
call the interleaver used (`np.random.default_rng(seed).permutation(n)`) and inverts it.
Reproducing the permutation rather than guessing it is what makes this lossless.
"""

import numpy as np


def pseudo_random_deinterleave(bits: np.ndarray, seed: int) -> np.ndarray:
    """Invert `synthetic_gen.interleavers.pseudo_random_interleave` for a known seed."""

    values = np.asarray(bits, dtype=np.uint8).ravel()
    if values.size == 0:
        return values
    permutation = np.random.default_rng(seed).permutation(values.size)
    restored = np.empty_like(values)
    # interleave did `out = values[permutation]`, so `restored[permutation] = values`.
    restored[permutation] = values
    return restored
