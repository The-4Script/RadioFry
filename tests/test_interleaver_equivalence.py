import numpy as np

from radiofry.decoding.deinterleavers.convolutional import convolutional_deinterleave
from radiofry.decoding.deinterleavers.diagonal import diagonal_deinterleave
from radiofry.synthetic_gen.interleavers import convolutional_interleave, diagonal_interleave


def test_convolutional_generator_and_decoder_stay_in_agreement() -> None:
    bits = np.random.default_rng(7).integers(0, 2, size=128, dtype=np.uint8)

    encoded = convolutional_interleave(bits, 4, 1)

    np.testing.assert_array_equal(convolutional_deinterleave(encoded, 4, 1), bits)


def test_diagonal_generator_and_decoder_stay_in_agreement() -> None:
    bits = np.random.default_rng(8).integers(0, 2, size=128, dtype=np.uint8)

    encoded = diagonal_interleave(bits, 8)

    np.testing.assert_array_equal(diagonal_deinterleave(encoded, 8), bits)