import numpy as np

from radiofry.correlation.bitstream_correlation import correlate_bitstream
from radiofry.decoding.demodulators.psk_demod import demodulate_psk
from radiofry.synthetic_gen.interleavers import block_interleave
from radiofry.decoding.deinterleave_search import search_deinterleave
from radiofry.decoding.fec.dispatch import decode_fec


def test_bpsk_demodulator_recovers_bits() -> None:
    samples = np.array([1, -1, -1, 1], dtype=np.complex64)

    result = demodulate_psk(samples, order=2)

    np.testing.assert_array_equal(result.bits, [0, 1, 1, 0])


def test_block_interleaver_preserves_bits_and_changes_order() -> None:
    bits = np.arange(8, dtype=np.uint8) % 2

    interleaved = block_interleave(bits, 2, 4)

    assert sorted(interleaved.tolist()) == sorted(bits.tolist())
    assert not np.array_equal(interleaved, bits)


def test_correlation_finds_repeated_sync_word() -> None:
    bits = np.array([1, 0, 1] + [int(bit) for bit in "01111110"] + [0, 1] + [int(bit) for bit in "01111110"] + [1])

    result = correlate_bitstream(bits)

    assert result.sync_pattern == "hdlc"
    assert result.sync_positions == (3, 13)
    assert result.period == 10


def test_decode_dispatch_preserves_bits_when_fec_is_none() -> None:
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)

    result = decode_fec(bits, "none")

    assert result.success
    np.testing.assert_array_equal(result.bits, bits)


def test_deinterleave_search_reports_pseudorandom_limitation() -> None:
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)

    result = search_deinterleave(bits, "pseudo_random")

    assert result.limitation is not None
