import numpy as np

from radiofry.correlation.bitstream_correlation import correlate_bitstream
from radiofry.decoding.demodulators.psk_demod import demodulate_psk
from radiofry.decoding.demodulators.fsk_demod import demodulate_fsk
from radiofry.decoding.demodulators.qam_demod import demodulate_qam
from radiofry.decoding.deinterleavers.block import block_deinterleave
from radiofry.decoding.fec.rs_wrapper import decode_reed_solomon
from radiofry.synthetic_gen.fec_dataset import reed_solomon_encode
from radiofry.synthetic_gen.interleavers import block_interleave
from radiofry.synthetic_gen.interleavers import convolutional_deinterleave, convolutional_interleave, diagonal_deinterleave, diagonal_interleave
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
    np.testing.assert_array_equal(block_deinterleave(interleaved, 2, 4), bits)


def test_deterministic_interleavers_round_trip() -> None:
    bits = np.arange(128, dtype=np.uint8) % 2

    np.testing.assert_array_equal(diagonal_deinterleave(diagonal_interleave(bits, 8), 8), bits)
    np.testing.assert_array_equal(convolutional_deinterleave(convolutional_interleave(bits, 4, 1), 4, 1), bits)


def test_fsk_demodulator_recovers_frequency_signs() -> None:
    increments = np.array([-0.8, 0.8, -0.8, 0.8], dtype=np.float32)
    samples = np.exp(1j * np.cumsum(increments)).astype(np.complex64)

    result = demodulate_fsk(samples)

    np.testing.assert_array_equal(result.bits, [1, 0, 1])


def test_qam16_demodulator_returns_expected_bit_count() -> None:
    samples = np.array([-3 - 3j, -3 + 3j, 3 - 3j, 3 + 3j], dtype=np.complex64)

    result = demodulate_qam(samples, order=16)

    assert result.bits.size == 16
    assert result.symbols.size == samples.size


def test_reed_solomon_round_trip() -> None:
    source = np.arange(32 * 8, dtype=np.uint8) % 2
    encoded = reed_solomon_encode(source)
    decoded = decode_reed_solomon(np.packbits(encoded).tobytes())

    assert decoded.success
    np.testing.assert_array_equal(decoded.bits, source)


def test_convolutional_search_is_available() -> None:
    bits = np.arange(128, dtype=np.uint8) % 2
    result = search_deinterleave(convolutional_interleave(bits, 4, 1), "convolutional")

    assert result.interleaver_type == "convolutional"


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
