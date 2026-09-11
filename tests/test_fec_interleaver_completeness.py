"""Known-answer tests for every FEC scheme and interleaver the PS requires.

`success=True` returning garbage is worse than an honest failure, so nothing here checks
that a decoder merely runs. Each case encodes a known message with the matching encoder,
corrupts it within (and beyond) the code's stated capability, decodes, and compares
against the original bits.

Two schemes are deliberately unrecoverable without side information, and the tests pin
BOTH halves of that: the honest refusal when the parameters are absent, and exact recovery
when they are supplied.

* **LDPC** needs its parity-check matrix. Without `H` the received bits are
  indistinguishable from any other bitstream.
* **Pseudo-random interleaving** needs the generator seed. The permutation is one of `n!`
  and the bitstream carries no information about which.

`reedsolo` and `scikit-commpy` are the declared `fec` extra; tests that need them skip
when they are absent rather than asserting a weaker property.
"""

import numpy as np
import pytest

from radiofry.decoding.deinterleave_search import search_deinterleave
from radiofry.decoding.deinterleavers import (
    block_deinterleave,
    convolutional_deinterleave,
    diagonal_deinterleave,
    pseudo_random_deinterleave,
)
from radiofry.decoding.fec.dispatch import decode_fec
from radiofry.decoding.fec.ldpc_wrapper import decode_ldpc
from radiofry.synthetic_gen.interleavers import pseudo_random_interleave

RNG = np.random.default_rng(5150)


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


requires_reedsolo = pytest.mark.skipif(
    not _has("reedsolo"), reason="reedsolo is part of the optional fec extra")
requires_commpy = pytest.mark.skipif(
    not _has("commpy"), reason="scikit-commpy is part of the optional fec extra")


# --- Reed-Solomon -----------------------------------------------------------------------


@requires_reedsolo
@pytest.mark.parametrize("errors", [0, 5, 16])
def test_reed_solomon_recovers_within_its_correction_limit(errors: int) -> None:
    """RSCodec(32) corrects up to 16 symbol errors; recovery must be exact, not close."""
    from reedsolo import RSCodec

    message = bytes(RNG.integers(0, 256, 100).tolist())
    encoded = bytearray(RSCodec(32).encode(message))
    for position in RNG.choice(len(encoded), errors, replace=False) if errors else []:
        encoded[position] ^= 0xFF

    result = decode_fec(np.unpackbits(np.frombuffer(bytes(encoded), dtype=np.uint8)),
                        "reed_solomon")

    assert result.success, result.message
    expected = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
    assert np.array_equal(result.bits[:expected.size], expected), (
        f"{errors} byte errors should be fully corrected")


@requires_reedsolo
def test_reed_solomon_refuses_beyond_its_correction_limit() -> None:
    """Failing loudly past 16 errors is correct behaviour, not a defect."""
    from reedsolo import RSCodec

    message = bytes(RNG.integers(0, 256, 100).tolist())
    encoded = bytearray(RSCodec(32).encode(message))
    for position in RNG.choice(len(encoded), 17, replace=False):
        encoded[position] ^= 0xFF

    result = decode_fec(np.unpackbits(np.frombuffer(bytes(encoded), dtype=np.uint8)),
                        "reed_solomon")

    assert not result.success
    assert "failed" in result.message.lower()


# --- convolutional / Viterbi --------------------------------------------------------------


@requires_commpy
@pytest.mark.parametrize("flips", [0, 5, 20])
def test_viterbi_recovers_the_message_through_bit_errors(flips: int) -> None:
    from commpy.channelcoding import Trellis, convcode

    trellis = Trellis(np.array([6]), np.array([[0o171, 0o133]]))
    message = RNG.integers(0, 2, 200).astype(int)
    encoded = convcode.conv_encode(message, trellis)
    if flips:
        encoded = encoded.copy()
        encoded[RNG.choice(encoded.size, flips, replace=False)] ^= 1

    result = decode_fec(np.asarray(encoded, dtype=np.uint8), "convolutional")

    assert result.success, result.message
    assert np.array_equal(result.bits[:message.size], message.astype(np.uint8)), (
        f"{flips} bit flips within the code's capability must be corrected")


@requires_commpy
@requires_reedsolo
def test_concatenated_recovers_through_both_layers() -> None:
    from commpy.channelcoding import Trellis, convcode
    from reedsolo import RSCodec

    message = bytes(RNG.integers(0, 256, 60).tolist())
    inner = np.unpackbits(np.frombuffer(bytes(RSCodec(32).encode(message)),
                                        dtype=np.uint8))
    encoded = convcode.conv_encode(inner.astype(int),
                                   Trellis(np.array([6]), np.array([[0o171, 0o133]])))

    result = decode_fec(np.asarray(encoded, dtype=np.uint8), "concatenated")

    assert result.success, result.message
    expected = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
    assert np.array_equal(result.bits[:expected.size], expected)


# --- LDPC ----------------------------------------------------------------------------------


def _hamming_parity_check() -> np.ndarray:
    """(7,4) Hamming as a small, exactly-known single-error-correcting code."""
    return np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.uint8)


def _hamming_codewords(parity_check: np.ndarray) -> list[np.ndarray]:
    """Every 7-bit vector with a zero syndrome - the code's 16 valid codewords."""
    words = []
    for value in range(128):
        vector = np.array([(value >> shift) & 1 for shift in range(6, -1, -1)],
                          dtype=np.uint8)
        if not ((parity_check @ vector) % 2).any():
            words.append(vector)
    return words


def test_the_hamming_fixture_is_a_real_code() -> None:
    """Guards the test itself: 16 codewords, minimum distance 3."""
    codewords = _hamming_codewords(_hamming_parity_check())

    assert len(codewords) == 16
    distances = [int(np.count_nonzero(a != b))
                 for i, a in enumerate(codewords) for b in codewords[i + 1:]]
    assert min(distances) == 3, "a distance-3 code corrects exactly one error"


def test_ldpc_refuses_without_code_parameters() -> None:
    """The default path, unchanged: no parity-check matrix means nothing to decode."""
    result = decode_fec(RNG.integers(0, 2, 256).astype(np.uint8), "ldpc")

    assert not result.success
    assert "code parameters" in result.message


def test_ldpc_corrects_a_single_error_when_the_code_is_known() -> None:
    parity_check = _hamming_parity_check()
    codeword = _hamming_codewords(parity_check)[9]

    for position in range(codeword.size):
        received = codeword.copy()
        received[position] ^= 1

        result = decode_ldpc(received, parity_check)

        assert result.success, f"bit {position}: {result.message}"
        assert np.array_equal(result.bits, codeword), f"bit {position} not corrected"


def test_ldpc_passes_a_clean_codeword_through_unchanged() -> None:
    parity_check = _hamming_parity_check()
    codeword = _hamming_codewords(parity_check)[3]

    result = decode_ldpc(codeword, parity_check)

    assert result.success
    assert np.array_equal(result.bits, codeword)
    assert "correcting 0 bit" in result.message


def test_ldpc_decodes_several_blocks_at_once() -> None:
    parity_check = _hamming_parity_check()
    codewords = _hamming_codewords(parity_check)
    chosen = [codewords[i] for i in (1, 5, 11, 14)]
    stream = np.concatenate(chosen).copy()
    stream[2] ^= 1          # one error in block 0
    stream[7 * 2 + 4] ^= 1  # one error in block 2

    result = decode_ldpc(stream, parity_check)

    assert result.success, result.message
    assert np.array_equal(result.bits, np.concatenate(chosen))


def test_ldpc_reports_failure_when_a_block_does_not_converge() -> None:
    """Two errors exceed a distance-3 code; the syndrome proves it did not succeed."""
    parity_check = _hamming_parity_check()
    received = _hamming_codewords(parity_check)[9].copy()
    received[0] ^= 1
    received[1] ^= 1

    result = decode_ldpc(received, parity_check, max_iterations=20)

    if result.success:
        # Bit-flipping may land on a *different* valid codeword; that is a decoding
        # error, not a crash, and it must not be reported as the original message.
        assert not np.array_equal(result.bits, _hamming_codewords(parity_check)[9])
    else:
        assert "converge" in result.message


def test_ldpc_carries_a_trailing_partial_block_through_untouched() -> None:
    parity_check = _hamming_parity_check()
    codeword = _hamming_codewords(parity_check)[6]
    tail = np.array([1, 0, 1], dtype=np.uint8)

    result = decode_ldpc(np.concatenate([codeword, tail]), parity_check)

    assert result.success
    assert np.array_equal(result.bits[-3:], tail), "a partial block must not be invented"
    assert "trailing" in result.message


def test_ldpc_rejects_a_malformed_parity_check_matrix() -> None:
    bits = RNG.integers(0, 2, 64).astype(np.uint8)

    assert not decode_ldpc(bits, np.array([1, 0, 1], dtype=np.uint8)).success
    assert not decode_ldpc(bits, np.empty((0, 0), dtype=np.uint8)).success
    assert not decode_ldpc(bits[:3], _hamming_parity_check()).success


def test_ldpc_can_return_only_the_systematic_bits() -> None:
    parity_check = _hamming_parity_check()
    codeword = _hamming_codewords(parity_check)[9]

    result = decode_ldpc(codeword, parity_check, systematic_length=4)

    assert result.bits.size == 4
    assert np.array_equal(result.bits, codeword[:4])


# --- interleavers ----------------------------------------------------------------------------


def test_block_interleaving_round_trips() -> None:
    from radiofry.synthetic_gen.interleavers import block_interleave

    bits = RNG.integers(0, 2, 64).astype(np.uint8)

    restored = block_deinterleave(block_interleave(bits, 8, 8), 8, 8)

    assert np.array_equal(restored, bits)


def test_diagonal_interleaving_round_trips() -> None:
    from radiofry.synthetic_gen.interleavers import diagonal_interleave

    bits = RNG.integers(0, 2, 64).astype(np.uint8)

    restored = diagonal_deinterleave(diagonal_interleave(bits, 8), 8)

    assert np.array_equal(restored, bits)


def test_convolutional_interleaving_round_trips() -> None:
    from radiofry.synthetic_gen.interleavers import convolutional_interleave

    bits = RNG.integers(0, 2, 64).astype(np.uint8)

    restored = convolutional_deinterleave(convolutional_interleave(bits, 4, 1), 4, 1)

    assert np.array_equal(restored, bits)


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_pseudo_random_interleaving_round_trips_with_the_seed(seed: int) -> None:
    """Exact, because the same generator call is reproduced rather than guessed."""
    bits = RNG.integers(0, 2, 256).astype(np.uint8)

    restored = pseudo_random_deinterleave(pseudo_random_interleave(bits, seed), seed)

    assert np.array_equal(restored, bits)


def test_the_wrong_seed_does_not_restore_the_bitstream() -> None:
    """Confirms the seed is doing the work, not some accidental identity."""
    bits = RNG.integers(0, 2, 256).astype(np.uint8)
    interleaved = pseudo_random_interleave(bits, 7)

    assert not np.array_equal(pseudo_random_deinterleave(interleaved, 8), bits)


def test_the_search_refuses_pseudo_random_without_a_seed() -> None:
    bits = RNG.integers(0, 2, 256).astype(np.uint8)

    result = search_deinterleave(bits, "pseudo_random")

    assert result.limitation is not None
    assert "seed" in result.limitation
    assert np.array_equal(result.bits, bits), "unchanged bits, not a fabricated guess"


def test_the_search_inverts_pseudo_random_when_given_the_seed() -> None:
    bits = RNG.integers(0, 2, 256).astype(np.uint8)
    interleaved = pseudo_random_interleave(bits, 4321)

    result = search_deinterleave(interleaved, "pseudo_random", seed=4321)

    assert result.limitation is None
    assert result.parameters == {"seed": 4321}
    assert np.array_equal(result.bits, bits)


def test_an_empty_bitstream_is_handled_by_every_interleaver_path() -> None:
    empty = np.empty(0, dtype=np.uint8)

    assert pseudo_random_deinterleave(empty, 3).size == 0
    assert search_deinterleave(empty, "pseudo_random", seed=3).bits.size == 0
