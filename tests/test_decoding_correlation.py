import numpy as np
import pytest
import re

from radiofry.correlation.bitstream_correlation import correlate_bitstream
from radiofry.decoding.demodulators.psk_demod import demodulate_psk
from radiofry.decoding.demodulators.fsk_demod import demodulate_fsk
from radiofry.decoding.demodulators.analog_demod import demodulate_ssb
from radiofry.decoding.demodulators.dispatch import demodulate_capture
from radiofry.decoding.demodulators import dispatch as dispatch_module
from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.parameter_estimation import ParameterEstimate
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
    assert np.iscomplexobj(result.symbols)
    np.testing.assert_allclose(np.abs(result.symbols), 1.0)


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


def test_fsk_rejects_unsupported_orders() -> None:
    with pytest.raises(ValueError, match="binary FSK"):
        demodulate_fsk(np.ones(8, dtype=np.complex64), order=4)


def test_ssb_uses_product_detector() -> None:
    sample_rate = 8_000
    time = np.arange(8_000) / sample_rate
    audio = np.sin(2 * np.pi * 40 * time)
    samples = np.exp(2j * np.pi * 500 * time) * audio

    recovered = demodulate_ssb(samples, sample_rate, 500)

    assert np.corrcoef(recovered[100:], audio[100:])[0, 1] > 0.9


def test_dispatch_routes_ssb_and_reports_timing_search() -> None:
    sample_rate = 8_000
    time = np.arange(800) / sample_rate
    audio = np.sin(2 * np.pi * 40 * time)
    signal = UnifiedSignalContainer(np.exp(2j * np.pi * 500 * time) * audio, sample_rate)
    parameters = ParameterEstimate(100.0, 20.0, 1_000.0, carrier_frequency_hz=500.0)

    dispatched = demodulate_capture(signal, "AM-SSB", parameters)

    assert dispatched.available
    assert dispatched.result is not None
    assert dispatched.result.modulation == "AM-SSB"
    assert "coarse timing search" in dispatched.message
    offset = int(re.search(r"offset (\d+)", dispatched.message).group(1))
    expected_audio = audio[offset::8][:dispatched.result.symbols.size]
    assert np.corrcoef(dispatched.result.symbols[10:], expected_audio[10:])[0, 1] > 0.99


def test_dispatch_keeps_dsb_and_ssb_demodulators_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    signal = UnifiedSignalContainer(np.ones(32, dtype=np.complex64), 8_000)
    parameters = ParameterEstimate(100.0, 20.0, 1_000.0, carrier_frequency_hz=500.0)
    calls: list[str] = []

    monkeypatch.setattr(dispatch_module, "demodulate_am", lambda samples: calls.append("dsb") or np.array([1.0], dtype=np.float32))
    monkeypatch.setattr(dispatch_module, "demodulate_ssb", lambda samples, sample_rate, carrier: calls.append("ssb") or np.array([2.0], dtype=np.float32))

    demodulate_capture(signal, "AM-DSB", parameters)
    demodulate_capture(signal, "AM-SSB", parameters)

    assert calls == ["dsb", "ssb"]


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


def test_reed_solomon_failure_is_reported_without_crashing() -> None:
    result = decode_reed_solomon(bytes((index * 37) % 256 for index in range(64)))

    assert not result.success
    assert "failed" in result.message.lower()


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


def test_correlation_finds_ccsds_asm() -> None:
    marker = "00011010110011111111110000011101"
    bits = np.array([1, 0] + [int(bit) for bit in marker] + [1, 1], dtype=np.uint8)

    result = correlate_bitstream(bits, sync_library={"ccsds": marker})

    assert result.sync_pattern == "ccsds"
    assert result.sync_positions == (2,)


def test_decode_dispatch_preserves_bits_when_fec_is_none() -> None:
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)

    result = decode_fec(bits, "none")

    assert result.success
    np.testing.assert_array_equal(result.bits, bits)


def test_ldpc_is_explicitly_classification_only() -> None:
    result = decode_fec(np.array([0, 1, 1, 0], dtype=np.uint8), "ldpc")

    assert not result.success
    assert "code parameters" in result.message


def test_deinterleave_search_reports_pseudorandom_limitation() -> None:
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)

    result = search_deinterleave(bits, "pseudo_random")

    assert result.limitation is not None
