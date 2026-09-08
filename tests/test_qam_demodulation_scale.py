"""Regression tests for scale-invariant QAM demodulation.

Covers the defect measured in BANK.md Entry 006: the pipeline delivers unit-RMS
symbols while the decision grid assumes average power 10 (16QAM) / 42 (64QAM), so
every amplitude level collapsed onto the innermost pair and only the sign bits
survived.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.decoding.demodulators.qam_demod import demodulate_qam
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1 import SampleSpec, add_awgn, generate_source_bits, modulate

FS, SPS, NUM_SYMBOLS = 200_000.0, 8, 1_024
QAM = [("16QAM", 16), ("64QAM", 64)]


def _grid(order: int) -> np.ndarray:
    side = int(round(order**0.5))
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    return (levels[:, None] + 1j * levels[None, :]).ravel()


def _pipeline_symbols(modulation: str, snr_db: float | None, seed: int = 5):
    """Bits plus the symbols the demodulator actually receives in the real pipeline."""
    spec = SampleSpec(
        modulation=modulation,
        num_symbols=NUM_SYMBOLS,
        samples_per_symbol=SPS,
        sample_rate_hz=FS,
        snr_db=snr_db,
        seed=seed,
    )
    bits = generate_source_bits(spec)
    iq = add_awgn(modulate(bits, spec), snr_db, np.random.default_rng(seed))[0]
    processed = preprocess(UnifiedSignalContainer(iq, FS))
    return bits, processed.iq[::SPS][:NUM_SYMBOLS]


def _ber(truth: np.ndarray, recovered: np.ndarray) -> float:
    count = min(truth.size, recovered.size)
    return float(np.mean(truth[:count] != recovered[:count]))


# --- noiseless: must be exact ----------------------------------------------------


@pytest.mark.parametrize("modulation,order", QAM)
def test_noiseless_capture_recovers_the_exact_transmitted_bits(modulation: str, order: int) -> None:
    bits, symbols = _pipeline_symbols(modulation, None)

    result = demodulate_qam(symbols, order)

    np.testing.assert_array_equal(result.bits[: bits.size], bits)


# --- arbitrary positive amplitude scaling ----------------------------------------


@pytest.mark.parametrize("modulation,order", QAM)
@pytest.mark.parametrize("scale", [1e-4, 0.01, 0.3162, 1.0, 3.1623, 100.0, 1e4])
def test_demodulation_is_invariant_to_arbitrary_positive_scaling(
    modulation: str, order: int, scale: float
) -> None:
    bits, symbols = _pipeline_symbols(modulation, None)

    result = demodulate_qam(symbols * scale, order)

    np.testing.assert_array_equal(result.bits[: bits.size], bits)


@pytest.mark.parametrize("modulation,order", QAM)
def test_scaling_the_input_does_not_change_the_recovered_bits(modulation: str, order: int) -> None:
    _, symbols = _pipeline_symbols(modulation, 20.0)

    baseline = demodulate_qam(symbols, order).bits
    scaled = demodulate_qam(symbols * 12_345.0, order).bits
    shrunk = demodulate_qam(symbols * 1e-6, order).bits

    np.testing.assert_array_equal(baseline, scaled)
    np.testing.assert_array_equal(baseline, shrunk)


def test_symbols_already_on_the_decision_grid_still_decode_exactly() -> None:
    # Backward compatibility: input at the historical +/-1, +/-3 scale must still work.
    indices = np.arange(16)
    result = demodulate_qam(_grid(16)[indices], 16)

    expected = ((indices[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8).ravel()
    np.testing.assert_array_equal(result.bits, expected)


# --- noisy cases -----------------------------------------------------------------


def test_16qam_at_20db_recovers_the_exact_bits() -> None:
    bits, symbols = _pipeline_symbols("16QAM", 20.0)

    assert _ber(bits, demodulate_qam(symbols, 16).bits) == 0.0


def test_64qam_at_20db_has_a_low_bit_error_rate() -> None:
    bits, symbols = _pipeline_symbols("64QAM", 20.0)

    assert _ber(bits, demodulate_qam(symbols, 64).bits) < 0.05


def test_16qam_degrades_gracefully_at_10db() -> None:
    bits, symbols = _pipeline_symbols("16QAM", 10.0)

    assert _ber(bits, demodulate_qam(symbols, 16).bits) < 0.15


@pytest.mark.parametrize("modulation,order", QAM)
def test_bit_error_rate_improves_as_snr_rises(modulation: str, order: int) -> None:
    bits_high, symbols_high = _pipeline_symbols(modulation, 20.0)
    bits_low, symbols_low = _pipeline_symbols(modulation, 5.0)

    high = _ber(bits_high, demodulate_qam(symbols_high, order).bits)
    low = _ber(bits_low, demodulate_qam(symbols_low, order).bits)

    assert high < low


# --- preserved contract ----------------------------------------------------------


@pytest.mark.parametrize("modulation,order", QAM)
def test_bit_count_matches_symbol_count_times_bits_per_symbol(modulation: str, order: int) -> None:
    _, symbols = _pipeline_symbols(modulation, None)

    result = demodulate_qam(symbols, order)

    assert result.bits.size == symbols.size * int(np.log2(order))
    assert result.symbols.size == symbols.size
    assert result.modulation == f"QAM{order}"


def test_unsupported_order_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="QAM order"):
        demodulate_qam(np.zeros(8, dtype=np.complex64), 32)


def test_all_zero_input_does_not_divide_by_zero() -> None:
    result = demodulate_qam(np.zeros(16, dtype=np.complex64), 16)

    assert result.bits.size == 64
    assert np.all(np.isfinite(result.symbols))
