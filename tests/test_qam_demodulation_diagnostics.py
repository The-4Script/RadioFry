"""Diagnostic evidence for the V1 QAM demodulation failure.

These assert conventions and target behaviour that stay true after any future fix:
the scale the pipeline delivers, the scale the demodulator's grid assumes, and the
fact that the demodulator is arithmetically correct once the two agree.

Reference: BANK.md Entry 006.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.decoding.demodulators.qam_demod import demodulate_qam
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1 import (
    SampleSpec,
    bits_to_symbol_indices,
    constellation_for,
    generate_source_bits,
    modulate,
)

FS, SPS, NUM_SYMBOLS = 200_000.0, 8, 1_024
QAM = [("16QAM", 16), ("64QAM", 64)]


def _grid(order: int) -> np.ndarray:
    """The integer constellation qam_demod.py slices against."""
    side = int(round(order**0.5))
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    return (levels[:, None] + 1j * levels[None, :]).ravel()


def _capture(modulation: str, seed: int = 5):
    spec = SampleSpec(
        modulation=modulation,
        num_symbols=NUM_SYMBOLS,
        samples_per_symbol=SPS,
        sample_rate_hz=FS,
        snr_db=None,
        seed=seed,
    )
    bits = generate_source_bits(spec)
    return bits, spec, modulate(bits, spec)


@pytest.mark.parametrize("modulation,order", QAM)
def test_v1_transmits_qam_at_unit_average_symbol_power(modulation: str, order: int) -> None:
    constellation = constellation_for(modulation)

    assert np.mean(np.abs(constellation) ** 2) == pytest.approx(1.0, rel=1e-9)
    assert constellation.size == order


@pytest.mark.parametrize("modulation,order", QAM)
def test_demodulator_grid_assumes_a_much_larger_average_power(modulation: str, order: int) -> None:
    # 10 for 16QAM and 42 for 64QAM: the scale gap that collapses the decision grid.
    expected = {16: 10.0, 64: 42.0}[order]

    assert np.mean(np.abs(_grid(order)) ** 2) == pytest.approx(expected)


@pytest.mark.parametrize("modulation,order", QAM)
def test_pipeline_delivers_unit_power_symbols_to_the_demodulator(modulation: str, order: int) -> None:
    _, _, iq = _capture(modulation)
    processed = preprocess(UnifiedSignalContainer(iq, FS))

    symbols = processed.iq[::SPS]

    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0, rel=0.05)


@pytest.mark.parametrize("modulation,order", QAM)
def test_symbol_decimation_recovers_the_transmitted_symbols(modulation: str, order: int) -> None:
    # Rules symbol timing out as a cause: the decimated samples are the symbols.
    bits, spec, iq = _capture(modulation)
    processed = preprocess(UnifiedSignalContainer(iq, FS))

    symbols = processed.iq[::SPS][:NUM_SYMBOLS]
    truth = constellation_for(modulation)[bits_to_symbol_indices(bits, spec.bits_per_symbol)]
    correlation = np.abs(np.vdot(truth, symbols)) / (np.linalg.norm(truth) * np.linalg.norm(symbols))

    assert correlation > 0.99


@pytest.mark.parametrize("modulation,order", QAM)
def test_demodulator_is_exact_once_the_symbol_scale_matches_its_grid(modulation: str, order: int) -> None:
    # The decision grid and the bit mapping are both correct; only the scale disagrees.
    bits, _, iq = _capture(modulation)
    processed = preprocess(UnifiedSignalContainer(iq, FS))
    grid_power = float(np.mean(np.abs(_grid(order)) ** 2))

    scaled = processed.iq[::SPS] * np.sqrt(grid_power)
    result = demodulate_qam(scaled, order)

    count = min(bits.size, result.bits.size)
    np.testing.assert_array_equal(result.bits[:count], bits[:count])


@pytest.mark.parametrize("modulation,order", QAM)
def test_bit_mapping_round_trips_through_the_demodulator_grid(modulation: str, order: int) -> None:
    # Confirms natural-binary MSB-first mapping is shared by generator and demodulator.
    indices = np.arange(order)
    result = demodulate_qam(_grid(order)[indices], order)

    bits_per_symbol = int(np.log2(order))
    expected = ((indices[:, None] >> np.arange(bits_per_symbol - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    np.testing.assert_array_equal(result.bits, expected)
