"""Mathematical correctness tests for the Synthetic Dataset V1 modulator core.

The reference constellations and de-mappers in this module are written directly
from the documented V1 convention (natural-binary, MSB-first symbol indices) so
they act as an independent oracle rather than re-using generator internals.
"""

import numpy as np
import pytest

from radiofry.synthetic_gen.v1 import SampleSpec, generate_source_bits, modulate
from radiofry.synthetic_gen.v1.channel import add_awgn

LINEAR_MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
ALL_MODULATIONS = LINEAR_MODULATIONS + ["BFSK"]


def _reference_constellation(modulation: str) -> np.ndarray:
    """Unit-average-power constellation indexed by natural-binary symbol index."""

    if modulation in {"BPSK", "QPSK", "8PSK"}:
        order = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[modulation]
        return np.exp(2j * np.pi * np.arange(order) / order)
    order = {"16QAM": 16, "64QAM": 64}[modulation]
    side = int(round(order**0.5))
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    points = np.array([levels[i] + 1j * levels[q] for i in range(side) for q in range(side)])
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def _index_to_bits(indices: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    shifts = np.arange(bits_per_symbol - 1, -1, -1)
    return ((indices[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def _reference_demap(iq: np.ndarray, spec: SampleSpec) -> np.ndarray:
    """Independent noiseless de-mapper used only as a test oracle."""

    sps = spec.samples_per_symbol
    if spec.modulation == "BFSK":
        increments = np.diff(np.unwrap(np.angle(iq)))
        bits = [
            1 if increments[n * sps : (n + 1) * sps - 1].mean() > 0 else 0
            for n in range(spec.num_symbols)
        ]
        return np.asarray(bits, dtype=np.uint8)
    constellation = _reference_constellation(spec.modulation)
    symbols = iq[sps // 2 :: sps][: spec.num_symbols]
    indices = np.argmin(np.abs(symbols[:, None] - constellation[None, :]), axis=1).astype(np.uint16)
    return _index_to_bits(indices, spec.bits_per_symbol)


def _spec(modulation: str, **overrides) -> SampleSpec:
    params = {
        "modulation": modulation,
        "num_symbols": 512,
        "samples_per_symbol": 8,
        "sample_rate_hz": 200_000.0,
        "snr_db": None,
        "seed": 1234,
    }
    params.update(overrides)
    return SampleSpec(**params)


# --- symbol mapping conventions -------------------------------------------------


def test_bpsk_maps_bit_zero_to_plus_one_and_bit_one_to_minus_one() -> None:
    spec = _spec("BPSK", num_symbols=2)

    waveform = modulate(np.array([0, 1], dtype=np.uint8), spec)

    np.testing.assert_allclose(waveform[0], 1 + 0j, atol=1e-6)
    np.testing.assert_allclose(waveform[spec.samples_per_symbol], -1 + 0j, atol=1e-6)


def test_qpsk_maps_natural_binary_indices_to_quadrant_phases() -> None:
    spec = _spec("QPSK", num_symbols=4)
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)

    waveform = modulate(bits, spec)
    symbols = waveform[:: spec.samples_per_symbol]

    np.testing.assert_allclose(symbols, [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], atol=1e-6)


def test_qam16_maps_bits_to_normalized_integer_grid_point() -> None:
    spec = _spec("16QAM", num_symbols=2)
    # index 0 -> (real_index 0, imag_index 0) -> -3-3j ; index 6 -> (1, 2) -> -1+1j
    bits = np.array([0, 0, 0, 0, 0, 1, 1, 0], dtype=np.uint8)

    symbols = modulate(bits, spec)[:: spec.samples_per_symbol]

    scale = np.sqrt(10.0)
    np.testing.assert_allclose(symbols[0], (-3 - 3j) / scale, atol=1e-6)
    np.testing.assert_allclose(symbols[1], (-1 + 1j) / scale, atol=1e-6)


@pytest.mark.parametrize("modulation", LINEAR_MODULATIONS)
def test_constellation_has_unit_average_symbol_power(modulation: str) -> None:
    spec = _spec(modulation, num_symbols=4096)
    bits = generate_source_bits(spec)

    waveform = modulate(bits, spec)

    np.testing.assert_allclose(np.mean(np.abs(waveform) ** 2), 1.0, rtol=0.05)


@pytest.mark.parametrize("modulation", LINEAR_MODULATIONS)
def test_every_constellation_point_is_reachable(modulation: str) -> None:
    spec = _spec(modulation, num_symbols=4096)
    constellation = _reference_constellation(modulation)

    symbols = modulate(generate_source_bits(spec), spec)[:: spec.samples_per_symbol]
    used = np.unique(np.argmin(np.abs(symbols[:, None] - constellation[None, :]), axis=1))

    assert used.size == constellation.size


# --- waveform structure ---------------------------------------------------------


@pytest.mark.parametrize("modulation", ALL_MODULATIONS)
def test_waveform_length_is_symbols_times_samples_per_symbol(modulation: str) -> None:
    spec = _spec(modulation, num_symbols=100, samples_per_symbol=6, sample_rate_hz=None, symbol_rate_hz=1_000.0)

    waveform = modulate(generate_source_bits(spec), spec)

    assert waveform.size == 600
    assert waveform.dtype == np.complex64


@pytest.mark.parametrize("modulation", ALL_MODULATIONS)
def test_noiseless_waveform_demaps_back_to_the_exact_source_bits(modulation: str) -> None:
    spec = _spec(modulation, num_symbols=256)
    bits = generate_source_bits(spec)

    recovered = _reference_demap(modulate(bits, spec), spec)

    np.testing.assert_array_equal(recovered, bits)


def test_bfsk_instantaneous_frequency_matches_the_configured_deviation() -> None:
    spec = _spec("BFSK", num_symbols=64)
    bits = generate_source_bits(spec)

    waveform = modulate(bits, spec)
    increments = np.diff(np.unwrap(np.angle(waveform)))
    expected = 2 * np.pi * spec.fsk_deviation_hz / spec.sample_rate_hz

    np.testing.assert_allclose(np.abs(increments), expected, atol=1e-5)


def test_bfsk_has_constant_envelope_and_continuous_phase() -> None:
    spec = _spec("BFSK", num_symbols=64)

    waveform = modulate(generate_source_bits(spec), spec)

    np.testing.assert_allclose(np.abs(waveform), 1.0, atol=1e-6)
    step = 2 * np.pi * spec.fsk_deviation_hz / spec.sample_rate_hz
    assert np.max(np.abs(np.diff(np.unwrap(np.angle(waveform))))) <= step + 1e-5


def test_bfsk_defaults_to_modulation_index_one_half() -> None:
    # h = 2*deviation/Rs. h=0.5 (deviation Rs/4) is the standard CPFSK/MSK index and the
    # one the shipped checkpoint was trained on; see BANK.md Entry 011.
    spec = _spec("BFSK", samples_per_symbol=8, sample_rate_hz=200_000.0)

    assert spec.symbol_rate_hz == 25_000.0
    assert spec.fsk_deviation_hz == 6_250.0
    assert spec.fsk_modulation_index == 0.5


def test_explicit_modulation_index_one_sets_the_known_hard_deviation() -> None:
    spec = _spec("BFSK", fsk_modulation_index=1.0)

    assert spec.fsk_deviation_hz == 12_500.0
    assert spec.fsk_modulation_index == 1.0


def test_explicit_deviation_still_wins_and_reports_its_index() -> None:
    spec = _spec("BFSK", fsk_deviation_hz=3_125.0)

    assert spec.fsk_deviation_hz == 3_125.0
    assert spec.fsk_modulation_index == 0.25


def test_supplying_both_deviation_and_index_inconsistently_is_rejected() -> None:
    with pytest.raises(ValueError, match="fsk_modulation_index"):
        _spec("BFSK", fsk_deviation_hz=12_500.0, fsk_modulation_index=0.5)


def test_supplying_both_consistently_is_accepted() -> None:
    spec = _spec("BFSK", fsk_deviation_hz=12_500.0, fsk_modulation_index=1.0)

    assert spec.fsk_deviation_hz == 12_500.0


def test_non_fsk_modulations_report_no_modulation_index() -> None:
    assert _spec("QPSK").fsk_modulation_index is None
    assert _spec("16QAM").fsk_modulation_index is None


# --- rate resolution ------------------------------------------------------------


def test_symbol_rate_is_derived_from_sample_rate_and_samples_per_symbol() -> None:
    spec = _spec("BPSK", sample_rate_hz=48_000.0, samples_per_symbol=4, symbol_rate_hz=None)

    assert spec.symbol_rate_hz == 12_000.0


def test_sample_rate_is_derived_from_symbol_rate_and_samples_per_symbol() -> None:
    spec = _spec("BPSK", sample_rate_hz=None, symbol_rate_hz=9_600.0, samples_per_symbol=10)

    assert spec.sample_rate_hz == 96_000.0


def test_samples_per_symbol_is_derived_from_the_two_rates() -> None:
    spec = _spec("BPSK", sample_rate_hz=96_000.0, symbol_rate_hz=12_000.0, samples_per_symbol=None)

    assert spec.samples_per_symbol == 8


def test_inconsistent_rate_triple_is_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        _spec("BPSK", sample_rate_hz=100_000.0, symbol_rate_hz=12_000.0, samples_per_symbol=8)


def test_non_integer_samples_per_symbol_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        _spec("BPSK", sample_rate_hz=100_000.0, symbol_rate_hz=30_000.0, samples_per_symbol=None)


def test_unknown_modulation_is_rejected() -> None:
    with pytest.raises(ValueError, match="modulation"):
        _spec("FM")


def test_underdetermined_rates_are_rejected() -> None:
    with pytest.raises(ValueError, match="two of"):
        _spec("BPSK", sample_rate_hz=None, symbol_rate_hz=None, samples_per_symbol=8)


# --- reproducibility ------------------------------------------------------------


def test_source_bits_are_reproducible_for_a_given_seed() -> None:
    first = generate_source_bits(_spec("QPSK", seed=99))
    second = generate_source_bits(_spec("QPSK", seed=99))

    np.testing.assert_array_equal(first, second)


def test_source_bits_change_with_the_bits_seed() -> None:
    first = generate_source_bits(_spec("QPSK", seed=99))
    second = generate_source_bits(_spec("QPSK", seed=100))

    assert not np.array_equal(first, second)


def test_source_bits_are_independent_of_the_snr_so_a_sweep_shares_one_payload() -> None:
    high = _spec("QPSK", seed=5, bits_seed=42, snr_db=20.0)
    low = _spec("QPSK", seed=6, bits_seed=42, snr_db=0.0)

    np.testing.assert_array_equal(generate_source_bits(high), generate_source_bits(low))


def test_source_bit_count_is_symbols_times_bits_per_symbol() -> None:
    spec = _spec("64QAM", num_symbols=100)

    assert generate_source_bits(spec).size == 600
    assert spec.bits_per_symbol == 6


# --- AWGN channel ---------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [20.0, 15.0, 10.0, 5.0, 0.0])
def test_awgn_realizes_the_requested_snr_within_a_tenth_of_a_decibel(snr_db: float) -> None:
    spec = _spec("QPSK", num_symbols=20_000, snr_db=snr_db)
    clean = modulate(generate_source_bits(spec), spec)

    noisy, report = add_awgn(clean, snr_db, np.random.default_rng(3))

    measured = 10 * np.log10(report.signal_power / report.realized_noise_power)
    assert abs(measured - snr_db) < 0.1
    measured_from_arrays = 10 * np.log10(
        np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noisy - clean) ** 2)
    )
    assert abs(measured_from_arrays - snr_db) < 0.1


def test_awgn_noise_is_circularly_symmetric_with_equal_quadrature_variance() -> None:
    spec = _spec("BPSK", num_symbols=20_000, snr_db=10.0)
    clean = modulate(generate_source_bits(spec), spec)

    noisy, _ = add_awgn(clean, 10.0, np.random.default_rng(11))
    noise = noisy - clean

    assert abs(np.var(noise.real) / np.var(noise.imag) - 1.0) < 0.05
    assert abs(np.mean(noise)) < 0.05


def test_awgn_is_reproducible_for_a_given_generator_seed() -> None:
    spec = _spec("BPSK", num_symbols=1_000, snr_db=5.0)
    clean = modulate(generate_source_bits(spec), spec)

    first, _ = add_awgn(clean, 5.0, np.random.default_rng(7))
    second, _ = add_awgn(clean, 5.0, np.random.default_rng(7))
    third, _ = add_awgn(clean, 5.0, np.random.default_rng(8))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_awgn_with_no_snr_returns_the_clean_signal_untouched() -> None:
    spec = _spec("8PSK", num_symbols=64)
    clean = modulate(generate_source_bits(spec), spec)

    noisy, report = add_awgn(clean, None, np.random.default_rng(1))

    np.testing.assert_array_equal(noisy, clean)
    assert report.realized_noise_power == 0.0
    assert report.noise_type == "none"
