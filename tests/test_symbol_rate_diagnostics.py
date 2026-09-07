"""Diagnostic evidence for the Synthetic V1 symbol-rate estimation failure.

These tests assert properties of the *signals* and of candidate spectral
features - not of the estimator's current output - so they stay valid after any
future fix to `dsp/parameter_estimation.py`. They document why the existing
envelope-squaring feature cannot recover a rectangular-pulse symbol rate.

Reference: BANK.md Entry 003.
"""

import numpy as np
import pytest

from radiofry.synthetic_gen.v1 import SampleSpec, generate_source_bits, modulate

FS = 200_000.0
SPS = 8
NUM_SYMBOLS = 1_024
TRUE_SYMBOL_RATE = FS / SPS  # 25 kHz, and an exact FFT bin for this capture length

CONSTANT_MODULUS = ["BPSK", "QPSK", "8PSK", "BFSK"]
AMPLITUDE_BEARING = ["16QAM", "64QAM"]


def _capture(modulation: str, *, fsk_deviation_hz: float | None = None) -> np.ndarray:
    spec = SampleSpec(
        modulation=modulation,
        num_symbols=NUM_SYMBOLS,
        samples_per_symbol=SPS,
        sample_rate_hz=FS,
        snr_db=None,
        seed=11,
        fsk_deviation_hz=fsk_deviation_hz,
    )
    return modulate(generate_source_bits(spec), spec)


def _feature_spectrum(iq: np.ndarray, feature) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of the estimator's FFT stage (parameter_estimation.py:91-95)."""
    centered = iq - np.mean(iq)
    powered = feature(centered)
    spectrum = np.abs(np.fft.rfft(powered - np.mean(powered)))
    rates = np.fft.rfftfreq(powered.size, d=1 / FS)
    spectrum[0] = 0.0
    return rates, spectrum


def _envelope_power(x: np.ndarray) -> np.ndarray:
    return np.abs(x) ** 2


def _instantaneous_frequency_step(x: np.ndarray) -> np.ndarray:
    frequency = np.diff(np.unwrap(np.angle(x)), prepend=0.0)
    return np.diff(frequency, prepend=0.0) ** 2


@pytest.mark.parametrize("modulation", CONSTANT_MODULUS)
def test_constant_modulus_v1_signals_carry_no_envelope_power_variation(modulation: str) -> None:
    # With rectangular pulses these constellations have a perfectly flat envelope,
    # so squaring the magnitude yields a constant - it cannot encode symbol timing.
    envelope = np.abs(_capture(modulation)) ** 2

    assert np.var(envelope) < 1e-12


@pytest.mark.parametrize("modulation", AMPLITUDE_BEARING)
def test_amplitude_bearing_v1_signals_do_vary_in_envelope_power(modulation: str) -> None:
    envelope = np.abs(_capture(modulation)) ** 2

    assert np.var(envelope) > 0.1


# BFSK is excluded: it is a frequency modulation, so after the estimator's DC removal
# its envelope carries a tone at twice the frequency deviation. See the dedicated test
# below - that tone is not a symbol-rate line even though V1's default deviation of
# Rs/2 makes the two coincide.
@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK"] + AMPLITUDE_BEARING)
def test_envelope_power_spectrum_is_a_null_at_the_true_symbol_rate(modulation: str) -> None:
    # Full-duty rectangular pulses do not overlap, so |x|^2 is a piecewise-constant
    # NRZ waveform whose spectrum is sinc-shaped with exact zeros at every multiple
    # of the symbol rate. The estimator searches for a peak exactly where the
    # feature is guaranteed to have none.
    rates, spectrum = _feature_spectrum(_capture(modulation), _envelope_power)
    bin_index = int(np.argmin(np.abs(rates - TRUE_SYMBOL_RATE)))
    neighbourhood = spectrum[max(1, bin_index - 200) : bin_index + 200]

    assert rates[bin_index] == pytest.approx(TRUE_SYMBOL_RATE)
    assert spectrum[bin_index] <= np.median(neighbourhood) * 1e-3


def test_bfsk_envelope_line_tracks_twice_the_deviation_not_the_symbol_rate() -> None:
    # Separating the two: with deviation Rs/8, twice the deviation is 6250 Hz while
    # the symbol rate stays 25 kHz. The envelope tone follows the deviation, so any
    # apparent BFSK "symbol-rate" detection in V1 is an artefact of the default
    # deviation being exactly Rs/2.
    deviation = TRUE_SYMBOL_RATE / 8
    rates, spectrum = _feature_spectrum(_capture("BFSK", fsk_deviation_hz=deviation), _envelope_power)
    floor = float(np.median(spectrum[1:]))
    at = lambda frequency: float(spectrum[int(np.argmin(np.abs(rates - frequency)))])

    assert at(2 * deviation) > 10 * floor
    assert at(TRUE_SYMBOL_RATE) < 10 * floor


@pytest.mark.parametrize("modulation", CONSTANT_MODULUS + AMPLITUDE_BEARING)
def test_instantaneous_frequency_step_feature_peaks_at_the_true_symbol_rate(modulation: str) -> None:
    # Symbol boundaries are phase discontinuities, so the second difference of the
    # unwrapped phase is an impulse train at the symbol rate for every V1 modulation.
    rates, spectrum = _feature_spectrum(_capture(modulation), _instantaneous_frequency_step)
    peak_rate = float(rates[int(np.argmax(spectrum))])

    assert peak_rate == pytest.approx(TRUE_SYMBOL_RATE, rel=0.01)


def test_envelope_feature_peak_is_far_from_the_true_symbol_rate_for_qam() -> None:
    # The surviving energy sits in the low-frequency main lobe, which is why the
    # estimator is biased far below the truth rather than randomly scattered.
    rates, spectrum = _feature_spectrum(_capture("16QAM"), _envelope_power)
    peak_rate = float(rates[int(np.argmax(spectrum))])

    assert peak_rate < 0.5 * TRUE_SYMBOL_RATE
