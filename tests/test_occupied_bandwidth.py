"""The reported occupied bandwidth must describe the signal, not the sample rate.

`estimate_parameters` used one power fraction (0.99) both to bound the signal band and to
decide what counted as out-of-band noise. Those two purposes pull in opposite directions:
at finite SNR the noise carries enough of the total power that a 0.99 criterion returns
very nearly the whole sampled band, so the reported bandwidth tracked the sample rate
instead of the symbol rate - ~182 kHz at fs = 200 kHz for every capture, 29x too wide at
samples-per-symbol 32 (BANK.md Entry 041).

The fix decouples the two fractions. These tests pin both halves: the bandwidth now
follows the symbol rate, and the SNR path - which needs the generous band - is unchanged.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.parameter_estimation import BANDWIDTH_FRACTION, estimate_parameters
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import generate_source_bits, modulate

FS = 200_000.0
SAMPLES = 8_192


def _capture(modulation: str, samples_per_symbol: int, snr_db: float,
             seed: int = 96_001) -> UnifiedSignalContainer:
    spec = SampleSpec(modulation=modulation, num_symbols=SAMPLES // samples_per_symbol,
                      samples_per_symbol=samples_per_symbol, sample_rate_hz=FS,
                      snr_db=float(snr_db), seed=seed)
    clean = modulate(generate_source_bits(spec), spec)
    noisy, _ = add_awgn(clean, spec.snr_db, np.random.default_rng([spec.seed, 2]))
    return preprocess(UnifiedSignalContainer(np.asarray(noisy), FS, "iq"))


@pytest.mark.parametrize("samples_per_symbol", [4, 8, 16, 32])
def test_the_bandwidth_follows_the_symbol_rate_not_the_sample_rate(
        samples_per_symbol: int) -> None:
    """A rectangular pulse's main lobe spans +/-Rs, so BW/Rs should be about 2.

    The pre-fix estimate gave 3.65 / 7.35 / 14.73 / 29.00 for sps 4/8/16/32 - it grew
    with oversampling because it was really measuring the sample rate.
    """
    symbol_rate = FS / samples_per_symbol

    estimate = estimate_parameters(_capture("QPSK", samples_per_symbol, 20.0))

    ratio = estimate.occupied_bandwidth_hz / symbol_rate
    assert 0.8 < ratio < 5.0, (
        f"sps {samples_per_symbol}: BW/Rs = {ratio:.2f}, expected roughly 2")


def test_the_bandwidth_is_no_longer_pinned_to_the_sample_rate() -> None:
    """The symptom that exposed the defect: every capture reported ~0.9 x fs."""
    narrow = estimate_parameters(_capture("QPSK", 32, 20.0)).occupied_bandwidth_hz
    wide = estimate_parameters(_capture("QPSK", 4, 20.0)).occupied_bandwidth_hz

    assert narrow < 0.5 * FS, "a 6.25 kHz-symbol-rate signal must not report ~fs"
    assert wide > narrow, "a higher symbol rate must report a wider band"


def test_a_wider_symbol_rate_reports_a_wider_band() -> None:
    """Monotonic in the thing it claims to measure."""
    widths = [estimate_parameters(_capture("QPSK", sps, 20.0)).occupied_bandwidth_hz
              for sps in (32, 16, 8, 4)]

    assert widths == sorted(widths), f"bandwidth must grow with symbol rate: {widths}"


def test_the_snr_estimate_still_uses_the_generous_band() -> None:
    """The noise floor needs a wide signal band, so SNR must not follow the bandwidth.

    Passing the tight fraction as `occupied_fraction` would visibly damage SNR; this pins
    that the two fractions stayed separate.
    """
    signal = _capture("QPSK", 8, 10.0)

    default = estimate_parameters(signal)
    coupled = estimate_parameters(signal, occupied_fraction=BANDWIDTH_FRACTION)

    assert abs(default.snr_db - 10.0) < 2.0, "SNR is accurate near 10 dB"
    assert abs(coupled.snr_db - 10.0) > abs(default.snr_db - 10.0), (
        "using the tight band for noise estimation must be worse, which is why they "
        "are separate parameters")


def test_changing_the_bandwidth_fraction_leaves_the_snr_untouched() -> None:
    signal = _capture("QPSK", 8, 15.0)

    wide = estimate_parameters(signal, bandwidth_fraction=0.95)
    tight = estimate_parameters(signal, bandwidth_fraction=0.70)

    assert wide.snr_db == pytest.approx(tight.snr_db), (
        "the reported bandwidth must not feed back into the SNR estimate")
    assert wide.occupied_bandwidth_hz > tight.occupied_bandwidth_hz


def test_the_carrier_estimate_is_unaffected() -> None:
    """The centroid is computed over the noise band, which did not change."""
    signal = _capture("QPSK", 8, 20.0)

    wide = estimate_parameters(signal, bandwidth_fraction=0.95)
    tight = estimate_parameters(signal, bandwidth_fraction=0.70)

    assert wide.carrier_frequency_hz == pytest.approx(tight.carrier_frequency_hz)


def test_degenerate_input_still_returns_nothing() -> None:
    empty = UnifiedSignalContainer(np.zeros(4, dtype=np.complex64), FS, "iq")

    estimate = estimate_parameters(empty)

    assert estimate.occupied_bandwidth_hz is None
    assert estimate.snr_db is None
