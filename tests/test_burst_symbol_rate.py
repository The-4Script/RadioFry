"""A bursty capture must not lock onto its own envelope instead of its symbol clock.

Real transmissions are rarely on for the whole capture - packets, TDMA slots, keyed
carriers. Any such gap puts a strong line at the burst envelope's fundamental (near fs/N)
with a dense harmonic series, and `_select_symbol_rate` rewards harmonic support. That
series outscored the real symbol-rate line, so a capture that was merely 75% duty reported
24 Hz against a true 25,000 Hz - a 99.9% error (BANK.md Entry 042).

The symbol-rate line was never missing; only the selection was wrong. Entry 039 measured
symbol rate as the master variable for BER (correct -> 0.0010, wrong -> 0.4881), so this
failure mode would have made every bursty real capture undemodulable.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.parameter_estimation import (
    MAX_SAMPLES_PER_SYMBOL,
    estimate_parameters,
)
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import generate_source_bits, modulate

FS = 200_000.0
SAMPLES = 8_192


def _capture(modulation: str = "QPSK", samples_per_symbol: int = 8,
             snr_db: float = 20.0, seed: int = 97_001) -> np.ndarray:
    spec = SampleSpec(modulation=modulation, num_symbols=SAMPLES // samples_per_symbol,
                      samples_per_symbol=samples_per_symbol, sample_rate_hz=FS,
                      snr_db=float(snr_db), seed=seed)
    clean = modulate(generate_source_bits(spec), spec)
    return np.asarray(add_awgn(clean, spec.snr_db,
                               np.random.default_rng([spec.seed, 2]))[0])


def _burst(capture: np.ndarray, duty: float) -> UnifiedSignalContainer:
    gated = capture.copy()
    gated[int(gated.size * duty):] = 0.0
    return preprocess(UnifiedSignalContainer(gated, FS, "iq"))


@pytest.mark.parametrize("duty", [1.0, 0.75, 0.5, 0.25, 0.10])
def test_the_symbol_rate_survives_any_duty_cycle(duty: float) -> None:
    """Was 24 Hz for every duty below 1.0; the true rate is 25,000 Hz."""
    estimate = estimate_parameters(_burst(_capture(), duty))

    assert estimate.symbol_rate_hz is not None
    assert estimate.symbol_rate_hz == pytest.approx(25_000.0, rel=0.10), (
        f"duty {duty:.0%}: got {estimate.symbol_rate_hz}")


def test_the_burst_envelope_line_is_not_selectable() -> None:
    """The envelope fundamental sits near fs/N; nothing that low may be chosen."""
    estimate = estimate_parameters(_burst(_capture(), 0.5))

    floor = FS / MAX_SAMPLES_PER_SYMBOL
    assert estimate.symbol_rate_hz > floor, (
        f"a rate below fs/{MAX_SAMPLES_PER_SYMBOL} = {floor:,.0f} Hz means the envelope "
        "was selected")
    assert estimate.symbol_rate_hz > 20 * (FS / SAMPLES), (
        "and specifically not the capture's own fundamental")


def test_a_continuous_capture_is_unaffected() -> None:
    """The repair must be inert for the case that already worked."""
    continuous = preprocess(UnifiedSignalContainer(_capture(), FS, "iq"))

    estimate = estimate_parameters(continuous)

    assert estimate.symbol_rate_hz == pytest.approx(25_000.0, rel=0.01)


@pytest.mark.parametrize("samples_per_symbol", [4, 8, 16, 32])
def test_every_supported_oversampling_factor_stays_above_the_floor(
        samples_per_symbol: int) -> None:
    """The floor must never exclude a rate the project actually generates."""
    symbol_rate = FS / samples_per_symbol

    assert symbol_rate > FS / MAX_SAMPLES_PER_SYMBOL, (
        f"sps {samples_per_symbol} produces {symbol_rate:,.0f} Hz, which the floor "
        "would reject")


def test_the_floor_is_generous_relative_to_the_validated_envelope() -> None:
    """Documented bound: 256 against a validated range of 4-32."""
    assert MAX_SAMPLES_PER_SYMBOL >= 8 * 32, (
        "the floor should sit well clear of the highest validated oversampling factor")


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "16QAM"])
def test_bursts_of_other_modulations_also_recover(modulation: str) -> None:
    estimate = estimate_parameters(_burst(_capture(modulation), 0.5))

    assert estimate.symbol_rate_hz == pytest.approx(25_000.0, rel=0.10)


def test_a_leading_gap_is_handled_as_well_as_a_trailing_one() -> None:
    """Real bursts do not conveniently start at sample zero."""
    capture = _capture()
    gated = capture.copy()
    gated[: gated.size // 2] = 0.0

    estimate = estimate_parameters(
        preprocess(UnifiedSignalContainer(gated, FS, "iq")))

    assert estimate.symbol_rate_hz == pytest.approx(25_000.0, rel=0.10)


def test_a_capture_that_is_entirely_silent_reports_nothing_rather_than_a_rate() -> None:
    silent = preprocess(UnifiedSignalContainer(
        np.zeros(SAMPLES, dtype=np.complex64), FS, "iq"))

    estimate = estimate_parameters(silent)

    assert estimate.symbol_rate_hz is None
