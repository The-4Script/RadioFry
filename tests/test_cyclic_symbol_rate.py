"""Cyclostationary symbol-rate estimation for FSK (BANK.md Entry 038).

Entry 037 measured that the transition-based features lose the symbol-rate line above
samples-per-symbol 8. The cyclic autocorrelation of the instantaneous frequency keeps it
for CPFSK/BFSK, because a rectangular frequency pulse has the excess bandwidth an
alpha = Rs cyclic feature requires.

It does NOT work for GFSK - Gaussian shaping removes that excess bandwidth by design -
so the estimate is gated on a signal-derived peak-to-background ratio and the existing
estimator is kept whenever the cyclic evidence is weak.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.parameter_estimation import (
    CYCLIC_PEAK_RATIO_MIN, _cyclic_symbol_rate, estimate_parameters)
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import modulate

FS, N = 200_000.0, 8_192


def _capture(mod, sps, snr_db=20.0, seed=503):
    spec = SampleSpec(modulation=mod, num_symbols=N // sps, samples_per_symbol=sps,
                      sample_rate_hz=FS, snr_db=snr_db, seed=seed)
    bits = np.random.default_rng([seed, 1]).integers(0, 2, spec.num_bits, dtype=np.uint8)
    iq, _ = add_awgn(modulate(bits, spec), snr_db, np.random.default_rng([seed, 2]))
    return preprocess(UnifiedSignalContainer(iq, FS))


def test_the_gate_threshold_is_published() -> None:
    assert CYCLIC_PEAK_RATIO_MIN == 6.0


def test_the_estimator_takes_only_samples_and_sample_rate() -> None:
    import inspect
    assert list(inspect.signature(_cyclic_symbol_rate).parameters) == ["iq", "sample_rate"]


@pytest.mark.parametrize("sps", [16, 32])
def test_cyclic_estimate_recovers_cpfsk_symbol_rate_above_sps_8(sps: int) -> None:
    rate, ratio = _cyclic_symbol_rate(_capture("BFSK", sps).iq, FS)

    assert rate == pytest.approx(FS / sps, rel=0.05)
    assert ratio > 1.0


def test_production_now_estimates_cpfsk_symbol_rate_at_sps_16() -> None:
    estimate = estimate_parameters(_capture("BFSK", 16))

    assert estimate.symbol_rate_hz == pytest.approx(FS / 16, rel=0.05)
    assert estimate.symbol_rate_feature == "cyclic_autocorrelation"


def test_cpfsk_sps_32_is_recovered_by_the_estimator_but_not_by_the_gate() -> None:
    """A documented limitation, pinned so it is not mistaken for a win.

    The raw cyclic estimator finds the sps=32 rate, but its peak-to-background ratio
    (~2.8x) sits below the zero-regression gate of 6.0, so production keeps the older
    estimate. Lowering the gate to capture this case regressed 4-8 other captures in the
    Entry 038 sweep, so it was not lowered.
    """
    signal = _capture("BFSK", 32)

    rate, ratio = _cyclic_symbol_rate(signal.iq, FS)

    assert rate == pytest.approx(FS / 32, rel=0.05), "the raw estimator does find it"
    assert ratio < CYCLIC_PEAK_RATIO_MIN, "but it does not clear the gate"
    assert estimate_parameters(signal).symbol_rate_feature != "cyclic_autocorrelation"


@pytest.mark.parametrize("sps", [4, 8])
def test_cpfsk_below_sps_16_is_unchanged_and_still_correct(sps: int) -> None:
    estimate = estimate_parameters(_capture("BFSK", sps))

    assert estimate.symbol_rate_hz == pytest.approx(FS / sps, rel=0.05)


def test_gfsk_is_not_broken_further_where_it_used_to_work() -> None:
    # GFSK sps=4 at 20 dB is the one GFSK case the existing estimator gets right.
    # The gate must leave it alone: the cyclic evidence there is weak by design.
    estimate = estimate_parameters(_capture("GFSK", 4))

    assert estimate.symbol_rate_hz == pytest.approx(FS / 4, rel=0.05)


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK", "PAM4", "16QAM", "64QAM"])
@pytest.mark.parametrize("sps", [8, 16])
def test_linear_modulations_are_not_regressed(modulation: str, sps: int) -> None:
    # These were already essentially perfect (Entry 037); the gate must not disturb them.
    estimate = estimate_parameters(_capture(modulation, sps))

    assert estimate.symbol_rate_hz == pytest.approx(FS / sps, rel=0.05)


def test_the_feature_name_records_when_the_cyclic_path_was_used() -> None:
    estimate = estimate_parameters(_capture("BFSK", 16))

    assert estimate.symbol_rate_feature == "cyclic_autocorrelation"


def test_a_weak_cyclic_feature_leaves_the_existing_estimator_in_charge() -> None:
    # GFSK: the cyclic ratio stays below the gate, so the feature name must NOT be the
    # cyclic one - the existing adaptive features remain responsible.
    estimate = estimate_parameters(_capture("GFSK", 32))

    assert estimate.symbol_rate_feature != "cyclic_autocorrelation"


def test_short_input_is_handled() -> None:
    rate, ratio = _cyclic_symbol_rate(np.ones(4, dtype=np.complex64), FS)

    assert rate is None
    assert ratio == 0.0


def test_the_estimate_is_scale_invariant() -> None:
    signal = _capture("BFSK", 16)
    a, _ = _cyclic_symbol_rate(signal.iq, FS)
    b, _ = _cyclic_symbol_rate(signal.iq * 1_000.0, FS)

    assert a == pytest.approx(b, rel=1e-6)
