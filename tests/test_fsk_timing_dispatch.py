"""Regression tests for FSK-aware timing-offset selection in dispatch.

BANK.md Entry 013: the linear-modulation smoothness heuristic minimises
mean|diff| of the decimated samples, which for constant-envelope CPFSK favours a
boundary-straddling offset where the two tones cancel. FSK now selects the offset
that minimises instantaneous-frequency variance inside each candidate symbol.
"""

import re

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.decoding.demodulators.dispatch import (
    _fsk_timing_offset,
    _linear_timing_offset,
    demodulate_capture,
)
from radiofry.dsp.parameter_estimation import ParameterEstimate
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1 import SampleSpec, add_awgn, generate_source_bits, modulate

FS, SPS, NSYM = 200_000.0, 8, 1_024
PARAMS = ParameterEstimate(None, None, FS / SPS)


def _capture(modulation="BFSK", snr_db=20.0, seed=3, h=0.5):
    spec = SampleSpec(
        modulation=modulation, num_symbols=NSYM, samples_per_symbol=SPS,
        sample_rate_hz=FS, snr_db=snr_db, seed=seed,
        fsk_modulation_index=h if modulation == "BFSK" else None,
    )
    bits = generate_source_bits(spec)
    iq = add_awgn(modulate(bits, spec), snr_db, np.random.default_rng(seed))[0]
    return bits, preprocess(UnifiedSignalContainer(iq, FS))


def _offset_from(result) -> int:
    return int(re.search(r"offset (\d+)", result.message).group(1))


def _ber(bits, result) -> float:
    got = result.result.bits
    n = min(bits.size, got.size)
    return float(np.mean(bits[:n] != got[:n]))


# --- routing: which timing path is used ------------------------------------------


@pytest.mark.parametrize("label", ["CPFSK", "GFSK"])
def test_fsk_labels_use_the_fsk_specific_timing_path(label: str) -> None:
    _, proc = _capture()

    chosen = _offset_from(demodulate_capture(proc, label, PARAMS))

    assert chosen == _fsk_timing_offset(proc.iq, SPS)


@pytest.mark.parametrize("label,modulation", [("QPSK", "QPSK"), ("BPSK", "BPSK"), ("QAM16", "16QAM")])
def test_non_fsk_labels_keep_the_existing_timing_heuristic(label: str, modulation: str) -> None:
    _, proc = _capture(modulation=modulation)

    chosen = _offset_from(demodulate_capture(proc, label, PARAMS))

    assert chosen == _linear_timing_offset(proc.iq, SPS)


def test_the_two_timing_criteria_disagree_on_cpfsk() -> None:
    # If they agreed the fix would be untestable; the linear heuristic picks the
    # boundary-straddling offset this test guards against.
    _, proc = _capture()

    assert _fsk_timing_offset(proc.iq, SPS) != _linear_timing_offset(proc.iq, SPS)


# --- the criterion uses only the received signal ----------------------------------


def test_fsk_timing_selection_depends_only_on_the_signal_and_sps() -> None:
    _, proc = _capture()

    # Same waveform, no bits, no ground truth, no BER available to the function.
    assert _fsk_timing_offset(np.array(proc.iq, copy=True), SPS) == _fsk_timing_offset(proc.iq, SPS)


def test_fsk_timing_selection_is_deterministic() -> None:
    _, proc = _capture()

    assert len({_fsk_timing_offset(proc.iq, SPS) for _ in range(5)}) == 1


def test_dispatch_timing_choice_is_deterministic() -> None:
    _, proc = _capture()

    offsets = {_offset_from(demodulate_capture(proc, "CPFSK", PARAMS)) for _ in range(3)}

    assert len(offsets) == 1


# --- end-to-end through production dispatch ---------------------------------------


@pytest.mark.parametrize("snr_db", [None, 20.0, 10.0])
@pytest.mark.parametrize("seed", [3, 11])
def test_h_half_fsk_reaches_low_ber_through_production_dispatch(snr_db, seed: int) -> None:
    bits, proc = _capture(snr_db=snr_db, seed=seed, h=0.5)

    result = demodulate_capture(proc, "CPFSK", PARAMS)

    assert result.available
    assert _ber(bits, result) < 0.10


def test_h_half_fsk_beats_the_old_linear_heuristic_substantially() -> None:
    bits, proc = _capture(snr_db=20.0, seed=3, h=0.5)

    new = demodulate_capture(proc, "CPFSK", PARAMS)
    from radiofry.decoding.demodulators.fsk_demod import demodulate_fsk

    old_offset = _linear_timing_offset(proc.iq, SPS)
    old_bits = demodulate_fsk(proc.iq[old_offset::SPS], order=2).bits
    n = min(bits.size, old_bits.size)
    old_ber = float(np.mean(bits[:n] != old_bits[:n]))

    assert _ber(bits, new) < old_ber / 4


def test_h_one_remains_a_known_hard_case(bits_free=None) -> None:
    # The timing fix does not and cannot repair h=1.0: +/-pi per symbol is ambiguous.
    bits, proc = _capture(snr_db=20.0, seed=3, h=1.0)

    result = demodulate_capture(proc, "CPFSK", PARAMS)

    assert result.available
    assert _ber(bits, result) > 0.15


# --- non-FSK must not regress ------------------------------------------------------


@pytest.mark.parametrize("label,modulation", [("BPSK", "BPSK"), ("QPSK", "QPSK")])
def test_non_fsk_demodulation_is_unaffected(label: str, modulation: str) -> None:
    bits, proc = _capture(modulation=modulation, snr_db=20.0)

    result = demodulate_capture(proc, label, PARAMS)

    assert result.available
    assert _ber(bits, result) == 0.0


# --- degenerate inputs -------------------------------------------------------------


def test_single_sample_per_symbol_selects_offset_zero() -> None:
    rng = np.random.default_rng(0)
    iq = (rng.normal(size=256) + 1j * rng.normal(size=256)).astype(np.complex64)

    assert _fsk_timing_offset(iq, 1) == 0


def test_capture_shorter_than_one_symbol_does_not_crash() -> None:
    iq = np.ones(3, dtype=np.complex64)

    assert _fsk_timing_offset(iq, SPS) == 0
