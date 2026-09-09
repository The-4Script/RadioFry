"""PAM4 demodulation and dispatch (BANK.md Entry 035).

Entry 033 measured PAM4 as classified but never demodulated: it had no dispatch route at
all, so `demodulate_capture` fell through to "No demodulator is registered for PAM4".

The mapping is NOT invented here. `synthetic_gen/v1/modulation.py` builds the PAM4
constellation as `levels = [-3,-1,1,3]` normalised to unit average power, and
`bits_to_symbol_indices` packs bits as **natural binary, MSB-first** - the project-wide
convention since Entry 001. These tests pin that mapping against the generator itself.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.decoding.demodulators.common import DemodulationResult
from radiofry.decoding.demodulators.dispatch import demodulate_capture
from radiofry.decoding.demodulators.pam_demod import demodulate_pam
from radiofry.dsp.parameter_estimation import ParameterEstimate
from radiofry.dsp.preprocessing import preprocess
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import (
    constellation_for, bits_to_symbol_indices, modulate)

FS, N, SPS = 200_000.0, 8_192, 8


def _capture(snr_db=None, seed=401, modulation="PAM4", sps=SPS):
    spec = SampleSpec(modulation=modulation, num_symbols=N // sps, samples_per_symbol=sps,
                      sample_rate_hz=FS, snr_db=snr_db, seed=seed)
    bits = np.random.default_rng([seed, 1]).integers(0, 2, spec.num_bits, dtype=np.uint8)
    clean = modulate(bits, spec)
    iq = clean if snr_db is None else add_awgn(
        clean, snr_db, np.random.default_rng([seed, 2]))[0]
    return bits, preprocess(UnifiedSignalContainer(iq, FS))


def _ber(tx, rx):
    n = min(tx.size, rx.size)
    return float(np.mean(tx[:n] != rx[:n])) if n else 1.0


def _symbols(signal, offset=None):
    """Symbol-rate samples using the dispatch timing the production path would pick."""
    from radiofry.decoding.demodulators.dispatch import _linear_timing_offset
    o = _linear_timing_offset(signal.iq, SPS) if offset is None else offset
    return signal.iq[o::SPS]


# --- the mapping, pinned against the generator ------------------------------------------


def test_the_generator_constellation_is_four_real_levels() -> None:
    points = constellation_for("PAM4")

    assert points.size == 4
    assert np.allclose(points.imag, 0.0)
    assert np.allclose(np.sort(points.real), np.sort(np.array([-3, -1, 1, 3]) / np.sqrt(5)))


def test_the_mapping_is_natural_binary_msb_first_not_gray() -> None:
    # index 0..3 -> levels -3,-1,+1,+3 in order; bits are the 2-bit index, MSB first.
    points = constellation_for("PAM4")
    order = np.argsort(points.real)

    assert list(order) == [0, 1, 2, 3], "index order must already be ascending amplitude"
    assert list(bits_to_symbol_indices(
        np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8), 2)) == [0, 1, 2, 3]


def test_demodulator_round_trips_the_generator_mapping_exactly() -> None:
    points = constellation_for("PAM4")
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    symbols = points[bits_to_symbol_indices(bits, 2)]

    result = demodulate_pam(symbols, order=4)

    assert np.array_equal(result.bits, bits)


# --- the demodulator contract ----------------------------------------------------------------


def test_result_uses_the_shared_demodulation_contract() -> None:
    _, signal = _capture()

    result = demodulate_pam(_symbols(signal), order=4)

    assert isinstance(result, DemodulationResult)
    assert result.modulation == "PAM4"
    assert result.metadata["order"] == 4
    assert result.bits.dtype == np.uint8
    assert set(np.unique(result.bits)) <= {0, 1}
    assert result.bits.size == 2 * result.symbols.size


def test_an_unsupported_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="PAM order"):
        demodulate_pam(np.zeros(8, dtype=np.complex64), order=8)


# --- accuracy across SNR ------------------------------------------------------------------------


def test_noiseless_pam4_recovers_every_bit() -> None:
    bits, signal = _capture(snr_db=None)

    result = demodulate_pam(_symbols(signal), order=4)

    assert _ber(bits, result.bits) == 0.0


@pytest.mark.parametrize("snr_db,max_ber", [(20.0, 0.001), (15.0, 0.01),
                                            (10.0, 0.06), (5.0, 0.20)])
def test_ber_degrades_gracefully_with_snr(snr_db: float, max_ber: float) -> None:
    bits, signal = _capture(snr_db=snr_db)

    result = demodulate_pam(_symbols(signal), order=4)

    assert _ber(bits, result.bits) <= max_ber


@pytest.mark.parametrize("seed", [401, 409, 419, 421, 431])
def test_high_snr_accuracy_holds_across_seeds(seed: int) -> None:
    bits, signal = _capture(snr_db=20.0, seed=seed)

    result = demodulate_pam(_symbols(signal), order=4)

    assert _ber(bits, result.bits) <= 0.001


def test_ber_is_monotone_in_snr() -> None:
    values = []
    for snr_db in (20.0, 15.0, 10.0, 5.0):
        bits, signal = _capture(snr_db=snr_db)
        values.append(_ber(bits, demodulate_pam(_symbols(signal), order=4).bits))

    assert values == sorted(values), values


# --- scale and rotation invariance -----------------------------------------------------------------


@pytest.mark.parametrize("gain", [1e-4, 0.01, 0.5, 1.0, 7.0, 1_000.0])
def test_demodulation_is_invariant_to_absolute_amplitude_scale(gain: float) -> None:
    # Upstream RMS normalisation means the absolute scale carries no information; the
    # QAM demodulator had exactly this defect (Entry 007).
    bits, signal = _capture(snr_db=20.0)
    symbols = _symbols(signal)

    baseline = demodulate_pam(symbols, order=4)
    scaled = demodulate_pam(symbols * gain, order=4)

    assert np.array_equal(baseline.bits, scaled.bits)


@pytest.mark.parametrize("degrees", [0.0, 17.0, 45.0, 90.0, 180.0, 245.0, 270.0])
def test_the_constellation_axis_is_recovered_under_any_rotation(degrees: float) -> None:
    """Levels are recovered exactly, up to an unresolvable 180-degree polarity flip.

    PAM4 is symmetric about the origin, so a blind receiver cannot tell level -3 from
    +3 without an external reference (differential coding, a pilot, or a known
    preamble). The axis estimator resolves the constellation LINE; the sign along it is
    genuinely not observable. Measured: index == true, or index == 3 - true, with
    nothing in between. BPSK has the same ambiguity and `demodulate_psk` does not
    resolve it either.
    """
    bits, signal = _capture(snr_db=20.0)
    symbols = _symbols(signal)
    rotated = symbols * np.exp(1j * np.deg2rad(degrees))
    truth = bits_to_symbol_indices(bits, 2)

    result = demodulate_pam(rotated, order=4)
    levels = [-3, -1, 1, 3]
    got = np.array([levels.index(int(round(x.real))) for x in result.symbols])
    n = min(got.size, truth.size)

    upright = float(np.mean(got[:n] == truth[:n]))
    flipped = float(np.mean(got[:n] == (3 - truth[:n])))
    assert max(upright, flipped) >= 0.999, f"{upright=} {flipped=}"


def test_polarity_is_correct_when_the_capture_is_not_rotated() -> None:
    # The ambiguity above must not become an excuse: with no rotation the sign is right.
    bits, signal = _capture(snr_db=20.0)

    assert _ber(bits, demodulate_pam(_symbols(signal), order=4).bits) <= 0.001


def test_no_ground_truth_is_needed_to_find_the_levels() -> None:
    import inspect
    from radiofry.decoding.demodulators import pam_demod

    signature = inspect.signature(pam_demod.demodulate_pam)

    assert list(signature.parameters) == ["samples", "order"]


# --- the dispatch route -----------------------------------------------------------------------------


def test_pam4_now_has_a_dispatch_route() -> None:
    bits, signal = _capture(snr_db=20.0)

    dispatched = demodulate_capture(signal, "PAM4", ParameterEstimate(None, None, FS / SPS))

    assert dispatched.available is True
    assert dispatched.result.modulation == "PAM4"
    assert "samples per symbol" in dispatched.message


def test_the_full_dispatch_path_recovers_bits_at_20_db() -> None:
    bits, signal = _capture(snr_db=20.0)

    dispatched = demodulate_capture(signal, "PAM4", ParameterEstimate(None, None, FS / SPS))

    assert _ber(bits, dispatched.result.bits) <= 0.01


def test_pam4_uses_the_existing_linear_timing_search_not_the_fsk_one() -> None:
    from radiofry.decoding.demodulators import dispatch as d

    assert "PAM4" not in d._FSK_LABELS
    assert "PAM4" not in d.ANALOG_LABELS


def test_pam4_still_requires_a_symbol_rate_like_every_digital_label() -> None:
    _, signal = _capture(snr_db=20.0)

    dispatched = demodulate_capture(signal, "PAM4", ParameterEstimate(None, None, None))

    assert dispatched.available is False
    assert "symbol-rate" in dispatched.message


# --- other labels must not leak into the PAM4 route ---------------------------------------------------


@pytest.mark.parametrize("label,expected", [
    ("BPSK", "2PSK"), ("QPSK", "4PSK"), ("QAM16", "QAM16"), ("CPFSK", "2FSK"),
])
def test_other_digital_labels_still_reach_their_own_demodulators(label, expected) -> None:
    _, signal = _capture(snr_db=20.0)

    dispatched = demodulate_capture(signal, label, ParameterEstimate(None, None, FS / SPS))

    assert dispatched.result.modulation == expected


def test_an_unregistered_label_is_still_rejected() -> None:
    _, signal = _capture(snr_db=20.0)

    dispatched = demodulate_capture(signal, "PAM8", ParameterEstimate(None, None, FS / SPS))

    assert dispatched.available is False
    assert "No demodulator is registered" in dispatched.message


def test_a_non_pam4_capture_routed_to_pam4_does_not_crash() -> None:
    # Misrouting must degrade to a bad BER, never an exception.
    _, signal = _capture(snr_db=20.0, modulation="QPSK")

    dispatched = demodulate_capture(signal, "PAM4", ParameterEstimate(None, None, FS / SPS))

    assert dispatched.available is True
    assert dispatched.result.bits.size > 0


# --- BER scoring integration --------------------------------------------------------------------------


def test_recovered_bits_reach_the_harness_ber_scorer() -> None:
    from radiofry.evaluation.harness import _score_report

    bits, signal = _capture(snr_db=20.0)
    dispatched = demodulate_capture(signal, "PAM4", ParameterEstimate(None, None, FS / SPS))
    record: dict = {"truth_symbol_rate_hz": FS / SPS, "truth_carrier_hz": 0.0,
                    "truth_snr_db": 20.0, "expected_family": "QAM-like",
                    "expected_cnn_label": "PAM4", "truth_interleaver": "none",
                    "truth_fec": "none"}
    report = {"stages": {"demodulation": {
        "available": True,
        "result": {"bits": dispatched.result.bits.tolist(), "modulation": "PAM4"}}}}

    _score_report(record, report, bits, has_bits=True)

    assert record["ber_status"] == "ok"
    assert record["ber_strict"] is not None
    assert record["ber_strict"] <= 0.01


# --- Entry 034 guard must be untouched -----------------------------------------------------------------


def test_the_entry_034_digital_family_guard_still_blocks_analog_labels() -> None:
    from radiofry.fusion.confidence_fusion import fuse_modulation

    result = fuse_modulation("WBFM", 0.99, "QAM-like", classical_confidence=0.60)

    assert result.label not in {"AM-DSB", "AM-SSB", "WBFM"}
    assert result.digital_family_block is True


def test_a_pam4_prediction_under_a_qam_family_is_still_accepted() -> None:
    from radiofry.fusion.confidence_fusion import fuse_modulation

    result = fuse_modulation("PAM4", 0.9, "QAM-like", classical_confidence=0.60)

    assert result.label == "PAM4"
    assert result.digital_family_block is False
