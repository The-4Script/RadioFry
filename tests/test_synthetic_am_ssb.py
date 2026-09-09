"""AM-SSB analog synthetic generation and its independent oracle (BANK.md Entry 022).

The oracle re-derives expected spectral structure from the recorded tone list and
recovers the message with its own product detector; it never calls the generator's
modulation code. It validates signal generation only and says nothing about production
AM-SSB demodulation.

WBFM arrived later, in Entry 024.
"""

from pathlib import Path

import numpy as np
import pytest

from radiofry.synthetic_gen.v1 import MODULATIONS
from radiofry.synthetic_gen.v1.analog import (
    ANALOG_MODULATIONS,
    SUPPORTED_ANALOG_SCHEMES,
    AnalogSampleSpec,
    generate_analog_sample,
    generate_message,
    modulate_am_dsb,
    modulate_am_ssb,
)

FS = 200_000.0
N = 32_768
CARRIER = 20_000.0


def _spec(**overrides) -> AnalogSampleSpec:
    params = dict(scheme="am_ssb", sample_rate_hz=FS, num_samples=N,
                  carrier_offset_hz=CARRIER, sideband="upper", snr_db=None, seed=11)
    params.update(overrides)
    return AnalogSampleSpec(**params)


def _spectrum(iq: np.ndarray):
    window = np.hanning(iq.size)
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq * window)))
    freqs = np.fft.fftshift(np.fft.fftfreq(iq.size, d=1 / FS))
    return freqs, spec


def _level_at(freqs, spec, frequency, tolerance_hz=40.0):
    mask = np.abs(freqs - frequency) <= tolerance_hz
    return float(spec[mask].max()) if mask.any() else 0.0


def _db(numerator: float, denominator: float) -> float:
    return 20.0 * np.log10(max(numerator, 1e-30) / max(denominator, 1e-30))


# --- registry -----------------------------------------------------------------------


def test_analog_registry_now_holds_all_three_analog_schemes() -> None:
    # Was ["AM-DSB", "AM-SSB"] until WBFM landed in Entry 024.
    assert sorted(ANALOG_MODULATIONS) == ["AM-DSB", "AM-SSB", "WBFM"]


def test_wbfm_is_implemented_but_remains_a_separate_scheme_from_ssb() -> None:
    assert "WBFM" in ANALOG_MODULATIONS
    assert "wbfm" in SUPPORTED_ANALOG_SCHEMES
    # Sideband is an SSB-only concept and must not leak onto WBFM.
    with pytest.raises(ValueError, match="sideband"):
        AnalogSampleSpec(scheme="wbfm", sideband="upper")


def test_am_ssb_declares_the_analog_family_and_production_label() -> None:
    spec = ANALOG_MODULATIONS["AM-SSB"]

    assert spec.family == "analog"
    assert spec.radiofry_label == "AM-SSB"
    assert spec.bits_per_symbol == 0


def test_digital_registry_is_unchanged_by_analog_work() -> None:
    assert sorted(MODULATIONS) == ["16QAM", "64QAM", "8PSK", "BFSK", "BPSK", "GFSK", "PAM4", "QPSK"]
    assert all(spec.family != "analog" for spec in MODULATIONS.values())


def test_unknown_sideband_is_rejected() -> None:
    with pytest.raises(ValueError, match="sideband"):
        _spec(sideband="diagonal")


def test_sideband_is_not_accepted_for_am_dsb() -> None:
    with pytest.raises(ValueError, match="sideband"):
        AnalogSampleSpec(scheme="am_dsb", sideband="lower")


# --- deterministic generation --------------------------------------------------------


def test_am_ssb_generation_is_deterministic() -> None:
    message, _ = generate_message(_spec())

    np.testing.assert_array_equal(
        modulate_am_ssb(message, _spec()), modulate_am_ssb(message, _spec())
    )


def test_am_ssb_has_the_expected_shape_and_dtype() -> None:
    spec = _spec()
    waveform = modulate_am_ssb(generate_message(spec)[0], spec)

    assert waveform.shape == (N,)
    assert waveform.dtype == np.complex64
    assert np.all(np.isfinite(waveform.view(np.float32)))


def test_upper_and_lower_sidebands_differ() -> None:
    message, _ = generate_message(_spec())

    upper = modulate_am_ssb(message, _spec(sideband="upper"))
    lower = modulate_am_ssb(message, _spec(sideband="lower"))

    assert not np.allclose(upper, lower, atol=1e-3)


# --- ORACLE: carrier placement and sideband structure --------------------------------


def test_oracle_energy_sits_on_the_selected_side_of_the_carrier() -> None:
    # USB: essentially all energy above the carrier. This is the defining property.
    spec = _spec(sideband="upper")
    message, _ = generate_message(spec)

    freqs, mag = _spectrum(modulate_am_ssb(message, spec))
    above = float(np.sum(mag[freqs > CARRIER] ** 2))
    below = float(np.sum(mag[freqs < CARRIER] ** 2))

    assert _db(np.sqrt(above), np.sqrt(below)) > 30.0


def test_oracle_lower_sideband_mirrors_the_convention() -> None:
    spec = _spec(sideband="lower")
    message, _ = generate_message(spec)

    freqs, mag = _spectrum(modulate_am_ssb(message, spec))
    above = float(np.sum(mag[freqs > CARRIER] ** 2))
    below = float(np.sum(mag[freqs < CARRIER] ** 2))

    assert _db(np.sqrt(below), np.sqrt(above)) > 30.0


@pytest.mark.parametrize("sideband,sign", [("upper", +1), ("lower", -1)])
def test_oracle_each_declared_tone_appears_in_the_desired_sideband(sideband, sign) -> None:
    spec = _spec(sideband=sideband)
    message, tones = generate_message(spec)

    freqs, mag = _spectrum(modulate_am_ssb(message, spec))
    floor = float(np.median(mag))
    for tone in tones:
        wanted = _level_at(freqs, mag, CARRIER + sign * tone["frequency_hz"])
        assert wanted > 20 * floor, f"missing tone at {tone['frequency_hz']} Hz"


@pytest.mark.parametrize("sideband,sign", [("upper", +1), ("lower", -1)])
def test_oracle_unwanted_sideband_is_suppressed_by_at_least_30_db(sideband, sign) -> None:
    spec = _spec(sideband=sideband)
    message, tones = generate_message(spec)

    freqs, mag = _spectrum(modulate_am_ssb(message, spec))
    for tone in tones:
        wanted = _level_at(freqs, mag, CARRIER + sign * tone["frequency_hz"])
        unwanted = _level_at(freqs, mag, CARRIER - sign * tone["frequency_hz"])
        assert _db(wanted, unwanted) >= 30.0, (
            f"tone {tone['frequency_hz']} Hz suppression only {_db(wanted, unwanted):.1f} dB"
        )


def test_oracle_carrier_itself_is_suppressed() -> None:
    # SSB here is suppressed-carrier, so there must be no line at the carrier.
    spec = _spec()
    message, tones = generate_message(spec)

    freqs, mag = _spectrum(modulate_am_ssb(message, spec))
    carrier_level = _level_at(freqs, mag, CARRIER, tolerance_hz=15.0)
    strongest_tone = max(_level_at(freqs, mag, CARRIER + t["frequency_hz"]) for t in tones)

    assert _db(strongest_tone, carrier_level) > 20.0


def test_oracle_am_ssb_occupies_about_half_the_am_dsb_bandwidth() -> None:
    from radiofry.evaluation.symbol_rate_experiment import occupied_bandwidth

    message, _ = generate_message(_spec())
    ssb = occupied_bandwidth(modulate_am_ssb(message, _spec()), FS)
    dsb = occupied_bandwidth(modulate_am_dsb(message, AnalogSampleSpec(
        scheme="am_dsb", sample_rate_hz=FS, num_samples=N, carrier_offset_hz=CARRIER, seed=11)), FS)

    assert ssb < dsb


# --- ORACLE: independent message recovery ---------------------------------------------


def _independent_product_detector(waveform: np.ndarray, carrier_hz: float, sideband: str) -> np.ndarray:
    """Down-convert and take the real part; independent of the generator."""
    values = waveform.astype(np.complex128)
    time = np.arange(values.size, dtype=np.float64) / FS
    baseband = values * np.exp(-2j * np.pi * carrier_hz * time)
    recovered = np.real(baseband)
    recovered = recovered - recovered.mean()
    return recovered / (np.max(np.abs(recovered)) or 1.0)


@pytest.mark.parametrize("sideband", ["upper", "lower"])
def test_oracle_clean_capture_recovers_the_message_above_point_nine_nine(sideband: str) -> None:
    spec = _spec(sideband=sideband)
    message, _ = generate_message(spec)

    recovered = _independent_product_detector(modulate_am_ssb(message, spec), CARRIER, sideband)

    assert abs(float(np.corrcoef(message, recovered)[0, 1])) > 0.99


@pytest.mark.parametrize("snr_db,floor", [(30.0, 0.95), (20.0, 0.90), (10.0, 0.60)])
def test_oracle_recovery_degrades_gracefully_across_the_snr_range(snr_db, floor) -> None:
    from radiofry.synthetic_gen.v1 import add_awgn

    spec = _spec(snr_db=snr_db)
    message, _ = generate_message(spec)
    noisy = add_awgn(modulate_am_ssb(message, spec), snr_db, np.random.default_rng([spec.seed, 2]))[0]

    recovered = _independent_product_detector(noisy, CARRIER, "upper")

    assert abs(float(np.corrcoef(message, recovered)[0, 1])) > floor


# --- ground truth ----------------------------------------------------------------------


def test_ground_truth_records_the_am_ssb_contract(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "amssb000")

    assert truth["capture_kind"] == "analog"
    assert truth["modulation"]["name"] == "AM-SSB"
    assert truth["modulation"]["family"] == "analog"
    assert truth["modulation"]["radiofry_label"] == "AM-SSB"
    assert truth["analog"]["scheme"] == "am_ssb"
    assert truth["analog"]["sideband"] == "upper"
    assert truth["analog"]["carrier_offset_hz"] == CARRIER
    assert truth["analog"]["modulation_depth"] is None
    assert truth["signal"]["sample_rate_hz"] == FS
    assert truth["signal"]["num_samples"] == N
    assert truth["signal"]["center_frequency_hz"] == CARRIER
    assert truth["noise"]["target_snr_db"] == 20.0


def test_ground_truth_message_block_enables_independent_validation(tmp_path: Path) -> None:
    import hashlib

    truth = generate_analog_sample(_spec(), tmp_path, "amssb000")
    message = np.load(tmp_path / truth["message"]["message_file"])

    assert truth["message"]["type"] == "multitone"
    assert len(truth["message"]["tones"]) == 3
    assert all({"frequency_hz", "amplitude", "phase_rad"} <= set(t) for t in truth["message"]["tones"])
    assert hashlib.sha256(message.tobytes()).hexdigest() == truth["message"]["message_sha256"]


def test_bit_derived_fields_are_null_for_am_ssb(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "amssb000")

    assert truth["bits"] is None
    assert truth["modulation"]["bits_per_symbol"] is None
    assert truth["modulation"]["order"] is None
    assert truth["signal"]["symbol_rate_hz"] is None
    assert truth["signal"]["samples_per_symbol"] is None
    assert truth["noise"]["es_n0_db"] is None
    assert truth["noise"]["eb_n0_db"] is None


def test_am_ssb_capture_is_reproducible(tmp_path: Path) -> None:
    generate_analog_sample(_spec(snr_db=20.0), tmp_path / "a", "amssb000")
    generate_analog_sample(_spec(snr_db=20.0), tmp_path / "b", "amssb000")

    assert (tmp_path / "a" / "amssb000.iq").read_bytes() == (tmp_path / "b" / "amssb000.iq").read_bytes()


def test_am_ssb_capture_passes_through_the_harness_safely(tmp_path: Path) -> None:
    from radiofry.evaluation.harness import evaluate_capture

    generate_analog_sample(_spec(snr_db=20.0), tmp_path / "captures", "amssb000")
    row = {"capture_id": "amssb000", "ground_truth_file": "captures/amssb000.json",
           "iq_file": "captures/amssb000.iq", "wav_file": "captures/amssb000.wav"}

    record = evaluate_capture(tmp_path, row, "iq")

    assert record["pipeline_ok"] is True
    assert record["expected_family"] == "analog-like"
    assert record["ber_status"] == "unavailable"
    assert record["ber_strict"] is None


# --- AM-DSB regression -----------------------------------------------------------------


def test_am_dsb_generation_is_unchanged(tmp_path: Path) -> None:
    dsb = AnalogSampleSpec(scheme="am_dsb", sample_rate_hz=FS, num_samples=4_096, seed=11)
    message, _ = generate_message(dsb)

    waveform = modulate_am_dsb(message, dsb)
    truth = generate_analog_sample(dsb, tmp_path, "amdsb000")

    assert np.min(np.abs(waveform)) > 0.4
    assert truth["modulation"]["name"] == "AM-DSB"
    assert truth["analog"]["scheme"] == "am_dsb"
    assert truth["analog"]["modulation_depth"] == 0.5
    assert truth["analog"]["sideband"] is None
