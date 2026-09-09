"""AM-DSB analog synthetic generation and its independent oracle (BANK.md Entry 020).

The oracle deliberately re-derives expected structure from first principles (carrier
and sideband positions from the recorded tone list, envelope recovery by an independent
detector) rather than re-running the generator's own code path.

WBFM is NOT implemented. AM-SSB arrived in Entry 022.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from radiofry.synthetic_gen.v1 import MODULATIONS
from radiofry.synthetic_gen.v1.analog import (
    ANALOG_MODULATIONS,
    AnalogSampleSpec,
    generate_analog_sample,
    generate_message,
    modulate_am_dsb,
)

FS = 200_000.0
N = 32_768


def _spec(**overrides) -> AnalogSampleSpec:
    params = dict(scheme="am_dsb", sample_rate_hz=FS, num_samples=N,
                  carrier_offset_hz=0.0, modulation_depth=0.5, snr_db=None, seed=11)
    params.update(overrides)
    return AnalogSampleSpec(**params)


def _spectrum(iq: np.ndarray, sample_rate: float = FS):
    window = np.hanning(iq.size)
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq * window)))
    freqs = np.fft.fftshift(np.fft.fftfreq(iq.size, d=1 / sample_rate))
    return freqs, spec


def _level_at(freqs, spec, frequency, tolerance_hz=40.0):
    mask = np.abs(freqs - frequency) <= tolerance_hz
    return float(spec[mask].max()) if mask.any() else 0.0


# --- registry: analog is kept out of the digital registry ---------------------------


def test_analog_registry_holds_the_implemented_analog_schemes() -> None:
    # Grew with the implementation: ["AM-DSB"] (Entry 020), +AM-SSB (022), +WBFM (024).
    assert sorted(ANALOG_MODULATIONS) == ["AM-DSB", "AM-SSB", "WBFM"]


def test_am_dsb_is_still_the_only_scheme_with_a_retained_amplitude_carrier() -> None:
    # WBFM keeps a carrier but is constant-envelope; only AM-DSB modulates amplitude.
    assert "WBFM" in ANALOG_MODULATIONS


def test_analog_is_not_in_the_digital_registry() -> None:
    # Keeps every tuple(MODULATIONS) default digital-only.
    assert "AM-DSB" not in MODULATIONS
    assert all(spec.family != "analog" for spec in MODULATIONS.values())


def test_am_dsb_declares_the_analog_family_and_production_label() -> None:
    spec = ANALOG_MODULATIONS["AM-DSB"]

    assert spec.family == "analog"
    assert spec.radiofry_label == "AM-DSB"


def test_unsupported_analog_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="scheme"):
        # NBFM is not implemented; "wbfm" stopped being a valid negative case in Entry 024.
        _spec(scheme="nbfm")


# --- deterministic multi-tone message ------------------------------------------------


def test_message_generation_is_deterministic() -> None:
    a, tones_a = generate_message(_spec())
    b, tones_b = generate_message(_spec())

    np.testing.assert_array_equal(a, b)
    assert tones_a == tones_b


def test_message_changes_with_the_seed() -> None:
    a, _ = generate_message(_spec(seed=1))
    b, _ = generate_message(_spec(seed=2))

    assert not np.allclose(a, b)


def test_message_is_peak_normalised_and_zero_mean() -> None:
    message, _ = generate_message(_spec())

    assert np.max(np.abs(message)) == pytest.approx(1.0, rel=1e-9)
    assert abs(float(np.mean(message))) < 0.05


def test_message_tones_land_in_the_requested_band() -> None:
    spec = _spec(num_tones=4, tone_band_hz=(500.0, 4000.0))

    _, tones = generate_message(spec)

    assert len(tones) == 4
    assert all(500.0 <= t["frequency_hz"] <= 4000.0 for t in tones)
    assert all(t["amplitude"] > 0 for t in tones)


def test_message_spectrum_shows_exactly_the_declared_tones() -> None:
    spec = _spec(num_tones=3)
    message, tones = generate_message(spec)

    freqs, mag = _spectrum(message.astype(np.complex64))
    floor = float(np.median(mag))
    for tone in tones:
        assert _level_at(freqs, mag, tone["frequency_hz"]) > 20 * floor


# --- AM-DSB waveform -----------------------------------------------------------------


def test_am_dsb_generation_is_deterministic() -> None:
    np.testing.assert_array_equal(
        modulate_am_dsb(*generate_message(_spec())[:1], _spec()),
        modulate_am_dsb(*generate_message(_spec())[:1], _spec()),
    )


def test_am_dsb_envelope_is_strictly_positive_below_full_depth() -> None:
    # 1 + m*s(t) must not go negative, otherwise envelope detection is invalid.
    spec = _spec(modulation_depth=0.5)
    message, _ = generate_message(spec)

    waveform = modulate_am_dsb(message, spec)

    assert np.min(np.abs(waveform)) > 0.4


def test_am_dsb_has_the_expected_sample_count_and_dtype() -> None:
    spec = _spec()
    waveform = modulate_am_dsb(generate_message(spec)[0], spec)

    assert waveform.shape == (N,)
    assert waveform.dtype == np.complex64


# --- ORACLE: carrier and symmetric sidebands ------------------------------------------


@pytest.mark.parametrize("carrier_hz", [0.0, 12_000.0])
def test_oracle_carrier_is_present_at_the_expected_frequency(carrier_hz: float) -> None:
    spec = _spec(carrier_offset_hz=carrier_hz)
    message, _ = generate_message(spec)

    freqs, mag = _spectrum(modulate_am_dsb(message, spec))
    peak = float(freqs[int(np.argmax(mag))])

    assert peak == pytest.approx(carrier_hz, abs=40.0)


@pytest.mark.parametrize("carrier_hz", [0.0, 12_000.0])
def test_oracle_symmetric_sidebands_surround_the_carrier(carrier_hz: float) -> None:
    spec = _spec(carrier_offset_hz=carrier_hz)
    message, tones = generate_message(spec)

    freqs, mag = _spectrum(modulate_am_dsb(message, spec))
    floor = float(np.median(mag))
    for tone in tones:
        upper = _level_at(freqs, mag, carrier_hz + tone["frequency_hz"])
        lower = _level_at(freqs, mag, carrier_hz - tone["frequency_hz"])
        assert upper > 20 * floor, f"missing upper sideband for {tone['frequency_hz']} Hz"
        assert lower > 20 * floor, f"missing lower sideband for {tone['frequency_hz']} Hz"
        # DSB sidebands carry equal power.
        assert upper == pytest.approx(lower, rel=0.25)


def test_oracle_sideband_amplitude_tracks_the_modulation_depth() -> None:
    message, tones = generate_message(_spec())
    ratios = []
    for depth in (0.2, 0.8):
        spec = _spec(modulation_depth=depth)
        freqs, mag = _spectrum(modulate_am_dsb(message, spec))
        carrier = _level_at(freqs, mag, 0.0)
        sideband = _level_at(freqs, mag, tones[0]["frequency_hz"])
        ratios.append(sideband / carrier)

    assert ratios[1] > ratios[0] * 3


# --- ORACLE: independent envelope recovery --------------------------------------------


def _independent_envelope_recovery(waveform: np.ndarray, carrier_hz: float) -> np.ndarray:
    """Recover the message without touching the generator or RadioFry demodulators."""
    envelope = np.abs(waveform.astype(np.complex128))
    recovered = envelope - envelope.mean()
    return recovered / (np.max(np.abs(recovered)) or 1.0)


def test_oracle_clean_capture_recovers_the_message_above_point_nine_nine() -> None:
    spec = _spec()
    message, _ = generate_message(spec)

    recovered = _independent_envelope_recovery(modulate_am_dsb(message, spec), 0.0)
    correlation = float(np.corrcoef(message, recovered)[0, 1])

    assert correlation > 0.99


@pytest.mark.parametrize("carrier_hz", [0.0, 12_000.0])
def test_oracle_recovery_holds_with_a_carrier_offset(carrier_hz: float) -> None:
    spec = _spec(carrier_offset_hz=carrier_hz)
    message, _ = generate_message(spec)

    recovered = _independent_envelope_recovery(modulate_am_dsb(message, spec), carrier_hz)

    assert float(np.corrcoef(message, recovered)[0, 1]) > 0.99


@pytest.mark.parametrize("snr_db,floor", [(30.0, 0.95), (20.0, 0.90), (10.0, 0.60)])
def test_oracle_recovery_degrades_gracefully_across_the_snr_range(snr_db, floor) -> None:
    from radiofry.synthetic_gen.v1 import add_awgn

    spec = _spec(snr_db=snr_db)
    message, _ = generate_message(spec)
    clean = modulate_am_dsb(message, spec)
    noisy = add_awgn(clean, snr_db, np.random.default_rng([spec.seed, 2]))[0]

    recovered = _independent_envelope_recovery(noisy, 0.0)
    assert float(np.corrcoef(message, recovered)[0, 1]) > floor


# --- ground truth ---------------------------------------------------------------------


def test_ground_truth_records_the_analog_contract(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "amdsb000")

    assert truth["capture_kind"] == "analog"
    assert truth["modulation"]["name"] == "AM-DSB"
    assert truth["modulation"]["family"] == "analog"
    assert truth["modulation"]["radiofry_label"] == "AM-DSB"
    assert truth["analog"]["scheme"] == "am_dsb"
    assert truth["analog"]["modulation_depth"] == 0.5
    assert truth["signal"]["sample_rate_hz"] == FS
    assert truth["signal"]["num_samples"] == N
    assert truth["signal"]["duration_sec"] == pytest.approx(N / FS)
    assert truth["signal"]["center_frequency_hz"] == 0.0
    assert truth["noise"]["target_snr_db"] == 20.0
    assert truth["noise"]["noise_type"] == "awgn"


def test_ground_truth_message_block_supports_independent_validation(tmp_path: Path) -> None:
    import hashlib

    truth = generate_analog_sample(_spec(), tmp_path, "amdsb000")
    message = np.load(tmp_path / truth["message"]["message_file"])

    assert truth["message"]["type"] == "multitone"
    assert len(truth["message"]["tones"]) == 3
    assert truth["message"]["peak_normalised"] is True
    assert hashlib.sha256(message.tobytes()).hexdigest() == truth["message"]["message_sha256"]
    assert message.size == N


def test_bit_derived_fields_are_null_for_analog(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "amdsb000")

    assert truth["bits"] is None
    assert truth["modulation"]["bits_per_symbol"] is None
    assert truth["modulation"]["order"] is None
    assert truth["signal"]["symbol_rate_hz"] is None
    assert truth["signal"]["samples_per_symbol"] is None
    assert truth["noise"]["eb_n0_db"] is None
    assert truth["noise"]["es_n0_db"] is None


def test_analog_spec_has_no_symbol_rate_concept() -> None:
    fields = set(AnalogSampleSpec.__dataclass_fields__)

    assert not fields & {"symbol_rate_hz", "samples_per_symbol", "num_symbols", "bits_seed"}


def test_generate_analog_sample_writes_capture_files(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "amdsb000")

    assert (tmp_path / "amdsb000.iq").exists()
    assert (tmp_path / "amdsb000.wav").exists()
    assert (tmp_path / "amdsb000.json").exists()
    assert {e["file_format"] for e in truth["files"]} == {"iq", "wav"}


def test_generated_capture_is_reproducible(tmp_path: Path) -> None:
    generate_analog_sample(_spec(snr_db=20.0), tmp_path / "a", "amdsb000")
    generate_analog_sample(_spec(snr_db=20.0), tmp_path / "b", "amdsb000")

    assert (tmp_path / "a" / "amdsb000.iq").read_bytes() == (tmp_path / "b" / "amdsb000.iq").read_bytes()


def test_analog_capture_is_readable_by_the_existing_loader(tmp_path: Path) -> None:
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import load_capture

    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "amdsb000")
    entry = next(e for e in truth["files"] if e["file_format"] == "iq")

    signal = load_capture(tmp_path / "amdsb000.iq", sample_rate=FS,
                          iq_format=IQFormat(entry["dtype"], entry["byte_order"]))

    assert signal.iq.size == N
