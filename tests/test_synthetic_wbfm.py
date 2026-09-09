"""WBFM analog synthetic generation and its independent oracle (BANK.md Entry 024).

The oracle recovers instantaneous frequency from the emitted waveform's phase
progression and FM-demodulates it with its own discriminator. It never calls
`modulate_wbfm`, and it rebuilds the expected message from the recorded tone list rather
than reusing the generator's message array.

This validates signal *generation* only. It says nothing about production WBFM
demodulation, classification or routing.
"""

import hashlib
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
    modulate_wbfm,
)

FS = 200_000.0
N = 32_768
CARRIER = 20_000.0
DEVIATION = 15_000.0
V1_CAPTURES = Path("data/synthetic_v1/captures")


def _spec(**overrides) -> AnalogSampleSpec:
    params = dict(scheme="wbfm", sample_rate_hz=FS, num_samples=N,
                  carrier_offset_hz=CARRIER, frequency_deviation_hz=DEVIATION,
                  snr_db=None, seed=23)
    params.update(overrides)
    return AnalogSampleSpec(**params)


# --- the independent oracle ----------------------------------------------------------


def _instantaneous_frequency(waveform: np.ndarray) -> np.ndarray:
    """f_i[n] = (Fs / 2*pi) * arg(x[n+1] * conj(x[n])).

    Derived from the emitted samples alone - no generator state, no metadata.
    """
    values = waveform.astype(np.complex128)
    return FS * np.angle(values[1:] * np.conj(values[:-1])) / (2.0 * np.pi)


def _independent_fm_discriminator(waveform: np.ndarray, carrier_hz: float,
                                  deviation_hz: float) -> np.ndarray:
    """Recover the normalised message: s_hat = (f_i - f_c) / delta_f."""
    return (_instantaneous_frequency(waveform) - carrier_hz) / deviation_hz


def _expected_message_from_tones(tones, num_samples: int) -> np.ndarray:
    """Rebuild the clean message from the recorded tone list, independently."""
    time = np.arange(num_samples, dtype=np.float64) / FS
    message = np.zeros(num_samples, dtype=np.float64)
    for tone in tones:
        message += tone["amplitude"] * np.sin(
            2 * np.pi * tone["frequency_hz"] * time + tone["phase_rad"])
    message -= message.mean()
    peak = float(np.max(np.abs(message)))
    return message / peak if peak > 0 else message


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    x, y = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def _tone_peaks(signal: np.ndarray, count: int, band_hz=(200.0, 4_000.0)):
    """Strongest `count` spectral lines of a real signal inside the message band."""
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, d=1 / FS)
    inside = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    idx = np.where(inside)[0]
    order = idx[np.argsort(spectrum[idx])[::-1]]
    picked: list[float] = []
    for i in order:
        if all(abs(freqs[i] - f) > 50.0 for f in picked):
            picked.append(float(freqs[i]))
        if len(picked) == count:
            break
    return sorted(picked)


def _occupied_bandwidth(waveform: np.ndarray, fraction: float = 0.99) -> float:
    """Width of the smallest contiguous band holding `fraction` of total power."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(waveform.astype(np.complex128)))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(waveform.size, d=1 / FS))
    total = spectrum.sum()
    order = np.argsort(spectrum)[::-1]
    kept = np.zeros(spectrum.size, dtype=bool)
    running = 0.0
    for i in order:
        kept[i] = True
        running += spectrum[i]
        if running >= fraction * total:
            break
    selected = freqs[kept]
    return float(selected.max() - selected.min())


# --- registry -------------------------------------------------------------------------


def test_wbfm_is_now_in_the_analog_registry() -> None:
    assert "WBFM" in ANALOG_MODULATIONS
    assert ANALOG_MODULATIONS["WBFM"].family == "analog"
    assert ANALOG_MODULATIONS["WBFM"].radiofry_label == "WBFM"


def test_analog_registry_holds_exactly_the_three_implemented_schemes() -> None:
    assert sorted(ANALOG_MODULATIONS) == ["AM-DSB", "AM-SSB", "WBFM"]
    assert SUPPORTED_ANALOG_SCHEMES == ("am_dsb", "am_ssb", "wbfm")


def test_wbfm_carries_no_digital_order_or_bit_width() -> None:
    spec = ANALOG_MODULATIONS["WBFM"]
    assert spec.order == 0
    assert spec.bits_per_symbol == 0


def test_digital_registry_is_untouched_by_wbfm() -> None:
    assert sorted(MODULATIONS) == ["16QAM", "64QAM", "8PSK", "BFSK", "BPSK",
                                   "GFSK", "PAM4", "QPSK"]
    assert "WBFM" not in MODULATIONS


# --- spec validation --------------------------------------------------------------------


def test_frequency_deviation_defaults_when_not_given() -> None:
    spec = AnalogSampleSpec(scheme="wbfm", sample_rate_hz=FS, num_samples=1_024)

    assert spec.frequency_deviation_hz is not None
    assert spec.frequency_deviation_hz > 0


def test_frequency_deviation_is_rejected_for_the_am_schemes() -> None:
    with pytest.raises(ValueError, match="frequency_deviation_hz"):
        AnalogSampleSpec(scheme="am_dsb", frequency_deviation_hz=5_000.0)


def test_a_non_positive_deviation_is_rejected() -> None:
    with pytest.raises(ValueError, match="frequency_deviation_hz"):
        _spec(frequency_deviation_hz=0.0)


def test_a_deviation_that_would_alias_is_rejected() -> None:
    # f_c + delta_f must stay inside the sampled band.
    with pytest.raises(ValueError, match="Nyquist|band"):
        _spec(carrier_offset_hz=60_000.0, frequency_deviation_hz=60_000.0)


def test_sideband_is_still_rejected_for_wbfm() -> None:
    with pytest.raises(ValueError, match="sideband"):
        _spec(sideband="upper")


# --- waveform physics -------------------------------------------------------------------


def test_wbfm_is_constant_envelope() -> None:
    spec = _spec()
    waveform = modulate_wbfm(generate_message(spec)[0], spec)

    magnitude = np.abs(waveform.astype(np.complex128))

    assert np.allclose(magnitude, 1.0, atol=1e-5)


def test_generation_is_deterministic_for_a_fixed_seed() -> None:
    spec = _spec()

    first = modulate_wbfm(generate_message(spec)[0], spec)
    second = modulate_wbfm(generate_message(spec)[0], spec)

    assert np.array_equal(first, second)


def test_measured_peak_deviation_matches_the_configured_deviation() -> None:
    # The message is peak-normalised to 1.0, so peak |f_i - f_c| must equal delta_f.
    spec = _spec()
    waveform = modulate_wbfm(generate_message(spec)[0], spec)

    excursion = np.abs(_instantaneous_frequency(waveform) - CARRIER)

    assert excursion.max() == pytest.approx(DEVIATION, rel=0.01)


def test_measured_deviation_tracks_a_changed_configuration() -> None:
    spec = _spec(frequency_deviation_hz=5_000.0)
    waveform = modulate_wbfm(generate_message(spec)[0], spec)

    excursion = np.abs(_instantaneous_frequency(waveform) - CARRIER)

    assert excursion.max() == pytest.approx(5_000.0, rel=0.01)


def test_mean_instantaneous_frequency_sits_on_the_carrier() -> None:
    # A zero-mean message must leave the average instantaneous frequency at f_c.
    spec = _spec()
    waveform = modulate_wbfm(generate_message(spec)[0], spec)

    mean_frequency = float(_instantaneous_frequency(waveform).mean())

    assert mean_frequency == pytest.approx(CARRIER, abs=DEVIATION * 0.01)


def test_instantaneous_frequency_is_not_merely_a_constant_tone() -> None:
    spec = _spec()
    waveform = modulate_wbfm(generate_message(spec)[0], spec)

    frequency = _instantaneous_frequency(waveform)

    assert frequency.std() > 0.1 * DEVIATION


# --- independent message recovery ---------------------------------------------------------


def test_independent_discriminator_recovers_the_expected_message(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(), tmp_path, "wbfm000")
    waveform = modulate_wbfm(generate_message(_spec())[0], _spec())

    recovered = _independent_fm_discriminator(waveform, CARRIER, DEVIATION)
    expected = _expected_message_from_tones(truth["message"]["tones"], N)

    assert _correlation(recovered, expected[1:]) > 0.99


def test_recovered_message_contains_the_expected_tone_frequencies(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(), tmp_path, "wbfm000")
    waveform = modulate_wbfm(generate_message(_spec())[0], _spec())

    recovered = _independent_fm_discriminator(waveform, CARRIER, DEVIATION)
    found = _tone_peaks(recovered, count=len(truth["message"]["tones"]))
    expected = sorted(t["frequency_hz"] for t in truth["message"]["tones"])

    for want, got in zip(expected, found):
        assert abs(want - got) < 25.0, f"expected tone {want:.1f} Hz, found {got:.1f} Hz"


# --- noise behaviour ------------------------------------------------------------------------


def test_recovery_degrades_monotonically_as_noise_increases(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(), tmp_path, "clean")
    expected = _expected_message_from_tones(truth["message"]["tones"], N)[1:]

    correlations = []
    for snr_db in (30.0, 20.0, 10.0, 0.0):
        from radiofry.synthetic_gen.v1.channel import add_awgn
        spec = _spec(snr_db=snr_db)
        clean = modulate_wbfm(generate_message(spec)[0], spec)
        noisy, _ = add_awgn(clean, snr_db, np.random.default_rng([spec.seed, 2]))
        recovered = _independent_fm_discriminator(noisy, CARRIER, DEVIATION)
        correlations.append(_correlation(recovered, expected))

    assert correlations == sorted(correlations, reverse=True), correlations
    assert correlations[0] > 0.9, "high-SNR recovery should still be strong"


def test_high_snr_recovery_stays_usable(tmp_path: Path) -> None:
    from radiofry.synthetic_gen.v1.channel import add_awgn
    truth = generate_analog_sample(_spec(), tmp_path, "clean")
    expected = _expected_message_from_tones(truth["message"]["tones"], N)[1:]
    spec = _spec(snr_db=30.0)
    clean = modulate_wbfm(generate_message(spec)[0], spec)
    noisy, _ = add_awgn(clean, 30.0, np.random.default_rng([spec.seed, 2]))

    correlation = _correlation(_independent_fm_discriminator(noisy, CARRIER, DEVIATION),
                               expected)

    assert correlation > 0.95


# --- occupied spectrum -----------------------------------------------------------------------


def test_measured_occupied_bandwidth_is_the_right_order_as_carsons_rule() -> None:
    # Carson's rule is an APPROXIMATION: B ~ 2*(delta_f + f_max). The measured 99%
    # occupied bandwidth of this finite multi-tone signal is only required to land in
    # the same ballpark, not to equal it.
    spec = _spec()
    waveform = modulate_wbfm(generate_message(spec)[0], spec)
    tones = generate_message(spec)[1]
    carson = 2.0 * (DEVIATION + max(t["frequency_hz"] for t in tones))

    measured = _occupied_bandwidth(waveform)

    assert 0.5 * carson < measured < 2.0 * carson, f"{measured=} {carson=}"


def test_wbfm_occupies_more_bandwidth_than_am_dsb_of_the_same_message() -> None:
    wbfm_spec = _spec()
    am_spec = AnalogSampleSpec(scheme="am_dsb", sample_rate_hz=FS, num_samples=N,
                               carrier_offset_hz=CARRIER, seed=23)

    wbfm_bw = _occupied_bandwidth(modulate_wbfm(generate_message(wbfm_spec)[0], wbfm_spec))
    am_bw = _occupied_bandwidth(modulate_am_dsb(generate_message(am_spec)[0], am_spec))

    assert wbfm_bw > am_bw


# --- ground truth --------------------------------------------------------------------------------


def test_ground_truth_describes_wbfm(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "wbfm000")

    assert truth["capture_kind"] == "analog"
    assert truth["modulation"]["name"] == "WBFM"
    assert truth["modulation"]["family"] == "analog"
    assert truth["modulation"]["radiofry_label"] == "WBFM"
    assert truth["signal"]["sample_rate_hz"] == FS
    assert truth["signal"]["num_samples"] == N
    assert truth["signal"]["center_frequency_hz"] == CARRIER
    assert truth["signal"]["duration_sec"] == pytest.approx(N / FS)


def test_ground_truth_records_the_fm_parameters(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(), tmp_path, "wbfm000")
    analog = truth["analog"]
    tones = truth["message"]["tones"]
    highest = max(t["frequency_hz"] for t in tones)

    assert analog["scheme"] == "wbfm"
    assert analog["frequency_deviation_hz"] == DEVIATION
    assert analog["modulation_index"] == pytest.approx(DEVIATION / highest, rel=1e-9)
    assert analog["carson_bandwidth_hz"] == pytest.approx(2 * (DEVIATION + highest))
    assert analog["carrier"] == "present"


def test_am_only_parameters_are_null_for_wbfm(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(), tmp_path, "wbfm000")

    assert truth["analog"]["modulation_depth"] is None
    assert truth["analog"]["sideband"] is None


def test_digital_fields_are_null_for_wbfm(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "wbfm000")

    assert truth["bits"] is None
    assert truth["modulation"]["order"] is None
    assert truth["modulation"]["bits_per_symbol"] is None
    assert truth["modulation"]["bit_mapping"] is None
    assert truth["signal"]["symbol_rate_hz"] is None
    assert truth["signal"]["samples_per_symbol"] is None
    # fsk_* are digital-FSK descriptors; WBFM's deviation lives in the analog block.
    assert truth["signal"]["fsk_deviation_hz"] is None
    assert truth["signal"]["fsk_modulation_index"] is None
    assert truth["noise"]["es_n0_db"] is None
    assert truth["noise"]["eb_n0_db"] is None
    assert truth["impairments"]["fec"] == "none"
    assert truth["impairments"]["interleaving"] == "none"


def test_ground_truth_records_the_message_and_noise(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(snr_db=20.0), tmp_path, "wbfm000")

    assert truth["message"]["type"] == "multitone"
    assert len(truth["message"]["tones"]) == truth["message"]["num_tones"]
    assert len(truth["message"]["message_sha256"]) == 64
    assert truth["noise"]["noise_type"] == "awgn"
    assert truth["noise"]["target_snr_db"] == 20.0
    assert truth["noise"]["realized_snr_db"] == pytest.approx(20.0, abs=1.0)


def test_message_sha256_matches_the_saved_message(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(), tmp_path, "wbfm000")

    saved = np.load(tmp_path / truth["message"]["message_file"])

    assert hashlib.sha256(saved.tobytes()).hexdigest() == truth["message"]["message_sha256"]


# --- output formats ---------------------------------------------------------------------------------


def test_generated_capture_is_byte_reproducible(tmp_path: Path) -> None:
    generate_analog_sample(_spec(snr_db=20.0), tmp_path / "a", "wbfm000")
    generate_analog_sample(_spec(snr_db=20.0), tmp_path / "b", "wbfm000")

    assert (tmp_path / "a" / "wbfm000.iq").read_bytes() == (tmp_path / "b" / "wbfm000.iq").read_bytes()


def test_iq_survives_the_existing_write_read_path(tmp_path: Path) -> None:
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import load_capture

    truth = generate_analog_sample(_spec(), tmp_path, "wbfm000")
    entry = next(e for e in truth["files"] if e["file_format"] == "iq")

    signal = load_capture(tmp_path / "wbfm000.iq", sample_rate=FS,
                          iq_format=IQFormat(entry["dtype"], entry["byte_order"]))

    assert signal.iq.size == N
    # The int16 round trip must preserve the FM physics, not just the sample count.
    excursion = np.abs(_instantaneous_frequency(signal.iq) - CARRIER)
    assert excursion.max() == pytest.approx(DEVIATION, rel=0.05)


def test_wav_output_is_produced_for_wbfm(tmp_path: Path) -> None:
    truth = generate_analog_sample(_spec(), tmp_path, "wbfm000")

    entry = next(e for e in truth["files"] if e["file_format"] == "wav")

    assert (tmp_path / "wbfm000.wav").exists()
    assert entry["channel_layout"] == "stereo_iq"
    assert entry["num_samples"] == N


# --- regression: nothing else moved --------------------------------------------------------------------


def test_am_dsb_waveform_is_unchanged() -> None:
    spec = AnalogSampleSpec(scheme="am_dsb", sample_rate_hz=FS, num_samples=4_096,
                            carrier_offset_hz=0.0, modulation_depth=0.5, seed=11)
    message = generate_message(spec)[0]

    waveform = modulate_am_dsb(message, spec)

    assert np.allclose(waveform.real, 1.0 + 0.5 * message, atol=1e-5)
    assert np.allclose(waveform.imag, 0.0, atol=1e-6)


def test_am_ssb_waveform_is_unchanged() -> None:
    from scipy.signal import hilbert
    spec = AnalogSampleSpec(scheme="am_ssb", sample_rate_hz=FS, num_samples=4_096,
                            carrier_offset_hz=CARRIER, sideband="upper", seed=11)
    message = generate_message(spec)[0]

    waveform = modulate_am_ssb(message, spec)

    time = np.arange(message.size) / FS
    expected = hilbert(message) * np.exp(2j * np.pi * CARRIER * time)
    assert np.allclose(waveform, expected.astype(np.complex64), atol=1e-5)


def test_am_dsb_and_am_ssb_ground_truth_still_null_the_fm_fields(tmp_path: Path) -> None:
    dsb = generate_analog_sample(
        AnalogSampleSpec(scheme="am_dsb", num_samples=1_024, seed=11), tmp_path, "d")
    ssb = generate_analog_sample(
        AnalogSampleSpec(scheme="am_ssb", num_samples=1_024, seed=11), tmp_path, "s")

    assert dsb["analog"]["frequency_deviation_hz"] is None
    assert dsb["analog"]["modulation_index"] is None
    assert ssb["analog"]["frequency_deviation_hz"] is None
    assert ssb["analog"]["carson_bandwidth_hz"] is None


@pytest.mark.skipif(not V1_CAPTURES.exists(), reason="frozen V1 dataset not present")
def test_frozen_v1_dataset_is_byte_identical() -> None:
    digest = hashlib.sha256()
    files = sorted(V1_CAPTURES.glob("*.iq"))

    for path in files:
        digest.update(path.read_bytes())

    assert len(files) == 40
    assert digest.hexdigest()[:32] == "d6d3f918687d0700a43e46211c3f04b9"
