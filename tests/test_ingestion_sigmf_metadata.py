"""SigMF sidecar center-frequency ingestion (BANK.md Entry 030).

Entry 029 established that the suppressed SSB carrier cannot be recovered blind to the
~1-2 Hz the demodulator needs, and that `estimate_parameters` already prefers
`metadata["center_frequency_hz"]` when present. The gap was that no parser ever
populated it. These tests pin the smallest legitimate source: a SigMF `.sigmf-meta`
sidecar, which is real capture metadata a receiver writes.

ANTI-LEAKAGE: every sidecar in this file is written by the test itself, representing what
a real recorder would emit. The synthetic generator does NOT write sidecars, and
production ingestion never reads the generator's ground-truth JSON. Where a test needs
the true carrier it uses it as an INDEPENDENT expected value, never as a production input.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.decoding.demodulators.analog_demod import demodulate_ssb
from radiofry.dsp.parameter_estimation import estimate_parameters
from radiofry.dsp.preprocessing import preprocess
from radiofry.ingestion.iq_parser import IQFormat, read_iq
from radiofry.ingestion.sidecar import SIGMF_FREQUENCY_KEY, read_sigmf_sidecar
from radiofry.ingestion.wav_parser import read_wav
from radiofry.synthetic_gen.v1.analog import (
    AnalogSampleSpec, generate_analog_sample, generate_message)

FS, N, TRUE_CARRIER = 200_000.0, 8_192, 20_000.0


def _write_sidecar(directory: Path, stem: str, *, frequency=TRUE_CARRIER,
                   sample_rate=FS, extra=None) -> Path:
    """Write the kind of .sigmf-meta a real recorder emits. Not generator output."""
    meta = {
        "global": {"core:datatype": "ci16_le", "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0}],
    }
    if sample_rate is not None:
        meta["global"]["core:sample_rate"] = sample_rate
    if frequency is not None:
        meta["captures"][0][SIGMF_FREQUENCY_KEY] = frequency
    if extra:
        meta["captures"][0].update(extra)
    path = directory / f"{stem}.sigmf-meta"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def _ssb_capture(tmp_path: Path, stem="ssb0"):
    """A real-looking capture on disk, plus the true carrier as an independent reference."""
    spec = AnalogSampleSpec(scheme="am_ssb", sample_rate_hz=FS, num_samples=N,
                            carrier_offset_hz=TRUE_CARRIER, sideband="upper", seed=101)
    truth = generate_analog_sample(spec, tmp_path, stem)
    message = generate_message(spec)[0]
    entry = next(e for e in truth["files"] if e["file_format"] == "iq")
    return tmp_path / f"{stem}.iq", message, IQFormat(entry["dtype"], entry["byte_order"])


def _corr(a, b) -> float:
    n = min(a.size, b.size)
    x, y = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / d) if d else 0.0


# --- the sidecar reader ---------------------------------------------------------------


def test_the_sigmf_frequency_key_is_the_standard_one() -> None:
    assert SIGMF_FREQUENCY_KEY == "core:frequency"


def test_a_sidecar_next_to_an_iq_file_is_found(tmp_path: Path) -> None:
    (tmp_path / "cap.iq").write_bytes(b"\x00\x00" * 8)
    _write_sidecar(tmp_path, "cap")

    found = read_sigmf_sidecar(tmp_path / "cap.iq")

    assert found["center_frequency_hz"] == TRUE_CARRIER
    assert found["sample_rate_hz"] == FS


def test_a_missing_sidecar_yields_nothing_rather_than_a_guess(tmp_path: Path) -> None:
    (tmp_path / "cap.iq").write_bytes(b"\x00\x00" * 8)

    assert read_sigmf_sidecar(tmp_path / "cap.iq") == {}


def test_a_sigmf_data_file_finds_its_own_sidecar(tmp_path: Path) -> None:
    (tmp_path / "cap.sigmf-data").write_bytes(b"\x00\x00" * 8)
    _write_sidecar(tmp_path, "cap")

    assert read_sigmf_sidecar(tmp_path / "cap.sigmf-data")["center_frequency_hz"] == TRUE_CARRIER


# --- malformed metadata must be rejected, never silently accepted -------------------------


@pytest.mark.parametrize("bad", ["not-a-number", None, float("nan"), float("inf"), -1.0, True, [20_000.0]])
def test_malformed_center_frequency_is_rejected(tmp_path: Path, bad) -> None:
    (tmp_path / "cap.iq").write_bytes(b"\x00\x00" * 8)
    _write_sidecar(tmp_path, "cap", frequency=bad)

    found = read_sigmf_sidecar(tmp_path / "cap.iq")

    assert "center_frequency_hz" not in found, f"{bad!r} was accepted"


def test_a_zero_center_frequency_is_legitimate_baseband(tmp_path: Path) -> None:
    (tmp_path / "cap.iq").write_bytes(b"\x00\x00" * 8)
    _write_sidecar(tmp_path, "cap", frequency=0.0)

    assert read_sigmf_sidecar(tmp_path / "cap.iq")["center_frequency_hz"] == 0.0


def test_unparseable_json_is_ignored_not_raised(tmp_path: Path) -> None:
    (tmp_path / "cap.iq").write_bytes(b"\x00\x00" * 8)
    (tmp_path / "cap.sigmf-meta").write_text("{ this is not json", encoding="utf-8")

    assert read_sigmf_sidecar(tmp_path / "cap.iq") == {}


def test_a_sidecar_without_captures_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "cap.iq").write_bytes(b"\x00\x00" * 8)
    (tmp_path / "cap.sigmf-meta").write_text(json.dumps({"global": {}}), encoding="utf-8")

    assert read_sigmf_sidecar(tmp_path / "cap.iq") == {}


# --- parser integration ---------------------------------------------------------------------


def test_read_iq_exposes_the_sidecar_center_frequency(tmp_path: Path) -> None:
    path, _, fmt = _ssb_capture(tmp_path)
    _write_sidecar(tmp_path, "ssb0")

    signal = read_iq(path, sample_rate=FS, fmt=fmt)

    assert signal.metadata["center_frequency_hz"] == TRUE_CARRIER
    assert signal.metadata["center_frequency_source"] == "sigmf_sidecar"


def test_read_iq_without_a_sidecar_is_unchanged(tmp_path: Path) -> None:
    path, _, fmt = _ssb_capture(tmp_path)

    signal = read_iq(path, sample_rate=FS, fmt=fmt)

    assert "center_frequency_hz" not in signal.metadata
    assert signal.metadata["dtype"] == fmt.dtype
    assert signal.metadata["byte_order"] == fmt.byte_order
    assert signal.metadata["path"] == str(path)
    assert signal.source_format == "iq"


def test_read_wav_picks_up_a_sidecar_too(tmp_path: Path) -> None:
    # An ordinary WAV carries no RF tuning frequency; a sidecar is the reliable route.
    spec = AnalogSampleSpec(scheme="am_ssb", sample_rate_hz=FS, num_samples=N,
                            carrier_offset_hz=TRUE_CARRIER, sideband="upper", seed=101)
    generate_analog_sample(spec, tmp_path, "ssb0")
    _write_sidecar(tmp_path, "ssb0")

    signal = read_wav(tmp_path / "ssb0.wav")

    assert signal.metadata["center_frequency_hz"] == TRUE_CARRIER
    assert signal.metadata["channel_mode"] == "stereo_iq"


def test_read_wav_without_a_sidecar_is_unchanged(tmp_path: Path) -> None:
    spec = AnalogSampleSpec(scheme="am_ssb", sample_rate_hz=FS, num_samples=N,
                            carrier_offset_hz=TRUE_CARRIER, sideband="upper", seed=101)
    generate_analog_sample(spec, tmp_path, "ssb0")

    signal = read_wav(tmp_path / "ssb0.wav")

    assert "center_frequency_hz" not in signal.metadata
    assert signal.metadata["channel_mode"] == "stereo_iq"
    assert signal.sample_rate == FS


def test_an_explicit_sample_rate_argument_still_wins_over_the_sidecar(tmp_path: Path) -> None:
    path, _, fmt = _ssb_capture(tmp_path)
    _write_sidecar(tmp_path, "ssb0", sample_rate=48_000.0)

    signal = read_iq(path, sample_rate=FS, fmt=fmt)

    assert signal.sample_rate == FS


def test_the_sidecar_supplies_a_sample_rate_when_the_caller_does_not(tmp_path: Path) -> None:
    path, _, fmt = _ssb_capture(tmp_path)
    _write_sidecar(tmp_path, "ssb0")

    signal = read_iq(path, fmt=fmt)

    assert signal.sample_rate == FS


# --- preservation through the pipeline ----------------------------------------------------------


def test_center_frequency_survives_preprocessing(tmp_path: Path) -> None:
    path, _, fmt = _ssb_capture(tmp_path)
    _write_sidecar(tmp_path, "ssb0")

    processed = preprocess(read_iq(path, sample_rate=FS, fmt=fmt))

    assert processed.metadata["center_frequency_hz"] == TRUE_CARRIER
    assert processed.metadata["preprocessed"] is True


def test_parameter_estimation_uses_the_ingested_center_frequency(tmp_path: Path) -> None:
    path, _, fmt = _ssb_capture(tmp_path)
    _write_sidecar(tmp_path, "ssb0")

    estimate = estimate_parameters(preprocess(read_iq(path, sample_rate=FS, fmt=fmt)))

    assert estimate.carrier_frequency_hz == pytest.approx(TRUE_CARRIER)
    assert estimate.method.startswith("hardware_center_frequency")


def test_without_a_sidecar_estimation_falls_back_to_the_biased_centroid(tmp_path: Path) -> None:
    # Pins the Entry 029 negative result: the blind estimate remains badly wrong, and
    # this entry does not pretend otherwise.
    path, _, fmt = _ssb_capture(tmp_path)

    estimate = estimate_parameters(preprocess(read_iq(path, sample_rate=FS, fmt=fmt)))

    assert estimate.method == "welch_psd_centroid+nth_power"
    assert abs(estimate.carrier_frequency_hz - TRUE_CARRIER) > 500.0


# --- the point of the exercise: SSB recovery -------------------------------------------------------


def test_ssb_recovery_works_end_to_end_with_sidecar_metadata(tmp_path: Path) -> None:
    path, message, fmt = _ssb_capture(tmp_path)
    _write_sidecar(tmp_path, "ssb0")
    signal = preprocess(read_iq(path, sample_rate=FS, fmt=fmt))
    estimate = estimate_parameters(signal)

    recovered = demodulate_ssb(signal.iq, signal.sample_rate, estimate.carrier_frequency_hz)

    assert _corr(recovered, message) > 0.99


def test_ssb_recovery_still_fails_without_the_sidecar(tmp_path: Path) -> None:
    path, message, fmt = _ssb_capture(tmp_path)
    signal = preprocess(read_iq(path, sample_rate=FS, fmt=fmt))
    estimate = estimate_parameters(signal)

    recovered = demodulate_ssb(signal.iq, signal.sample_rate, estimate.carrier_frequency_hz)

    assert abs(_corr(recovered, message)) < 0.2


# --- anti-leakage ------------------------------------------------------------------------------------


def test_the_synthetic_generator_does_not_write_a_sigmf_sidecar(tmp_path: Path) -> None:
    """Production ingestion must never be able to read the generator's ground truth."""
    spec = AnalogSampleSpec(scheme="am_ssb", sample_rate_hz=FS, num_samples=N,
                            carrier_offset_hz=TRUE_CARRIER, sideband="upper", seed=101)
    generate_analog_sample(spec, tmp_path, "ssb0")

    assert list(tmp_path.glob("*.sigmf-meta")) == []
    assert not (tmp_path / "ssb0.sigmf-meta").exists()


def test_ingestion_ignores_the_generators_ground_truth_json(tmp_path: Path) -> None:
    # ssb0.json holds the true carrier. Ingestion must not consume it.
    path, _, fmt = _ssb_capture(tmp_path)
    assert json.loads((tmp_path / "ssb0.json").read_text())["analog"]["carrier_offset_hz"] == TRUE_CARRIER

    signal = read_iq(path, sample_rate=FS, fmt=fmt)

    assert "center_frequency_hz" not in signal.metadata


def test_the_sidecar_reader_never_looks_at_a_plain_json_file(tmp_path: Path) -> None:
    (tmp_path / "cap.iq").write_bytes(b"\x00\x00" * 8)
    (tmp_path / "cap.json").write_text(
        json.dumps({"captures": [{"core:frequency": 12_345.0}]}), encoding="utf-8")

    assert read_sigmf_sidecar(tmp_path / "cap.iq") == {}


# --- digital behaviour must not move --------------------------------------------------------------------


def test_headerless_digital_iq_ingestion_is_unchanged() -> None:
    v1 = Path("data/synthetic_v1/captures")
    if not v1.exists():
        pytest.skip("frozen V1 dataset not present")

    signal = read_iq(v1 / "QPSK_snr20dB_r000.iq", sample_rate=200_000.0)

    assert "center_frequency_hz" not in signal.metadata
    assert signal.metadata["dtype"] == "int16"
    assert signal.iq.size > 0


def test_frozen_v1_estimation_is_unaffected_by_this_entry() -> None:
    v1 = Path("data/synthetic_v1/captures")
    if not v1.exists():
        pytest.skip("frozen V1 dataset not present")

    estimate = estimate_parameters(
        preprocess(read_iq(v1 / "QPSK_snr20dB_r000.iq", sample_rate=200_000.0)))

    # No sidecar exists for V1, so the centroid path is still used - unchanged behaviour.
    assert estimate.method == "welch_psd_centroid+nth_power"


def test_a_container_built_by_hand_still_works() -> None:
    # No sidecar can reach a hand-built container: metadata stays empty and the
    # estimator's own analysis is used, exactly as before this entry.
    time = np.arange(4_096) / FS
    signal = UnifiedSignalContainer(np.exp(2j * np.pi * 5_000 * time), FS)

    estimate = estimate_parameters(signal)

    assert "center_frequency_hz" not in signal.metadata
    assert estimate.method == "welch_psd_centroid+nth_power"
    assert estimate.carrier_frequency_hz == pytest.approx(5_000.0, abs=FS / 1024)


def test_metadata_supplied_directly_by_a_caller_is_still_honoured() -> None:
    # The pre-existing contract from test_dsp_fusion: an explicit hardware centre
    # frequency in metadata wins. This entry only adds a new way to populate it.
    time = np.arange(4_096) / FS
    signal = UnifiedSignalContainer(
        np.exp(2j * np.pi * 5_000 * time), FS, metadata={"center_frequency_hz": 915_000_000})

    estimate = estimate_parameters(signal)

    assert estimate.carrier_frequency_hz == 915_000_000
    assert estimate.method.startswith("hardware_center_frequency")
