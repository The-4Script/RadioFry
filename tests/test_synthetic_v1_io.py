"""Round-trip tests: V1 writers must be readable by RadioFry's own ingestion path."""

from pathlib import Path

import numpy as np
import pytest

from radiofry.ingestion.iq_parser import IQFormat, read_iq
from radiofry.ingestion.wav_parser import read_wav
from radiofry.synthetic_gen.v1 import SampleSpec, generate_source_bits, modulate
from radiofry.synthetic_gen.v1.writers import write_iq_file, write_wav_file


def _waveform(modulation: str = "QPSK", num_symbols: int = 256) -> tuple[np.ndarray, SampleSpec]:
    spec = SampleSpec(
        modulation=modulation,
        num_symbols=num_symbols,
        samples_per_symbol=8,
        sample_rate_hz=200_000.0,
        seed=21,
    )
    return modulate(generate_source_bits(spec), spec), spec


def test_int16_iq_file_round_trips_through_read_iq(tmp_path: Path) -> None:
    waveform, spec = _waveform()
    path = tmp_path / "capture.iq"

    report = write_iq_file(path, waveform, fmt=IQFormat("int16", "little"))
    signal = read_iq(path, sample_rate=spec.sample_rate_hz, fmt=IQFormat("int16", "little"))

    recovered = signal.iq / report.scale_factor
    np.testing.assert_allclose(recovered.real, waveform.real, atol=2.0 / report.scale_factor)
    np.testing.assert_allclose(recovered.imag, waveform.imag, atol=2.0 / report.scale_factor)
    assert signal.iq.size == waveform.size
    assert signal.sample_rate == spec.sample_rate_hz


def test_int16_iq_file_contains_no_header_bytes(tmp_path: Path) -> None:
    waveform, _ = _waveform(num_symbols=10)
    path = tmp_path / "capture.iq"

    report = write_iq_file(path, waveform, fmt=IQFormat("int16", "little"))

    assert path.stat().st_size == waveform.size * 2 * 2
    assert report.bytes_written == path.stat().st_size
    assert report.channel_layout == "interleaved_iq"


def test_float32_iq_file_round_trips_without_loss(tmp_path: Path) -> None:
    waveform, spec = _waveform()
    path = tmp_path / "capture.f32.iq"

    report = write_iq_file(path, waveform, fmt=IQFormat("float32", "little"))
    signal = read_iq(path, sample_rate=spec.sample_rate_hz, fmt=IQFormat("float32", "little"))

    assert report.scale_factor == 1.0
    np.testing.assert_array_equal(signal.iq, waveform)


def test_big_endian_int16_iq_file_round_trips(tmp_path: Path) -> None:
    waveform, spec = _waveform(num_symbols=64)
    path = tmp_path / "capture.be.iq"

    report = write_iq_file(path, waveform, fmt=IQFormat("int16", "big"))
    signal = read_iq(path, sample_rate=spec.sample_rate_hz, fmt=IQFormat("int16", "big"))

    recovered = signal.iq / report.scale_factor
    np.testing.assert_allclose(recovered.real, waveform.real, atol=2.0 / report.scale_factor)
    assert report.byte_order == "big"


def test_int16_writer_uses_the_requested_headroom_without_clipping(tmp_path: Path) -> None:
    waveform, _ = _waveform("16QAM")
    path = tmp_path / "capture.iq"

    report = write_iq_file(path, waveform, fmt=IQFormat("int16", "little"), full_scale_fraction=0.95)
    raw = np.fromfile(path, dtype="<i2")

    assert report.clipped_samples == 0
    assert np.max(np.abs(raw)) <= 32767
    assert np.max(np.abs(raw)) >= 0.90 * 32767


def test_int16_writer_handles_an_all_zero_waveform(tmp_path: Path) -> None:
    path = tmp_path / "zeros.iq"

    report = write_iq_file(path, np.zeros(16, dtype=np.complex64), fmt=IQFormat("int16", "little"))

    assert report.scale_factor == 1.0
    assert report.clipped_samples == 0
    np.testing.assert_array_equal(np.fromfile(path, dtype="<i2"), np.zeros(32, dtype="<i2"))


def test_stereo_int16_wav_round_trips_through_read_wav(tmp_path: Path) -> None:
    waveform, spec = _waveform()
    path = tmp_path / "capture.wav"

    report = write_wav_file(path, waveform, spec.sample_rate_hz, dtype="int16")
    signal = read_wav(path)

    recovered = signal.iq * 32768.0 / report.scale_factor
    np.testing.assert_allclose(recovered.real, waveform.real, atol=4.0 / report.scale_factor)
    np.testing.assert_allclose(recovered.imag, waveform.imag, atol=4.0 / report.scale_factor)
    assert signal.metadata["channel_mode"] == "stereo_iq"
    assert signal.sample_rate == spec.sample_rate_hz
    assert report.channel_layout == "stereo_iq"


def test_stereo_float32_wav_round_trips_without_scaling(tmp_path: Path) -> None:
    waveform, spec = _waveform()
    path = tmp_path / "capture.f32.wav"

    report = write_wav_file(path, waveform, spec.sample_rate_hz, dtype="float32")
    signal = read_wav(path)

    assert report.scale_factor == 1.0
    np.testing.assert_allclose(signal.iq.real, waveform.real, atol=1e-6)
    np.testing.assert_allclose(signal.iq.imag, waveform.imag, atol=1e-6)


def test_wav_writer_rejects_a_non_integer_sample_rate(tmp_path: Path) -> None:
    waveform, _ = _waveform(num_symbols=8)

    with pytest.raises(ValueError, match="integer"):
        write_wav_file(tmp_path / "bad.wav", waveform, 1_000.5, dtype="int16")


def test_writers_reject_an_unsupported_dtype(tmp_path: Path) -> None:
    waveform, _ = _waveform(num_symbols=8)

    with pytest.raises(ValueError, match="dtype"):
        write_wav_file(tmp_path / "bad.wav", waveform, 8_000, dtype="int8")
