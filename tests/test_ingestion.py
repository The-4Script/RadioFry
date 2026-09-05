from pathlib import Path

import numpy as np
from scipy.io import wavfile

from radiofry.dsp.preprocessing import preprocess
from radiofry.ingestion.iq_parser import IQFormat, read_iq
from radiofry.ingestion.wav_parser import read_wav


def test_read_interleaved_int16_iq(tmp_path: Path) -> None:
    values = np.array([1, -2, 3, -4, 5, -6], dtype="<i2")
    path = tmp_path / "capture.iq"
    values.tofile(path)

    signal = read_iq(path, sample_rate=2_000, fmt=IQFormat("int16", "little"))

    np.testing.assert_allclose(signal.iq, [1 - 2j, 3 - 4j, 5 - 6j])
    assert signal.sample_rate == 2_000
    assert signal.duration_sec == 0.0015


def test_read_stereo_wav_as_iq(tmp_path: Path) -> None:
    path = tmp_path / "capture.wav"
    wavfile.write(path, 8_000, np.array([[100, -50], [200, -100]], dtype=np.int16))

    signal = read_wav(path)

    assert signal.metadata["channel_mode"] == "stereo_iq"
    assert signal.iq.dtype == np.complex64
    np.testing.assert_allclose(signal.iq.real, [100 / 32768, 200 / 32768])


def test_preprocess_removes_dc_and_normalizes_power() -> None:
    from radiofry.contracts import UnifiedSignalContainer

    signal = UnifiedSignalContainer(np.array([3 + 1j, 5 + 1j, 7 + 1j]), 1_000)
    processed = preprocess(signal)

    np.testing.assert_allclose(np.mean(processed.iq), 0, atol=1e-6)
    np.testing.assert_allclose(np.mean(np.abs(processed.iq) ** 2), 1, atol=1e-6)