import json
import h5py

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import radiofry.pipeline as pipeline
from radiofry.models.modulation_cnn import ModulationCNN
from radiofry.models.signal_features import add_signal_features
from radiofry.reporting.report_builder import build_pdf_report, build_report, report_json
from radiofry.training.train_modulation import load_rml_dataset


def test_modulation_cnn_output_shape() -> None:
    model = ModulationCNN(input_channels=2, num_classes=11)
    output = model(torch.zeros(4, 2, 128))

    assert output.shape == (4, 11)


def test_engineered_signal_features_have_four_channels() -> None:
    features = add_signal_features(np.ones((2, 128), dtype=np.float32), include_engineered=True)

    assert features.shape == (4, 128)
    assert np.isfinite(features).all()


def test_report_serializes_dataclasses_and_numpy_values() -> None:
    report = build_report(source={"samples": np.int64(8)}, stages={"scores": np.array([0.2, 0.8])})

    loaded = json.loads(report_json(report))
    assert loaded["source"]["samples"] == 8
    assert loaded["stages"]["scores"] == [0.2, 0.8]


def test_report_serializes_complex_values_as_coordinates() -> None:
    loaded = json.loads(report_json(build_report(source={}, stages={"symbols": np.array([1 + 2j])})))

    assert loaded["stages"]["symbols"] == [{"real": 1.0, "imag": 2.0}]


def test_pdf_report_is_generated(tmp_path) -> None:
    output = tmp_path / "report.pdf"

    build_pdf_report(build_report(source={"format": "test"}, stages={}), str(output))

    assert output.read_bytes().startswith(b"%PDF")


def test_rml_hdf5_loader_reads_iq_labels_and_snrs(tmp_path) -> None:
    path = tmp_path / "mini-rml.h5"
    with h5py.File(path, "w") as handle:
        handle["X"] = np.zeros((4, 2, 16), dtype=np.float32)
        handle["Y"] = np.array([b"BPSK", b"QPSK", b"BPSK", b"QPSK"])
        handle["Z"] = np.array([-10, 0, 10, 18])

    frames, targets, snrs, labels = load_rml_dataset(path, max_samples=4)

    assert frames.shape == (4, 2, 16)
    assert targets.tolist() == [0, 1, 0, 1]
    assert snrs.tolist() == [-10, 0, 10, 18]
    assert labels == ["BPSK", "QPSK"]


def test_rml_hdf5_loader_reads_official_sibling_class_names(tmp_path) -> None:
    path = tmp_path / "signals.h5"
    path.write_bytes(b"")
    path.unlink()
    with h5py.File(path, "w") as handle:
        handle["X"] = np.zeros((2, 8, 2), dtype=np.float32)
        handle["Y"] = np.eye(2, dtype=np.int64)
        handle["Z"] = np.array([[0], [10]])
    (tmp_path / "classes.txt").write_text("classes = ['BPSK', 'QPSK']", encoding="utf-8")

    _, _, _, labels = load_rml_dataset(path, max_samples=2)

    assert labels == ["BPSK", "QPSK"]


def test_pipeline_classifies_fec_after_deinterleaving(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPrediction:
        available = True
        label = "none"
        confidence = 1.0
        top_k = (("BPSK", 1.0),)
        message = ""

    class DummyDeinterleaving:
        bits = np.array([1, 0, 1, 0], dtype=np.uint8)

    class DummyDecoded:
        bits = np.array([1, 0], dtype=np.uint8)

    signal = pipeline.UnifiedSignalContainer(
        iq=np.ones(16, dtype=np.complex64),
        sample_rate=8.0,
        source_format="test",
        metadata={},
    )
    classified_inputs: list[np.ndarray] = []

    monkeypatch.setattr(pipeline, "preprocess", lambda signal, target_sample_rate=None: signal)
    monkeypatch.setattr(pipeline, "estimate_parameters", lambda signal: {})
    monkeypatch.setattr(pipeline, "estimate_modulation_family", lambda iq: type("Classical", (), {"family": "PSK-like"})())
    monkeypatch.setattr(pipeline, "predict_modulation", lambda signal, model_path: DummyPrediction())
    monkeypatch.setattr(pipeline, "predict_bitstream", lambda bits, model_path: classified_inputs.append(np.array(bits)) or DummyPrediction())
    monkeypatch.setattr(pipeline, "search_deinterleave", lambda bits, interleaver_type: DummyDeinterleaving())
    monkeypatch.setattr(pipeline, "decode_fec", lambda bits, scheme: DummyDecoded())
    monkeypatch.setattr(pipeline, "correlate_bitstream", lambda bits: {})
    monkeypatch.setattr(pipeline, "build_report", lambda source, stages: stages)
    monkeypatch.setattr(pipeline, "demodulate_capture", lambda signal, label, parameters: type("Dispatch", (), {"available": False, "result": None})())

    pipeline.analyze_capture(signal, bits=np.array([0, 1, 0, 1], dtype=np.uint8))

    assert len(classified_inputs) == 2
    np.testing.assert_array_equal(classified_inputs[0], [0, 1, 0, 1])
    np.testing.assert_array_equal(classified_inputs[1], [1, 0, 1, 0])