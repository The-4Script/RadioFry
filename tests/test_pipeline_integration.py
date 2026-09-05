import json
from pathlib import Path

import numpy as np

import radiofry.pipeline as pipeline
from radiofry.contracts import UnifiedSignalContainer
from radiofry.correlation.bitstream_correlation import CorrelationResult
from radiofry.decoding.deinterleave_search import DeinterleaveResult
from radiofry.decoding.fec.viterbi_wrapper import FECResult
from radiofry.models.bitstream_inference import BitstreamPrediction
from radiofry.models.modulation_inference import ModulationPrediction
from radiofry.reporting.report_builder import report_json


def _signal(sample_rate: float | None = 8_000.0) -> UnifiedSignalContainer:
    time = np.arange(256, dtype=np.float32)
    return UnifiedSignalContainer(np.exp(1j * 2 * np.pi * time / 16), sample_rate, "fixture", {})


def test_pipeline_unknown_rate_degrades_without_crashing(tmp_path: Path) -> None:
    bits = np.array([0, 1, 1, 0] * 32, dtype=np.uint8)

    report = pipeline.analyze_capture(
        _signal(None),
        bits=bits,
        model_path=tmp_path / "missing-modulation.pt",
        interleaver_model_path=tmp_path / "missing-interleaver.pkl",
        fec_model_path=tmp_path / "missing-fec.pkl",
    )

    encoded = json.loads(report_json(report))
    assert encoded["stages"]["parameters"]["symbol_rate_hz"] is None
    assert encoded["stages"]["bitstream_analysis"]["available"] is True
    assert encoded["stages"]["bitstream_analysis"]["fec"]["available"] is False
    assert encoded["stages"]["runtime"]["ready"] is False


def test_pipeline_open_set_rejection_skips_demodulation(monkeypatch) -> None:
    prediction = ModulationPrediction("BPSK", 0.2, (("BPSK", 0.2), ("QPSK", 0.1)))
    classical = type("Classical", (), {"family": "PSK-like", "confidence": 0.8})()
    monkeypatch.setattr(pipeline, "predict_modulation", lambda signal, path: prediction)
    monkeypatch.setattr(pipeline, "estimate_modulation_family", lambda iq: classical)

    report = pipeline.analyze_capture(_signal())

    fusion = report["stages"]["fusion"]
    demodulation = report["stages"]["demodulation"]
    assert fusion["label"] == "Unclassified"
    assert fusion["review_recommended"]
    assert not demodulation["available"]


def test_pipeline_report_contains_all_bitstream_stages(monkeypatch) -> None:
    deinterleaved = np.array([1, 0, 1, 0], dtype=np.uint8)
    monkeypatch.setattr(pipeline, "predict_bitstream", lambda bits, path: BitstreamPrediction("none", 1.0))
    monkeypatch.setattr(pipeline, "search_deinterleave", lambda bits, kind: DeinterleaveResult(deinterleaved, kind, {}, 0.0))
    monkeypatch.setattr(pipeline, "decode_fec", lambda bits, scheme: FECResult(deinterleaved, scheme, True))
    monkeypatch.setattr(pipeline, "correlate_bitstream", lambda bits: CorrelationResult(None, (), None, bits[:0], bits, 0.0, 0.0))

    report = pipeline.analyze_capture(_signal(), bits=np.array([0, 1, 0, 1], dtype=np.uint8))
    stages = report["stages"]["bitstream_analysis"]

    assert {"interleaver", "deinterleaving", "fec", "fec_decoding", "correlation", "bits"} <= stages.keys()
    json.loads(report_json(report))


def test_pipeline_applies_manual_symbol_interleaver_and_fec_overrides(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "predict_bitstream", lambda bits, path: BitstreamPrediction("none", 1.0))

    def deinterleave(bits, kind):
        calls["interleaver"] = kind
        return DeinterleaveResult(np.asarray(bits), kind, {}, 0.0)

    def decode(bits, scheme):
        calls["fec"] = scheme
        return FECResult(np.asarray(bits), scheme, True)

    monkeypatch.setattr(pipeline, "search_deinterleave", deinterleave)
    monkeypatch.setattr(pipeline, "decode_fec", decode)
    report = pipeline.analyze_capture(
        _signal(),
        bits=np.array([0, 1, 0, 1], dtype=np.uint8),
        symbol_rate_override=500.0,
        interleaver_override="diagonal",
        fec_override="reed_solomon",
    )

    assert report["stages"]["parameters"]["symbol_rate_hz"] == 500.0
    assert calls == {"interleaver": "diagonal", "fec": "reed_solomon"}
    assert report["stages"]["bitstream_analysis"]["selected_fec"] == "reed_solomon"
