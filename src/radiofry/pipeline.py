"""Runtime orchestration for the format-independent analysis path."""

from pathlib import Path
from typing import Any

from .contracts import UnifiedSignalContainer
from .dsp.cyclostationary import estimate_modulation_family
from .dsp.parameter_estimation import estimate_parameters
from .dsp.preprocessing import preprocess
from .ingestion.iq_parser import IQFormat, read_iq
from .ingestion.wav_parser import read_wav
from .models.modulation_inference import ModulationPrediction, predict_modulation
from .models.bitstream_inference import predict_bitstream
from .decoding.demodulators.dispatch import demodulate_capture
from .correlation.bitstream_correlation import correlate_bitstream
from .decoding.deinterleave_search import search_deinterleave
from .decoding.fec.dispatch import decode_fec
from .reporting.report_builder import build_report


def load_capture(path: str | Path, *, sample_rate: float | None = None, iq_format: IQFormat | None = None) -> UnifiedSignalContainer:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return read_wav(path)
    if path.suffix.lower() == ".iq":
        return read_iq(path, sample_rate=sample_rate, fmt=iq_format)
    raise ValueError("supported capture formats are .wav and .iq")


def analyze_capture(
    signal: UnifiedSignalContainer,
    *,
    target_sample_rate: float | None = None,
    model_path: str | Path = "models_saved/modulation_cnn.pt",
    bits: Any | None = None,
    interleaver_model_path: str | Path = "models_saved/interleaver_classifier.pkl",
    fec_model_path: str | Path = "models_saved/fec_classifier.pkl",
) -> dict[str, Any]:
    processed = preprocess(signal, target_sample_rate=target_sample_rate)
    parameters = estimate_parameters(processed)
    classical = estimate_modulation_family(processed.iq)
    prediction = predict_modulation(processed, model_path)
    fusion = None
    if prediction.available:
        from .fusion.confidence_fusion import fuse_modulation
        fusion = fuse_modulation(prediction.label, prediction.confidence, classical.family, alternatives=tuple(label for label, _ in prediction.top_k[1:]))
    demodulation = None
    if bits is None and fusion is not None:
        dispatched = demodulate_capture(processed, fusion.label, parameters)
        demodulation = dispatched
        if dispatched.available and dispatched.result is not None:
            bits = dispatched.result.bits
    bitstream_stages: dict[str, Any] = {
        "available": bits is not None,
        "message": "Provide demodulated bits to run interleaver and FEC classification." if bits is None else "",
    }
    if bits is not None:
        interleaver = predict_bitstream(bits, interleaver_model_path)
        fec = predict_bitstream(bits, fec_model_path)
        deinterleaved = search_deinterleave(bits, interleaver.label if interleaver.available else "unknown")
        decoded = decode_fec(deinterleaved.bits, fec.label if fec.available else "unknown")
        correlation = correlate_bitstream(decoded.bits)
        bitstream_stages.update({
            "interleaver": interleaver,
            "deinterleaving": deinterleaved,
            "fec": fec,
            "fec_decoding": decoded,
            "correlation": correlation,
            "bits": len(bits),
        })
    demodulation_stage = demodulation or {"available": False, "message": "Demodulation was not attempted because the fused modulation decision was unavailable."}
    return build_report(
        source={"format": processed.source_format, "sample_rate": processed.sample_rate, "samples": processed.iq.size},
        stages={"parameters": parameters, "classical_modulation": classical, "cnn_modulation": prediction, "fusion": fusion or {"label": "Unclassified", "trust_score": 0.0, "review_recommended": True, "message": prediction.message}, "demodulation": demodulation_stage, "bitstream_analysis": bitstream_stages},
    )
