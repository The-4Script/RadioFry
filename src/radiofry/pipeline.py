"""Runtime orchestration for the format-independent analysis path."""

from pathlib import Path
from typing import Any
from dataclasses import replace

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
from .runtime import check_runtime_artifacts


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# BANK.md Entry 040. The previous default, `modulation_cnn.pt`, is an 11-class RadioML
# model; Entry 039 measured it at 35.4% fused accuracy on RadioFry's own synthetic
# distribution, collapsing to 10.4% at 32 samples per symbol. This checkpoint is trained
# by `training/train_v2_synthetic.py` across samples-per-symbol 4/8/16/32 and measures
# 99.5% fused on held-out seeds. It is digital-only: analog subtypes are decided by the
# classical gate (Entry 032), which never consulted the CNN, and analog routing was
# verified unchanged at 6/6 for AM-DSB, AM-SSB and WBFM. The RadioML checkpoint is
# retained in models_saved/ so the Entry 039 baseline stays reproducible.
DEFAULT_MODULATION_MODEL = "models_saved/modulation_cnn_v3_spsaug.pt"
DEFAULT_MAX_CAPTURE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_CAPTURE_SAMPLES = 5_000_000


def _resolve_default_path(path: str | Path, relative_to_repository: bool = True) -> Path:
    candidate = Path(path)
    if relative_to_repository and not candidate.is_absolute():
        return _REPOSITORY_ROOT / candidate
    return candidate


def load_capture(
    path: str | Path,
    *,
    sample_rate: float | None = None,
    iq_format: IQFormat | None = None,
    max_bytes: int | None = DEFAULT_MAX_CAPTURE_BYTES,
    max_samples: int | None = DEFAULT_MAX_CAPTURE_SAMPLES,
) -> UnifiedSignalContainer:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return read_wav(path, max_bytes=max_bytes, max_samples=max_samples)
    if path.suffix.lower() == ".iq":
        return read_iq(path, sample_rate=sample_rate, fmt=iq_format, max_bytes=max_bytes, max_samples=max_samples)
    raise ValueError("supported capture formats are .wav and .iq")


def analyze_capture(
    signal: UnifiedSignalContainer,
    *,
    target_sample_rate: float | None = None,
    model_path: str | Path = DEFAULT_MODULATION_MODEL,
    bits: Any | None = None,
    interleaver_model_path: str | Path = "models_saved/interleaver_classifier.pkl",
    fec_model_path: str | Path = "models_saved/fec_classifier.pkl",
    symbol_rate_override: float | None = None,
    interleaver_override: str | None = None,
    fec_override: str | None = None,
) -> dict[str, Any]:
    processed = preprocess(signal, target_sample_rate=target_sample_rate)
    parameters = estimate_parameters(processed)
    if symbol_rate_override is not None:
        if symbol_rate_override <= 0:
            raise ValueError("symbol_rate_override must be positive")
        parameters = replace(parameters, symbol_rate_hz=float(symbol_rate_override), method="manual_symbol_rate")
    classical = estimate_modulation_family(processed.iq)
    prediction = predict_modulation(processed, _resolve_default_path(model_path))
    fusion = None
    if prediction.available:
        from .fusion.confidence_fusion import fuse_modulation
        # top_k[1:] is already ranked by the CNN; pass the confidences through too so
        # fusion can recover an analog label without a second inference pass.
        ranked = tuple((label, confidence) for label, confidence in prediction.top_k[1:])
        fusion = fuse_modulation(prediction.label, prediction.confidence, classical.family, alternatives=tuple(label for label, _ in ranked), ranked_alternatives=ranked, classical_evidence=getattr(classical, "evidence", None), classical_confidence=classical.confidence)
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
        interleaver = predict_bitstream(bits, _resolve_default_path(interleaver_model_path))
        selected_interleaver = interleaver_override or (interleaver.label if interleaver.available else "unknown")
        if selected_interleaver not in {"auto", "unknown", "none", "block", "convolutional", "diagonal", "pseudo_random"}:
            raise ValueError(f"Unsupported interleaver override: {selected_interleaver}")
        if selected_interleaver == "auto":
            selected_interleaver = interleaver.label if interleaver.available else "unknown"
        deinterleaved = search_deinterleave(bits, selected_interleaver)
        fec = predict_bitstream(deinterleaved.bits, _resolve_default_path(fec_model_path))
        selected_fec = fec_override or (fec.label if fec.available else "unknown")
        if selected_fec not in {"auto", "unknown", "none", "convolutional", "reed_solomon", "concatenated", "ldpc"}:
            raise ValueError(f"Unsupported FEC override: {selected_fec}")
        if selected_fec == "auto":
            selected_fec = fec.label if fec.available else "unknown"
        decoded = decode_fec(deinterleaved.bits, selected_fec)
        correlation = correlate_bitstream(decoded.bits)
        bitstream_stages.update({
            "interleaver": interleaver,
            "selected_interleaver": selected_interleaver,
            "deinterleaving": deinterleaved,
            "fec": fec,
            "selected_fec": selected_fec,
            "fec_decoding": decoded,
            "correlation": correlation,
            "bits": len(bits),
        })
    demodulation_stage = demodulation or {"available": False, "message": "Demodulation was not attempted because the fused modulation decision was unavailable."}
    runtime = check_runtime_artifacts({
        "modulation": model_path,
        "interleaver": interleaver_model_path,
        "fec": fec_model_path,
    })
    return build_report(
        source={"format": processed.source_format, "sample_rate": processed.sample_rate, "samples": processed.iq.size},
        stages={"runtime": runtime, "parameters": parameters, "classical_modulation": classical, "cnn_modulation": prediction, "fusion": fusion or {"label": "Unclassified", "trust_score": 0.0, "review_recommended": True, "message": prediction.message}, "demodulation": demodulation_stage, "bitstream_analysis": bitstream_stages},
    )
