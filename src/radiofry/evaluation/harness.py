"""Run the unmodified RadioFry pipeline over a Synthetic V1 dataset and score it.

The harness only calls the public pipeline entry points (``load_capture`` and
``analyze_capture``) and reads what the resulting report already contains. It
never re-implements a stage, never patches a known mismatch, and records any
metric the pipeline does not expose as explicitly unavailable.
"""

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from radiofry.ingestion.iq_parser import IQFormat
from radiofry.pipeline import analyze_capture, load_capture
from radiofry.synthetic_gen.v1 import load_ground_truth

from .metrics import expected_family_for, relative_error, score_bits, signed_error

RESULT_SCHEMA = "radiofry.evaluation.v1"

# Metrics that cannot be fairly scored against V1 ground truth, and why.
UNAVAILABLE_METRICS: dict[str, str] = {
    "occupied_bandwidth": (
        "V1 ground truth does not define a 99%-power occupied bandwidth. Rectangular pulses have "
        "sinc-shaped sidelobes, so no closed-form truth value exists. The raw estimate and its ratio "
        "to the true symbol rate are recorded for inspection instead of an error."
    ),
    "sample_rate_estimation": (
        "The pipeline never estimates sample rate; it is supplied at ingestion (IQ) or read from the "
        "file header (WAV). There is nothing to score."
    ),
    "fec_decoding_correctness": (
        "V1 applies no FEC, so decoder correctness has no ground truth. Only the 'no FEC' "
        "classification and the reported success flag are recorded."
    ),
    "interleaver_recovery_correctness": (
        "V1 applies no interleaving, so de-interleaver correctness has no ground truth. Only the "
        "'none' classification is scored."
    ),
    "header_payload_identification": (
        "V1 carries no protocol framing, so header/payload boundaries have no ground truth. Any sync "
        "detection is counted as a false positive."
    ),
    "soft_decision_quality": "The demodulators emit hard bits only; no soft/LLR output exists to score.",
    "bit_error_rate_analog": (
        "An analog capture carries no transmitted bits, so it has no bit-error rate. The "
        "dispatcher still synthesises bits for analog labels by thresholding the demodulated "
        "waveform at its median; those are never scored, because comparing them to anything "
        "would publish a meaningless number."
    ),
    "calibrated_confidence": (
        "CNN confidence is a raw softmax value and fusion trust is a hand-set formula. Neither is "
        "calibrated, so no calibration error is reported."
    ),
}

RECORD_FIELDS: list[str] = [
    "capture_id", "modulation", "modulation_family", "order", "bits_per_symbol",
    "expected_cnn_label", "expected_family", "file_format", "capture_file", "sample_rate_source",
    "target_snr_db", "es_n0_db", "truth_sample_rate_hz", "truth_symbol_rate_hz",
    "truth_carrier_hz", "truth_snr_db", "truth_interleaver", "truth_fec", "expected_bits",
    "ingestion_ok", "ingestion_error", "pipeline_ok", "pipeline_error", "end_to_end_status",
    "est_bandwidth_hz", "bandwidth_status", "bandwidth_to_symbol_rate_ratio",
    "est_snr_db", "snr_error_db",
    "est_carrier_hz", "carrier_error_hz",
    "est_symbol_rate_hz", "symbol_rate_error_hz", "symbol_rate_rel_error", "symbol_rate_confidence",
    "parameter_method",
    "classical_family", "classical_confidence", "classical_family_correct",
    "cnn_available", "cnn_label", "cnn_confidence", "cnn_correct", "cnn_top_k",
    "cnn_topk_correct", "cnn_message",
    "fusion_label", "fusion_trust_score", "fusion_review_recommended", "fusion_rejected",
    "fusion_correct",
    "demod_available", "demod_scheme", "demod_message", "recovered_bits", "bit_count_ratio",
    "ber_status", "ber_reason", "ber_strict", "ber_aligned", "ber_alignment_offset", "compared_bits",
    "interleaver_label", "interleaver_confidence", "selected_interleaver", "interleaver_correct",
    "fec_label", "fec_confidence", "selected_fec", "fec_correct", "fec_success",
    "sync_pattern", "sync_match_score", "sync_false_positive",
]


def load_manifest(dataset_dir: str | Path) -> list[dict[str, str]]:
    """Read the V1 dataset manifest as a list of rows."""

    manifest = Path(dataset_dir) / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _blank_record() -> dict[str, Any]:
    return {field: None for field in RECORD_FIELDS}


def _load_signal(path: Path, metadata: dict[str, Any], file_format: str):
    if file_format == "wav":
        return load_capture(path), "wav_header"
    entry = next(item for item in metadata["files"] if item["file_format"] == "iq")
    signal = load_capture(
        path,
        sample_rate=metadata["signal"]["sample_rate_hz"],
        iq_format=IQFormat(entry["dtype"], entry["byte_order"]),
    )
    return signal, "ground_truth"


def evaluate_capture(
    dataset_root: str | Path,
    row: dict[str, str],
    file_format: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the pipeline on one capture and score every stage it exposes."""

    root = Path(dataset_root)
    truth = load_ground_truth(root / row["ground_truth_file"])
    metadata = truth["metadata"]
    source_bits = truth["source_bits"]
    has_source_bits = source_bits is not None
    # Bit-less (analog) captures still need an array to hand to the scorer, but must
    # never be treated as having zero-length ground truth to compare against.
    scoreable_bits = source_bits if has_source_bits else np.array([], dtype=np.uint8)
    no_bits_reason = "analog_no_transmitted_bits"

    record = _blank_record()
    record.update(
        capture_id=metadata["capture_id"],
        modulation=metadata["modulation"]["name"],
        modulation_family=metadata["modulation"]["family"],
        order=metadata["modulation"]["order"],
        bits_per_symbol=metadata["modulation"]["bits_per_symbol"],
        expected_cnn_label=metadata["modulation"]["radiofry_label"],
        expected_family=expected_family_for(metadata["modulation"]["family"]),
        file_format=file_format,
        capture_file=row.get(f"{file_format}_file", ""),
        target_snr_db=metadata["noise"]["target_snr_db"],
        es_n0_db=metadata["noise"].get("es_n0_db"),
        truth_sample_rate_hz=metadata["signal"]["sample_rate_hz"],
        truth_symbol_rate_hz=metadata["signal"]["symbol_rate_hz"],
        truth_carrier_hz=metadata["signal"]["center_frequency_hz"],
        truth_snr_db=metadata["noise"]["target_snr_db"],
        truth_interleaver=metadata["impairments"]["interleaving"],
        truth_fec=metadata["impairments"]["fec"],
        expected_bits=int(source_bits.size) if has_source_bits else None,
        bandwidth_status="unavailable_no_ground_truth",
        ingestion_ok=False,
        ingestion_error="",
        pipeline_ok=False,
        pipeline_error="",
        end_to_end_status="ingestion_failed",
    )
    record.update(_unscored_bits(
        scoreable_bits, "demodulation_unavailable" if has_source_bits else no_bits_reason))

    capture_rel = row.get(f"{file_format}_file", "")
    if not capture_rel:
        record["ingestion_error"] = f"manifest has no {file_format} file for this capture"
        return record

    try:
        signal, rate_source = _load_signal(root / capture_rel, metadata, file_format)
    except Exception as error:  # ingestion must never abort the sweep
        record["ingestion_error"] = f"{type(error).__name__}: {error}"
        return record
    record.update(ingestion_ok=True, sample_rate_source=rate_source)

    try:
        report = analyze_capture(signal) if model_path is None else analyze_capture(signal, model_path=model_path)
    except Exception as error:  # a stage crash is a measurable outcome, not a harness failure
        record.update(pipeline_error=f"{type(error).__name__}: {error}", end_to_end_status="pipeline_error")
        return record

    record["pipeline_ok"] = True
    _score_report(record, report, scoreable_bits, has_bits=has_source_bits)
    return record


def _unscored_bits(source_bits: np.ndarray, reason: str) -> dict[str, Any]:
    scored = score_bits(source_bits, None, available=False, reason=reason)
    return {
        "ber_status": scored.status,
        "ber_reason": scored.reason,
        "ber_strict": scored.strict,
        "ber_aligned": scored.aligned,
        "ber_alignment_offset": scored.alignment_offset,
        "compared_bits": scored.compared_bits,
        "recovered_bits": 0,
    }


def _score_report(
    record: dict[str, Any],
    report: dict[str, Any],
    source_bits: np.ndarray,
    *,
    has_bits: bool = True,
) -> None:
    stages = report.get("stages", {}) or {}
    parameters = stages.get("parameters") or {}
    classical = stages.get("classical_modulation") or {}
    cnn = stages.get("cnn_modulation") or {}
    fusion = stages.get("fusion") or {}
    demodulation = stages.get("demodulation") or {}
    bitstream = stages.get("bitstream_analysis") or {}

    # --- parameter estimation
    bandwidth = parameters.get("occupied_bandwidth_hz")
    record.update(
        est_bandwidth_hz=bandwidth,
        bandwidth_to_symbol_rate_ratio=relative_ratio(bandwidth, record["truth_symbol_rate_hz"]),
        est_snr_db=parameters.get("snr_db"),
        snr_error_db=signed_error(parameters.get("snr_db"), record["truth_snr_db"]),
        est_carrier_hz=parameters.get("carrier_frequency_hz"),
        carrier_error_hz=signed_error(parameters.get("carrier_frequency_hz"), record["truth_carrier_hz"]),
        est_symbol_rate_hz=parameters.get("symbol_rate_hz"),
        symbol_rate_error_hz=signed_error(parameters.get("symbol_rate_hz"), record["truth_symbol_rate_hz"]),
        symbol_rate_rel_error=relative_error(parameters.get("symbol_rate_hz"), record["truth_symbol_rate_hz"]),
        symbol_rate_confidence=parameters.get("symbol_rate_confidence"),
        parameter_method=parameters.get("method"),
    )

    # --- classical family
    family = classical.get("family")
    record.update(
        classical_family=family,
        classical_confidence=classical.get("confidence"),
        classical_family_correct=None if family is None else family == record["expected_family"],
    )

    # --- CNN
    top_k = cnn.get("top_k") or []
    top_labels = [str(item[0]) for item in top_k if isinstance(item, (list, tuple)) and item]
    cnn_label = cnn.get("label")
    record.update(
        cnn_available=cnn.get("available"),
        cnn_label=cnn_label,
        cnn_confidence=cnn.get("confidence"),
        cnn_correct=None if cnn_label is None else cnn_label == record["expected_cnn_label"],
        cnn_top_k="|".join(f"{item[0]}:{float(item[1]):.4f}" for item in top_k if len(item) >= 2),
        cnn_topk_correct=record["expected_cnn_label"] in top_labels if top_labels else None,
        cnn_message=cnn.get("message", ""),
    )

    # --- fusion
    fusion_label = fusion.get("label")
    record.update(
        fusion_label=fusion_label,
        fusion_trust_score=fusion.get("trust_score"),
        fusion_review_recommended=fusion.get("review_recommended"),
        fusion_rejected=None if fusion_label is None else fusion_label == "Unclassified",
        fusion_correct=None if fusion_label is None else fusion_label == record["expected_cnn_label"],
    )

    # --- demodulation and BER
    demod_available = bool(demodulation.get("available"))
    result = demodulation.get("result") or {}
    recovered = result.get("bits")
    record.update(
        demod_available=demod_available,
        demod_scheme=result.get("modulation"),
        demod_message=demodulation.get("message", ""),
    )
    # A bit-less capture is never scored, even though `dispatch` hands back
    # median-threshold bits for analog labels.
    scoreable = has_bits and demod_available and recovered is not None
    scored = score_bits(
        source_bits,
        recovered if scoreable else None,
        available=scoreable,
        reason=(
            ""
            if scoreable
            else "analog_no_transmitted_bits"
            if not has_bits
            else "demodulation_unavailable"
        ),
    )
    record.update(
        ber_status=scored.status,
        ber_reason=scored.reason,
        ber_strict=scored.strict,
        ber_aligned=scored.aligned,
        ber_alignment_offset=scored.alignment_offset,
        compared_bits=scored.compared_bits,
        recovered_bits=scored.recovered_bits,
        bit_count_ratio=(
            relative_ratio(float(scored.recovered_bits), float(scored.expected_bits))
            if has_bits
            else None
        ),
    )

    # --- downstream stages whose V1 ground truth is "none"
    interleaver = bitstream.get("interleaver") or {}
    fec = bitstream.get("fec") or {}
    decoded = bitstream.get("fec_decoding") or {}
    correlation = bitstream.get("correlation") or {}
    interleaver_label = interleaver.get("label")
    fec_label = fec.get("label")
    sync_pattern = correlation.get("sync_pattern")
    record.update(
        interleaver_label=interleaver_label,
        interleaver_confidence=interleaver.get("confidence"),
        selected_interleaver=bitstream.get("selected_interleaver"),
        interleaver_correct=None if interleaver_label is None else interleaver_label == record["truth_interleaver"],
        fec_label=fec_label,
        fec_confidence=fec.get("confidence"),
        selected_fec=bitstream.get("selected_fec"),
        fec_correct=None if fec_label is None else fec_label == record["truth_fec"],
        fec_success=decoded.get("success"),
        sync_pattern=sync_pattern,
        sync_match_score=correlation.get("sync_match_score"),
        sync_false_positive=None if not bitstream.get("available") else sync_pattern is not None,
    )

    record["end_to_end_status"] = _end_to_end_status(record)


def _end_to_end_status(record: dict[str, Any]) -> str:
    if record.get("fusion_label") in (None, "Unclassified"):
        return "no_modulation_decision"
    if not record.get("demod_available"):
        return "demodulation_failed"
    if record.get("ber_status") != "ok":
        return "demodulation_failed"
    return "bits_recovered"


def relative_ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return float(value) / float(reference)


def run_evaluation(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("iq", "wav"),
    limit: int | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Path]:
    """Evaluate a whole V1 dataset and write CSV, JSON and markdown outputs."""

    from .report import write_outputs

    dataset_root = Path(dataset_dir)
    rows = load_manifest(dataset_root)
    if limit is not None:
        rows = rows[:limit]
    records = [
        evaluate_capture(dataset_root, row, file_format, model_path=model_path)
        for row in rows
        for file_format in formats
        if row.get(f"{file_format}_file")
    ]
    return write_outputs(records, Path(output_dir), dataset_root=dataset_root, capture_count=len(rows))
