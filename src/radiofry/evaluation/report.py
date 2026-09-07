"""Aggregation and reporting for the Synthetic V1 evaluation harness."""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .harness import RECORD_FIELDS, RESULT_SCHEMA, UNAVAILABLE_METRICS
from .metrics import confusion_matrix

Record = dict[str, Any]


def _values(records: Sequence[Record], field: str) -> list[float]:
    return [float(r[field]) for r in records if r.get(field) is not None]


def _rate(records: Sequence[Record], field: str) -> float | None:
    flags = [bool(r[field]) for r in records if r.get(field) is not None]
    return float(np.mean(flags)) if flags else None


def _median(records: Sequence[Record], field: str) -> float | None:
    values = _values(records, field)
    return float(np.median(values)) if values else None


def _mean(records: Sequence[Record], field: str) -> float | None:
    values = _values(records, field)
    return float(np.mean(values)) if values else None


def group_stats(records: Sequence[Record]) -> dict[str, Any]:
    """Summary statistics for one slice of the evaluation."""

    symbol_rate_errors = _values(records, "symbol_rate_rel_error")
    return {
        "count": len(records),
        "ingestion_success_rate": _rate(records, "ingestion_ok"),
        "pipeline_success_rate": _rate(records, "pipeline_ok"),
        "end_to_end_success_rate": (
            float(np.mean([r.get("end_to_end_status") == "bits_recovered" for r in records])) if records else None
        ),
        "classical_family_accuracy": _rate(records, "classical_family_correct"),
        "cnn_accuracy": _rate(records, "cnn_correct"),
        "cnn_topk_accuracy": _rate(records, "cnn_topk_correct"),
        "mean_cnn_confidence": _mean(records, "cnn_confidence"),
        "fusion_accuracy": _rate(records, "fusion_correct"),
        "fusion_rejection_rate": _rate(records, "fusion_rejected"),
        "review_recommended_rate": _rate(records, "fusion_review_recommended"),
        "mean_fusion_trust_score": _mean(records, "fusion_trust_score"),
        "demod_success_rate": _rate(records, "demod_available"),
        "ber_scored_count": sum(1 for r in records if r.get("ber_status") == "ok"),
        "median_ber_strict": _median(records, "ber_strict"),
        "median_ber_aligned": _median(records, "ber_aligned"),
        "mean_snr_error_db": _mean(records, "snr_error_db"),
        "median_snr_error_db": _median(records, "snr_error_db"),
        "median_carrier_error_hz": _median(records, "carrier_error_hz"),
        "median_symbol_rate_rel_error": _median(records, "symbol_rate_rel_error"),
        "symbol_rate_within_1pct_rate": (
            float(np.mean([abs(v) <= 0.01 for v in symbol_rate_errors])) if symbol_rate_errors else None
        ),
        "median_bandwidth_to_symbol_rate_ratio": _median(records, "bandwidth_to_symbol_rate_ratio"),
        "interleaver_none_accuracy": _rate(records, "interleaver_correct"),
        "fec_none_accuracy": _rate(records, "fec_correct"),
        "sync_false_positive_rate": _rate(records, "sync_false_positive"),
    }


def _grouped(records: Sequence[Record], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Record]] = {}
    for record in records:
        buckets.setdefault(str(record.get(key)), []).append(record)
    return {name: group_stats(rows) for name, rows in sorted(buckets.items())}


def aggregate(records: Sequence[Record]) -> dict[str, Any]:
    """Build every aggregate view of the evaluation records."""

    scored = [r for r in records if r.get("pipeline_ok")]
    ber_rows: list[dict[str, Any]] = []
    keys = sorted({(r["modulation"], r["target_snr_db"], r["file_format"]) for r in records}, key=str)
    for modulation, snr, file_format in keys:
        subset = [
            r for r in records
            if r["modulation"] == modulation and r["target_snr_db"] == snr and r["file_format"] == file_format
        ]
        ber_rows.append({
            "modulation": modulation,
            "target_snr_db": snr,
            "file_format": file_format,
            "count": len(subset),
            "scored": sum(1 for r in subset if r.get("ber_status") == "ok"),
            "median_ber_strict": _median(subset, "ber_strict"),
            "median_ber_aligned": _median(subset, "ber_aligned"),
        })

    return {
        "overall": group_stats(records),
        "by_modulation": _grouped(records, "modulation"),
        "by_snr": _grouped(records, "target_snr_db"),
        "by_format": _grouped(records, "file_format"),
        "by_modulation_and_snr": {
            f"{mod}@{snr}": group_stats([r for r in records if r["modulation"] == mod and r["target_snr_db"] == snr])
            for mod, snr in sorted({(r["modulation"], r["target_snr_db"]) for r in records}, key=str)
        },
        "ber_vs_snr": ber_rows,
        "confusion_matrices": {
            "cnn": confusion_matrix([(r["expected_cnn_label"], str(r["cnn_label"])) for r in scored]),
            "classical_family": confusion_matrix([(r["expected_family"], str(r["classical_family"])) for r in scored]),
            "fusion": confusion_matrix([(r["expected_cnn_label"], str(r["fusion_label"])) for r in scored]),
        },
        "failure_reasons": {
            "end_to_end_status": dict(Counter(r.get("end_to_end_status") for r in records)),
            "ber_reason": dict(Counter(r.get("ber_reason") for r in records if r.get("ber_status") != "ok")),
            "ingestion_error": dict(Counter(r["ingestion_error"] for r in records if r.get("ingestion_error"))),
            "pipeline_error": dict(Counter(r["pipeline_error"] for r in records if r.get("pipeline_error"))),
            "demod_message": dict(Counter(r["demod_message"] for r in records if r.get("demod_message"))),
        },
    }


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%" if percent else f"{value:.{digits}g}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_summary(records: Sequence[Record], aggregates: dict[str, Any], *, dataset_root: Path) -> str:
    """Render the concise human-readable evaluation report."""

    overall = aggregates["overall"]
    lines = [
        "# RadioFry Baseline Evaluation - Synthetic Dataset V1",
        "",
        f"Dataset: `{dataset_root}`  |  Records scored: {len(records)}  "
        f"|  Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This measures the **unmodified** RadioFry pipeline on impairment-free V1 captures "
        "(CFO=0, timing offset=0, no fading, no FEC, no interleaving). Known mismatches are "
        "reported as measured, never compensated.",
        "",
        "## Ingestion and pipeline outcomes",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Ingestion success", _fmt(overall["ingestion_success_rate"], percent=True)],
                ["Pipeline completed without exception", _fmt(overall["pipeline_success_rate"], percent=True)],
                ["End-to-end (bits recovered)", _fmt(overall["end_to_end_success_rate"], percent=True)],
                ["Demodulation attempted successfully", _fmt(overall["demod_success_rate"], percent=True)],
                ["BER scored", f"{overall['ber_scored_count']} / {overall['count']}"],
            ],
        ),
        "",
        "## Modulation classification",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Classical family accuracy", _fmt(overall["classical_family_accuracy"], percent=True)],
                ["CNN top-1 accuracy", _fmt(overall["cnn_accuracy"], percent=True)],
                ["CNN top-3 accuracy", _fmt(overall["cnn_topk_accuracy"], percent=True)],
                ["Mean CNN softmax (uncalibrated)", _fmt(overall["mean_cnn_confidence"])],
                ["Fusion accuracy", _fmt(overall["fusion_accuracy"], percent=True)],
                ["Fusion rejection rate", _fmt(overall["fusion_rejection_rate"], percent=True)],
                ["Review recommended", _fmt(overall["review_recommended_rate"], percent=True)],
                ["Mean fusion trust score", _fmt(overall["mean_fusion_trust_score"])],
            ],
        ),
        "",
        "### CNN accuracy by modulation",
        "",
        _table(
            ["Modulation", "n", "CNN top-1", "CNN top-3", "Classical family", "Fusion"],
            [
                [
                    name,
                    str(stats["count"]),
                    _fmt(stats["cnn_accuracy"], percent=True),
                    _fmt(stats["cnn_topk_accuracy"], percent=True),
                    _fmt(stats["classical_family_accuracy"], percent=True),
                    _fmt(stats["fusion_accuracy"], percent=True),
                ]
                for name, stats in aggregates["by_modulation"].items()
            ],
        ),
        "",
        "### CNN confusion matrix (truth -> prediction)",
        "",
        _confusion_table(aggregates["confusion_matrices"]["cnn"]),
        "",
        "### Classical family confusion matrix",
        "",
        _confusion_table(aggregates["confusion_matrices"]["classical_family"]),
        "",
        "## Parameter estimation",
        "",
        _table(
            ["Modulation", "n", "Median SNR err (dB)", "Median carrier err (Hz)",
             "Median symbol-rate rel err", "Symbol rate within 1%", "Median BW / symbol rate"],
            [
                [
                    name,
                    str(stats["count"]),
                    _fmt(stats["median_snr_error_db"]),
                    _fmt(stats["median_carrier_error_hz"]),
                    _fmt(stats["median_symbol_rate_rel_error"]),
                    _fmt(stats["symbol_rate_within_1pct_rate"], percent=True),
                    _fmt(stats["median_bandwidth_to_symbol_rate_ratio"]),
                ]
                for name, stats in aggregates["by_modulation"].items()
            ],
        ),
        "",
        "SNR error is measured against the V1 total-band SNR definition. The pipeline estimates an "
        "in-band-mean / out-of-band-median PSD ratio, which is a different quantity - the difference "
        "is reported, not corrected.",
        "",
        "### Parameter error by SNR",
        "",
        _table(
            ["Target SNR (dB)", "n", "Median SNR err (dB)", "Median carrier err (Hz)", "Median symbol-rate rel err"],
            [
                [
                    name,
                    str(stats["count"]),
                    _fmt(stats["median_snr_error_db"]),
                    _fmt(stats["median_carrier_error_hz"]),
                    _fmt(stats["median_symbol_rate_rel_error"]),
                ]
                for name, stats in aggregates["by_snr"].items()
            ],
        ),
        "",
        "## BER vs SNR",
        "",
        "`ber_strict` is the headline number (no shifting). `ber_aligned` is a diagnostic showing the "
        "best BER over a bounded bit-shift search; a large gap means a timing/framing offset. Neither "
        "search corrects PSK phase ambiguity, which the pipeline does not resolve.",
        "",
        _ber_pivot(aggregates["ber_vs_snr"], "iq"),
        "",
        _ber_pivot(aggregates["ber_vs_snr"], "wav"),
        "",
        "A dash means no BER could be scored for that cell (demodulation did not run, or produced "
        "no bits). The per-cell counts are in `results.json` under `aggregates.ber_vs_snr`.",
        "",
        "## IQ vs WAV",
        "",
        _table(
            ["Metric"] + list(aggregates["by_format"].keys()),
            [
                [label] + [_fmt(stats[key], percent=pct) for stats in aggregates["by_format"].values()]
                for label, key, pct in [
                    ("Records", "count", False),
                    ("Ingestion success", "ingestion_success_rate", True),
                    ("End-to-end success", "end_to_end_success_rate", True),
                    ("CNN top-1 accuracy", "cnn_accuracy", True),
                    ("Fusion accuracy", "fusion_accuracy", True),
                    ("Median SNR error (dB)", "median_snr_error_db", False),
                    ("Median symbol-rate rel err", "median_symbol_rate_rel_error", False),
                    ("Median BER (strict)", "median_ber_strict", False),
                ]
            ],
        ),
        "",
        _iq_wav_disagreement(records),
        "",
        "## Downstream stages (V1 truth: no interleaving, no FEC, no sync words)",
        "",
        "These stages only run when demodulation produced bits, so the denominator is small.",
        "",
        _table(
            ["Metric", "Value", "Scored (n)"],
            [
                ["Interleaver correctly reported as 'none'",
                 _fmt(overall["interleaver_none_accuracy"], percent=True),
                 str(_scored_count(records, "interleaver_correct"))],
                ["FEC correctly reported as 'none'",
                 _fmt(overall["fec_none_accuracy"], percent=True),
                 str(_scored_count(records, "fec_correct"))],
                ["Sync-word false-positive rate",
                 _fmt(overall["sync_false_positive_rate"], percent=True),
                 str(_scored_count(records, "sync_false_positive"))],
            ],
        ),
        "",
        "## Failure reasons",
        "",
        _counter_table("End-to-end status", aggregates["failure_reasons"]["end_to_end_status"]),
        "",
        _counter_table("BER unavailable reason", aggregates["failure_reasons"]["ber_reason"]),
        "",
        _counter_table("Demodulation messages", aggregates["failure_reasons"]["demod_message"]),
        "",
        _counter_table("Ingestion errors", aggregates["failure_reasons"]["ingestion_error"]),
        "",
        _counter_table("Pipeline errors", aggregates["failure_reasons"]["pipeline_error"]),
        "",
        "## Unavailable / unsupported metrics",
        "",
        "These are recorded as unavailable rather than estimated:",
        "",
    ]
    lines.extend(f"- **{name}** - {reason}" for name, reason in UNAVAILABLE_METRICS.items())
    lines.append("")
    return "\n".join(lines)


def _scored_count(records: Sequence[Record], field: str) -> int:
    return sum(1 for record in records if record.get(field) is not None)


def _ber_pivot(ber_rows: Sequence[dict[str, Any]], file_format: str) -> str:
    rows = [row for row in ber_rows if row["file_format"] == file_format]
    if not rows:
        return f"### BER (strict) by modulation and SNR - {file_format}\n\n_no records_"
    snrs = sorted({row["target_snr_db"] for row in rows}, key=lambda value: (value is None, value), reverse=True)
    modulations = sorted({row["modulation"] for row in rows})
    lookup = {(row["modulation"], row["target_snr_db"]): row for row in rows}
    body = []
    for modulation in modulations:
        cells = []
        for snr in snrs:
            row = lookup.get((modulation, snr))
            cells.append("-" if row is None or row["median_ber_strict"] is None else _fmt(row["median_ber_strict"]))
        body.append([modulation] + cells)
    header = ["Modulation"] + [f"{_fmt(snr)} dB" for snr in snrs]
    return f"### BER (strict) by modulation and SNR - {file_format}\n\n" + _table(header, body)


def _confusion_table(matrix: dict[str, dict[str, int]]) -> str:
    if not matrix:
        return "_no scored records_"
    predictions = sorted({label for row in matrix.values() for label in row})
    rows = [[truth] + [str(matrix.get(truth, {}).get(label, 0)) for label in predictions] for truth in sorted(matrix)]
    return _table(["truth \\ predicted"] + predictions, rows)


def _counter_table(title: str, counts: dict[Any, int]) -> str:
    if not counts:
        return f"**{title}:** none"
    rows = [[str(name), str(count)] for name, count in sorted(counts.items(), key=lambda item: -item[1])]
    return f"**{title}**\n\n" + _table(["Value", "Count"], rows)


def _iq_wav_disagreement(records: Sequence[Record]) -> str:
    by_capture: dict[str, dict[str, Record]] = {}
    for record in records:
        by_capture.setdefault(record["capture_id"], {})[record["file_format"]] = record
    paired = [item for item in by_capture.values() if "iq" in item and "wav" in item]
    if not paired:
        return "_No capture had both an IQ and a WAV record._"
    label_disagreements = sum(1 for item in paired if item["iq"]["cnn_label"] != item["wav"]["cnn_label"])
    status_disagreements = sum(
        1 for item in paired if item["iq"]["end_to_end_status"] != item["wav"]["end_to_end_status"]
    )
    return (
        f"Paired captures: {len(paired)}. CNN label differs between IQ and WAV on "
        f"{label_disagreements} of them; end-to-end status differs on {status_disagreements}."
    )


def write_outputs(
    records: Sequence[Record],
    output_dir: Path,
    *,
    dataset_root: Path,
    capture_count: int,
) -> dict[str, Path]:
    """Write results.csv, results.json and summary.md; return their paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate(records)

    csv_path = output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: ("" if record.get(key) is None else record.get(key)) for key in RECORD_FIELDS})

    json_path = output / "results.json"
    payload = {
        "schema": RESULT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(dataset_root), "capture_count": capture_count, "record_count": len(records)},
        "unavailable_metrics": UNAVAILABLE_METRICS,
        "aggregates": aggregates,
        "records": list(records),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    summary_path = output / "summary.md"
    summary_path.write_text(render_summary(records, aggregates, dataset_root=dataset_root), encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "summary": summary_path}
