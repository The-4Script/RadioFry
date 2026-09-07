"""End-to-end tests for the Synthetic V1 evaluation harness.

These verify that the harness records what the pipeline actually produced -
including honest "unavailable" markers - not that RadioFry performs well.
"""

import csv
import json
from pathlib import Path

import pytest

from radiofry.evaluation.harness import evaluate_capture, load_manifest, run_evaluation
from radiofry.synthetic_gen.v1 import DatasetSpec, generate_dataset


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("v1_dataset")
    generate_dataset(
        DatasetSpec(
            output_dir=root,
            modulations=("BPSK",),
            snr_sweep_db=(20.0,),
            num_symbols=128,
            samples_per_symbol=8,
            sample_rate_hz=200_000.0,
            seed=5,
        )
    )
    return root


@pytest.fixture(scope="module")
def iq_record(tiny_dataset: Path) -> dict:
    row = load_manifest(tiny_dataset)[0]
    return evaluate_capture(tiny_dataset, row, "iq")


def test_evaluate_capture_records_the_capture_identity(iq_record: dict) -> None:
    assert iq_record["capture_id"] == "BPSK_snr20dB_r000"
    assert iq_record["modulation"] == "BPSK"
    assert iq_record["file_format"] == "iq"
    assert iq_record["target_snr_db"] == 20.0
    assert iq_record["expected_family"] == "PSK-like"
    assert iq_record["expected_cnn_label"] == "BPSK"


def test_evaluate_capture_reports_ingestion_and_pipeline_status(iq_record: dict) -> None:
    assert iq_record["ingestion_ok"] is True
    assert iq_record["pipeline_ok"] is True
    assert iq_record["ingestion_error"] == ""
    assert iq_record["end_to_end_status"] in {
        "bits_recovered",
        "demodulation_failed",
        "no_modulation_decision",
    }


def test_evaluate_capture_records_parameter_estimates_against_truth(iq_record: dict) -> None:
    assert iq_record["truth_symbol_rate_hz"] == 25_000.0
    assert iq_record["truth_carrier_hz"] == 0.0
    assert iq_record["truth_snr_db"] == 20.0
    for field in ("est_symbol_rate_hz", "est_snr_db", "est_carrier_hz", "est_bandwidth_hz"):
        assert field in iq_record
    for field in ("symbol_rate_error_hz", "snr_error_db", "carrier_error_hz"):
        assert field in iq_record


def test_bandwidth_error_is_reported_as_unavailable_ground_truth(iq_record: dict) -> None:
    assert iq_record["bandwidth_status"] == "unavailable_no_ground_truth"
    assert "bandwidth_error_hz" not in iq_record


def test_evaluate_capture_records_every_classification_stage(iq_record: dict) -> None:
    for field in (
        "classical_family",
        "classical_family_correct",
        "cnn_label",
        "cnn_confidence",
        "cnn_correct",
        "cnn_topk_correct",
        "fusion_label",
        "fusion_trust_score",
        "fusion_review_recommended",
        "fusion_correct",
    ):
        assert field in iq_record


def test_evaluate_capture_records_bit_error_scoring_with_a_status(iq_record: dict) -> None:
    assert iq_record["ber_status"] in {"ok", "unavailable"}
    assert iq_record["expected_bits"] == 128
    if iq_record["ber_status"] == "ok":
        assert 0.0 <= iq_record["ber_strict"] <= 1.0
        assert 0.0 <= iq_record["ber_aligned"] <= 1.0
    else:
        assert iq_record["ber_strict"] is None
        assert iq_record["ber_reason"]


def test_evaluate_capture_records_downstream_stages_whose_v1_truth_is_none(iq_record: dict) -> None:
    assert iq_record["truth_interleaver"] == "none"
    assert iq_record["truth_fec"] == "none"
    for field in ("interleaver_label", "fec_label", "fec_success", "sync_pattern", "sync_false_positive"):
        assert field in iq_record


def test_missing_capture_file_is_recorded_as_an_ingestion_failure(tiny_dataset: Path) -> None:
    row = dict(load_manifest(tiny_dataset)[0])
    row["iq_file"] = "captures/does_not_exist.iq"

    record = evaluate_capture(tiny_dataset, row, "iq")

    assert record["ingestion_ok"] is False
    assert record["ingestion_error"]
    assert record["pipeline_ok"] is False
    assert record["end_to_end_status"] == "ingestion_failed"
    assert record["ber_status"] == "unavailable"


def test_wav_captures_are_evaluated_through_the_wav_ingestion_path(tiny_dataset: Path) -> None:
    row = load_manifest(tiny_dataset)[0]

    record = evaluate_capture(tiny_dataset, row, "wav")

    assert record["file_format"] == "wav"
    assert record["ingestion_ok"] is True
    assert record["sample_rate_source"] == "wav_header"


def test_iq_captures_take_the_sample_rate_from_ground_truth(iq_record: dict) -> None:
    assert iq_record["sample_rate_source"] == "ground_truth"


# --- full run -------------------------------------------------------------------


def test_run_evaluation_writes_machine_readable_and_human_readable_outputs(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    outputs = run_evaluation(tiny_dataset, tmp_path / "results")

    assert outputs["csv"].exists()
    assert outputs["json"].exists()
    assert outputs["summary"].exists()

    with outputs["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # one capture x two formats
    assert {row["file_format"] for row in rows} == {"iq", "wav"}


def test_run_evaluation_json_carries_records_aggregates_and_unavailable_metrics(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    outputs = run_evaluation(tiny_dataset, tmp_path / "results")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))

    assert payload["schema"] == "radiofry.evaluation.v1"
    assert len(payload["records"]) == 2
    assert "by_modulation" in payload["aggregates"]
    assert "by_snr" in payload["aggregates"]
    assert "by_format" in payload["aggregates"]
    assert "occupied_bandwidth" in payload["unavailable_metrics"]
    assert payload["dataset"]["capture_count"] == 1


def test_run_evaluation_summary_contains_the_required_sections(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    outputs = run_evaluation(tiny_dataset, tmp_path / "results")

    summary = outputs["summary"].read_text(encoding="utf-8")

    for heading in (
        "Ingestion and pipeline outcomes",
        "Modulation classification",
        "Parameter estimation",
        "BER vs SNR",
        "IQ vs WAV",
        "Unavailable / unsupported metrics",
        "Failure reasons",
    ):
        assert heading in summary


def test_summary_pivots_ber_by_modulation_and_snr_for_each_format(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    outputs = run_evaluation(tiny_dataset, tmp_path / "results")

    summary = outputs["summary"].read_text(encoding="utf-8")

    assert "### BER (strict) by modulation and SNR - iq" in summary
    assert "### BER (strict) by modulation and SNR - wav" in summary


def test_summary_reports_a_denominator_for_every_rate(tiny_dataset: Path, tmp_path: Path) -> None:
    # A rate like "0.0%" is misleading without the number of records it was computed over.
    outputs = run_evaluation(tiny_dataset, tmp_path / "results")

    summary = outputs["summary"].read_text(encoding="utf-8")

    assert "Scored (n)" in summary


def test_cli_evaluates_a_dataset(tiny_dataset: Path, tmp_path: Path) -> None:
    from radiofry.evaluation.cli import main

    exit_code = main(["--dataset", str(tiny_dataset), "--output", str(tmp_path / "cli"), "--formats", "iq"])

    assert exit_code == 0
    assert (tmp_path / "cli" / "results.csv").exists()
    assert (tmp_path / "cli" / "summary.md").exists()


def test_run_evaluation_can_restrict_the_formats_it_scores(tiny_dataset: Path, tmp_path: Path) -> None:
    outputs = run_evaluation(tiny_dataset, tmp_path / "results", formats=("iq",))

    with outputs["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["file_format"] for row in rows} == {"iq"}
