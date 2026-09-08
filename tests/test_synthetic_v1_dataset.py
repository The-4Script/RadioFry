"""Ground-truth completeness and reproducibility tests for Synthetic Dataset V1."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from radiofry.ingestion.iq_parser import IQFormat
from radiofry.pipeline import load_capture
from radiofry.synthetic_gen.v1 import SampleSpec
from radiofry.synthetic_gen.v1.cli import main
from radiofry.synthetic_gen.v1.generator import (
    DatasetSpec,
    capture_id_for,
    generate_dataset,
    generate_sample,
    load_ground_truth,
)


def _sample_spec(**overrides) -> SampleSpec:
    params = {
        "modulation": "QPSK",
        "num_symbols": 128,
        "samples_per_symbol": 8,
        "sample_rate_hz": 200_000.0,
        "snr_db": 20.0,
        "seed": 77,
    }
    params.update(overrides)
    return SampleSpec(**params)


def _dataset_spec(output_dir: Path, **overrides) -> DatasetSpec:
    params = {
        "output_dir": output_dir,
        "modulations": ("BPSK", "16QAM"),
        "snr_sweep_db": (20.0, 0.0),
        "captures_per_condition": 1,
        "num_symbols": 64,
        "samples_per_symbol": 8,
        "sample_rate_hz": 200_000.0,
        "seed": 2026,
    }
    params.update(overrides)
    return DatasetSpec(**params)


# --- single capture -------------------------------------------------------------


def test_generate_sample_writes_capture_ground_truth_and_bit_files(tmp_path: Path) -> None:
    truth = generate_sample(_sample_spec(), tmp_path, "cap000")

    assert (tmp_path / "cap000.iq").exists()
    assert (tmp_path / "cap000.wav").exists()
    assert (tmp_path / "cap000.json").exists()
    assert (tmp_path / truth["bits"]["source_bits_file"]).exists()
    assert (tmp_path / truth["bits"]["transmitted_bits_file"]).exists()


def test_ground_truth_records_every_required_field(tmp_path: Path) -> None:
    truth = generate_sample(_sample_spec(), tmp_path, "cap000")

    assert truth["modulation"]["name"] == "QPSK"
    assert truth["modulation"]["order"] == 4
    assert truth["modulation"]["bits_per_symbol"] == 2
    assert truth["modulation"]["radiofry_label"] == "QPSK"
    assert truth["modulation"]["bit_mapping"] == "natural_binary_msb_first"
    assert truth["signal"]["sample_rate_hz"] == 200_000.0
    assert truth["signal"]["symbol_rate_hz"] == 25_000.0
    assert truth["signal"]["samples_per_symbol"] == 8
    assert truth["signal"]["num_symbols"] == 128
    assert truth["signal"]["num_samples"] == 1_024
    assert truth["noise"]["target_snr_db"] == 20.0
    assert truth["noise"]["noise_type"] == "awgn"
    assert truth["bits"]["num_source_bits"] == 256
    assert truth["bits"]["num_transmitted_bits"] == 256
    assert truth["seeds"]["seed"] == 77
    assert truth["seeds"]["bits_seed"] == 77
    assert {entry["file_format"] for entry in truth["files"]} == {"iq", "wav"}
    assert truth["schema"] == "radiofry.synthetic.v1"


def test_ground_truth_records_the_realized_snr_close_to_the_target(tmp_path: Path) -> None:
    truth = generate_sample(_sample_spec(num_symbols=8192, snr_db=10.0), tmp_path, "cap000")

    assert abs(truth["noise"]["realized_snr_db"] - 10.0) < 0.2
    assert truth["noise"]["snr_definition"].startswith("total_band")


def test_ground_truth_records_derived_es_n0_and_eb_n0(tmp_path: Path) -> None:
    # 8 samples/symbol spreads the noise over 8x the symbol bandwidth (+9.03 dB),
    # and QPSK carries 2 bits per symbol (-3.01 dB).
    truth = generate_sample(_sample_spec(snr_db=10.0), tmp_path, "cap000")

    assert truth["noise"]["es_n0_db"] == pytest.approx(19.03, abs=0.01)
    assert truth["noise"]["eb_n0_db"] == pytest.approx(16.02, abs=0.01)


def test_derived_snr_fields_are_absent_for_a_noiseless_capture(tmp_path: Path) -> None:
    truth = generate_sample(_sample_spec(snr_db=None), tmp_path, "cap000")

    assert truth["noise"]["noise_type"] == "none"
    assert truth["noise"]["es_n0_db"] is None
    assert truth["noise"]["eb_n0_db"] is None


def test_v1_impairments_are_recorded_as_disabled(tmp_path: Path) -> None:
    truth = generate_sample(_sample_spec(), tmp_path, "cap000")

    impairments = truth["impairments"]
    assert impairments["carrier_frequency_offset_hz"] == 0.0
    assert impairments["timing_offset_samples"] == 0.0
    assert impairments["phase_offset_rad"] == 0.0
    assert impairments["fading"] == "disabled"
    assert impairments["multipath"] == "disabled"
    assert impairments["interference"] == "disabled"
    assert impairments["fec"] == "none"
    assert impairments["interleaving"] == "none"


def test_source_and_transmitted_bits_are_identical_when_no_coding_is_applied(tmp_path: Path) -> None:
    generate_sample(_sample_spec(), tmp_path, "cap000")

    loaded = load_ground_truth(tmp_path / "cap000.json")

    np.testing.assert_array_equal(loaded["source_bits"], loaded["transmitted_bits"])
    assert loaded["metadata"]["bits"]["source_equals_transmitted"] is True


def test_load_ground_truth_returns_bits_matching_the_recorded_digest(tmp_path: Path) -> None:
    import hashlib

    generate_sample(_sample_spec(), tmp_path, "cap000")
    loaded = load_ground_truth(tmp_path / "cap000.json")

    digest = hashlib.sha256(loaded["source_bits"].tobytes()).hexdigest()
    assert digest == loaded["metadata"]["bits"]["source_bits_sha256"]
    assert loaded["source_bits"].size == 256


def test_generated_capture_is_readable_by_the_existing_radiofry_loader(tmp_path: Path) -> None:
    truth = generate_sample(_sample_spec(), tmp_path, "cap000")
    iq_entry = next(entry for entry in truth["files"] if entry["file_format"] == "iq")

    signal = load_capture(
        tmp_path / "cap000.iq",
        sample_rate=truth["signal"]["sample_rate_hz"],
        iq_format=IQFormat(iq_entry["dtype"], iq_entry["byte_order"]),
    )
    wav_signal = load_capture(tmp_path / "cap000.wav")

    assert signal.iq.size == truth["signal"]["num_samples"]
    assert signal.sample_rate == 200_000.0
    assert wav_signal.iq.size == truth["signal"]["num_samples"]
    assert wav_signal.sample_rate == 200_000.0


def test_regenerating_the_same_spec_produces_identical_capture_bytes(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    generate_sample(_sample_spec(), first_dir, "cap000")
    generate_sample(_sample_spec(), second_dir, "cap000")

    assert (first_dir / "cap000.iq").read_bytes() == (second_dir / "cap000.iq").read_bytes()
    assert (first_dir / "cap000.wav").read_bytes() == (second_dir / "cap000.wav").read_bytes()


def test_changing_the_seed_changes_the_capture(tmp_path: Path) -> None:
    generate_sample(_sample_spec(seed=1), tmp_path / "a", "cap000")
    generate_sample(_sample_spec(seed=2), tmp_path / "b", "cap000")

    assert (tmp_path / "a" / "cap000.iq").read_bytes() != (tmp_path / "b" / "cap000.iq").read_bytes()


def test_only_requested_formats_are_written(tmp_path: Path) -> None:
    generate_sample(_sample_spec(), tmp_path, "cap000", formats=("iq",))

    assert (tmp_path / "cap000.iq").exists()
    assert not (tmp_path / "cap000.wav").exists()


# --- dataset sweep --------------------------------------------------------------


def test_capture_id_encodes_modulation_snr_and_replicate() -> None:
    assert capture_id_for("QPSK", 20.0, 0) == "QPSK_snr20dB_r000"
    assert capture_id_for("16QAM", 0.0, 3) == "16QAM_snr0dB_r003"
    assert capture_id_for("BPSK", None, 1) == "BPSK_noiseless_r001"


def test_generate_dataset_covers_every_modulation_and_snr_combination(tmp_path: Path) -> None:
    manifest_path = generate_dataset(_dataset_spec(tmp_path, captures_per_condition=2))

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2 * 2 * 2
    assert {row["modulation"] for row in rows} == {"BPSK", "16QAM"}
    assert {row["target_snr_db"] for row in rows} == {"20.0", "0.0"}
    assert (tmp_path / "dataset.json").exists()
    assert all((tmp_path / "captures" / f"{row['capture_id']}.iq").exists() for row in rows)


def test_dataset_manifest_rows_point_at_readable_ground_truth(tmp_path: Path) -> None:
    manifest_path = generate_dataset(_dataset_spec(tmp_path))

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        row = next(iter(csv.DictReader(handle)))
    loaded = load_ground_truth(tmp_path / row["ground_truth_file"])

    assert loaded["metadata"]["capture_id"] == row["capture_id"]
    assert loaded["source_bits"].size == int(row["num_symbols"]) * int(row["bits_per_symbol"])


def test_snr_sweep_reuses_one_payload_so_only_the_noise_changes(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path, modulations=("BPSK",), snr_sweep_db=(20.0, 0.0)))

    high = load_ground_truth(tmp_path / "captures" / "BPSK_snr20dB_r000.json")
    low = load_ground_truth(tmp_path / "captures" / "BPSK_snr0dB_r000.json")

    np.testing.assert_array_equal(high["source_bits"], low["source_bits"])
    assert high["metadata"]["seeds"]["seed"] != low["metadata"]["seeds"]["seed"]


def test_dataset_generation_is_reproducible(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path / "run1"))
    generate_dataset(_dataset_spec(tmp_path / "run2"))

    first = sorted((tmp_path / "run1" / "captures").glob("*.iq"))
    second = sorted((tmp_path / "run2" / "captures").glob("*.iq"))

    assert [path.name for path in first] == [path.name for path in second]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(first, second))


def test_dataset_json_describes_the_generation_configuration(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path))

    summary = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))

    assert summary["schema"] == "radiofry.synthetic.v1.dataset"
    assert summary["capture_count"] == 4
    assert summary["config"]["seed"] == 2026
    assert summary["config"]["modulations"] == ["BPSK", "16QAM"]
    assert summary["config"]["snr_sweep_db"] == [20.0, 0.0]


def test_fsk_captures_are_swept_over_both_modulation_indices(tmp_path: Path) -> None:
    manifest_path = generate_dataset(
        _dataset_spec(tmp_path, modulations=("BFSK", "BPSK"), snr_sweep_db=(20.0,))
    )

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fsk = [r for r in rows if r["modulation"] == "BFSK"]
    psk = [r for r in rows if r["modulation"] == "BPSK"]
    assert {r["fsk_modulation_index"] for r in fsk} == {"0.5", "1.0"}
    assert len(fsk) == 2
    assert len(psk) == 1  # non-FSK modulations are not swept
    assert psk[0]["fsk_modulation_index"] == ""


def test_fsk_capture_ids_record_the_modulation_index(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path, modulations=("BFSK",), snr_sweep_db=(20.0,)))

    captures = {p.stem for p in (tmp_path / "captures").glob("*.iq")}

    assert captures == {"BFSK_h0.5_snr20dB_r000", "BFSK_h1.0_snr20dB_r000"}


def test_non_fsk_capture_ids_are_unchanged(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path, modulations=("BPSK",), snr_sweep_db=(20.0,)))

    assert (tmp_path / "captures" / "BPSK_snr20dB_r000.iq").exists()


def test_ground_truth_records_the_deviation_and_index_and_hardness_flag(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path, modulations=("BFSK",), snr_sweep_db=(20.0,)))

    standard = load_ground_truth(tmp_path / "captures" / "BFSK_h0.5_snr20dB_r000.json")["metadata"]
    hard = load_ground_truth(tmp_path / "captures" / "BFSK_h1.0_snr20dB_r000.json")["metadata"]

    assert standard["signal"]["fsk_deviation_hz"] == 6_250.0
    assert standard["signal"]["fsk_modulation_index"] == 0.5
    assert standard["known_hard"] is False
    assert hard["signal"]["fsk_deviation_hz"] == 12_500.0
    assert hard["signal"]["fsk_modulation_index"] == 1.0
    assert hard["known_hard"] is True
    assert "modulation index" in hard["known_hard_reason"]


def test_non_fsk_ground_truth_is_not_marked_hard_and_has_no_index(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path, modulations=("BPSK",), snr_sweep_db=(20.0,)))

    meta = load_ground_truth(tmp_path / "captures" / "BPSK_snr20dB_r000.json")["metadata"]

    assert meta["signal"]["fsk_modulation_index"] is None
    assert meta["known_hard"] is False


def test_the_two_fsk_arms_share_a_payload_so_only_the_index_differs(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path, modulations=("BFSK",), snr_sweep_db=(20.0,)))

    standard = load_ground_truth(tmp_path / "captures" / "BFSK_h0.5_snr20dB_r000.json")
    hard = load_ground_truth(tmp_path / "captures" / "BFSK_h1.0_snr20dB_r000.json")

    np.testing.assert_array_equal(standard["source_bits"], hard["source_bits"])
    assert standard["metadata"]["seeds"]["seed"] == hard["metadata"]["seeds"]["seed"]


def test_the_fsk_sweep_is_configurable(tmp_path: Path) -> None:
    generate_dataset(
        _dataset_spec(tmp_path, modulations=("BFSK",), snr_sweep_db=(20.0,),
                      fsk_modulation_indices=(0.25,))
    )

    captures = {p.stem for p in (tmp_path / "captures").glob("*.iq")}

    assert captures == {"BFSK_h0.25_snr20dB_r000"}


def test_fsk_sweep_generation_is_reproducible(tmp_path: Path) -> None:
    generate_dataset(_dataset_spec(tmp_path / "a", modulations=("BFSK",), snr_sweep_db=(20.0,)))
    generate_dataset(_dataset_spec(tmp_path / "b", modulations=("BFSK",), snr_sweep_db=(20.0,)))

    first = sorted((tmp_path / "a" / "captures").glob("*.iq"))
    second = sorted((tmp_path / "b" / "captures").glob("*.iq"))
    assert [p.name for p in first] == [p.name for p in second]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(first, second))


def test_dataset_spec_rejects_an_unsupported_modulation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="modulation"):
        _dataset_spec(tmp_path, modulations=("FM",))


# --- command line ---------------------------------------------------------------


def test_cli_generates_a_dataset(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--output",
            str(tmp_path),
            "--modulations",
            "BPSK",
            "QPSK",
            "--snr-db",
            "10",
            "--num-symbols",
            "32",
            "--captures-per-condition",
            "1",
        ]
    )

    assert exit_code == 0
    with (tmp_path / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["modulation"] for row in rows} == {"BPSK", "QPSK"}
