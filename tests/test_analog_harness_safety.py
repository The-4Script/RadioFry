"""AM-DSB capture through the existing evaluation harness (BANK.md Entry 021).

Validates the analog safety path only. Production analog demodulation is NOT
implemented, and this suite makes no claim about demodulation quality - it proves the
harness survives an analog capture and refuses to publish a bit-error rate for it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from radiofry.evaluation.harness import UNAVAILABLE_METRICS, evaluate_capture
from radiofry.evaluation.metrics import expected_family_for
from radiofry.synthetic_gen.v1 import load_ground_truth
from radiofry.synthetic_gen.v1.analog import AnalogSampleSpec, generate_analog_sample

V1_CAPTURES = Path("data/synthetic_v1/captures")


@pytest.fixture
def analog_dataset(tmp_path: Path):
    """A one-capture analog 'dataset' shaped the way the harness expects."""
    captures = tmp_path / "captures"
    truth = generate_analog_sample(
        AnalogSampleSpec(snr_db=20.0, num_samples=8_192, seed=5), captures, "amdsb000"
    )
    row = {
        "capture_id": "amdsb000",
        "ground_truth_file": "captures/amdsb000.json",
        "iq_file": "captures/amdsb000.iq",
        "wav_file": "captures/amdsb000.wav",
    }
    return tmp_path, row, truth


# --- the first blocker: ground-truth loading ---------------------------------------


def test_load_ground_truth_handles_a_null_bits_block(analog_dataset) -> None:
    root, _, _ = analog_dataset

    loaded = load_ground_truth(root / "captures" / "amdsb000.json")

    assert loaded["source_bits"] is None
    assert loaded["transmitted_bits"] is None
    assert loaded["metadata"]["capture_kind"] == "analog"


@pytest.mark.skipif(
    not (V1_CAPTURES / "QPSK_snr20dB_r000.json").exists(),
    reason="frozen V1 dataset not present",
)
def test_load_ground_truth_still_returns_arrays_for_digital() -> None:
    loaded = load_ground_truth(V1_CAPTURES / "QPSK_snr20dB_r000.json")

    assert isinstance(loaded["source_bits"], np.ndarray)
    assert loaded["source_bits"].size == 8_192


# --- the harness must survive an analog capture -------------------------------------


@pytest.mark.parametrize("file_format", ["iq", "wav"])
def test_harness_does_not_crash_on_an_analog_capture(analog_dataset, file_format: str) -> None:
    root, row, _ = analog_dataset

    record = evaluate_capture(root, row, file_format)

    assert record["ingestion_ok"] is True
    assert record["pipeline_ok"] is True
    assert record["ingestion_error"] == ""
    assert record["pipeline_error"] == ""


def test_analog_family_is_resolved_to_the_analog_like_vocabulary(analog_dataset) -> None:
    root, row, _ = analog_dataset

    record = evaluate_capture(root, row, "iq")

    assert record["expected_family"] == "analog-like"
    assert expected_family_for("analog") == "analog-like"


def test_analog_capture_identity_is_recorded(analog_dataset) -> None:
    root, row, _ = analog_dataset

    record = evaluate_capture(root, row, "iq")

    assert record["capture_id"] == "amdsb000"
    assert record["modulation"] == "AM-DSB"
    assert record["modulation_family"] == "analog"
    assert record["expected_cnn_label"] == "AM-DSB"


# --- BER must be unavailable, never fabricated ---------------------------------------


def test_ber_is_explicitly_unavailable_for_analog(analog_dataset) -> None:
    root, row, _ = analog_dataset

    record = evaluate_capture(root, row, "iq")

    assert record["ber_status"] == "unavailable"
    assert record["ber_reason"] == "analog_no_transmitted_bits"
    assert record["ber_strict"] is None
    assert record["ber_aligned"] is None


def test_no_fake_ber_even_when_dispatch_synthesises_bits(analog_dataset) -> None:
    # dispatch turns the analog waveform into bits with `analog > median(analog)`.
    # Those bits must never be scored against anything.
    root, row, _ = analog_dataset

    record = evaluate_capture(root, row, "iq")

    assert record["ber_strict"] is None, "a threshold-derived BER was published"
    assert record["compared_bits"] == 0
    assert record["expected_bits"] is None


def test_dispatch_really_does_synthesise_analog_bits() -> None:
    # Confirms Entry 018 measured fact 3 still holds, so the guard below is guarding
    # something real rather than a hypothetical.
    from radiofry.contracts import UnifiedSignalContainer
    from radiofry.decoding.demodulators.dispatch import demodulate_capture
    from radiofry.dsp.parameter_estimation import ParameterEstimate
    from radiofry.synthetic_gen.v1.analog import AnalogSampleSpec, generate_message, modulate_am_dsb

    spec = AnalogSampleSpec(num_samples=8_192, seed=5)
    signal = UnifiedSignalContainer(modulate_am_dsb(generate_message(spec)[0], spec), 200_000.0)

    result = demodulate_capture(signal, "AM-DSB", ParameterEstimate(None, None, 25_000.0))

    assert result.available
    assert result.result.bits.size > 0  # median-threshold bits, physically meaningless


def test_the_guard_refuses_those_bits_even_when_they_are_present() -> None:
    # Exercises the production guard directly: the harness scorer must return
    # "unavailable" when the capture has no truth bits, no matter what dispatch handed
    # back. The end-to-end record above does not reach this branch, because the CNN
    # currently labels the capture PAM4, which has no dispatch route.
    from radiofry.evaluation.harness import _score_report

    record: dict = {"truth_symbol_rate_hz": None, "truth_carrier_hz": 0.0, "truth_snr_db": 20.0,
                    "expected_family": "analog-like", "expected_cnn_label": "AM-DSB",
                    "truth_interleaver": "none", "truth_fec": "none"}
    report = {"stages": {"demodulation": {"available": True,
                                          "result": {"bits": [1, 0, 1, 1, 0], "modulation": "AM-DSB"}}}}

    _score_report(record, report, np.array([], dtype=np.uint8), has_bits=False)

    assert record["ber_status"] == "unavailable"
    assert record["ber_reason"] == "analog_no_transmitted_bits"
    assert record["ber_strict"] is None
    assert record["compared_bits"] == 0


def test_analog_unavailability_is_documented_in_the_metric_registry() -> None:
    assert "bit_error_rate_analog" in UNAVAILABLE_METRICS
    assert "analog" in UNAVAILABLE_METRICS["bit_error_rate_analog"].lower()


# --- bit-derived and symbol-rate fields stay unavailable ------------------------------


def test_bit_derived_ground_truth_stays_null_through_the_harness(analog_dataset) -> None:
    root, row, truth = analog_dataset

    record = evaluate_capture(root, row, "iq")

    assert truth["noise"]["eb_n0_db"] is None
    assert truth["noise"]["es_n0_db"] is None
    assert record["es_n0_db"] is None
    assert record["bits_per_symbol"] is None
    assert record["order"] is None


def test_no_symbol_rate_requirement_is_introduced(analog_dataset) -> None:
    root, row, _ = analog_dataset

    record = evaluate_capture(root, row, "iq")

    assert record["truth_symbol_rate_hz"] is None
    assert record["symbol_rate_error_hz"] is None
    assert record["symbol_rate_rel_error"] is None


def test_ground_truth_file_is_not_mutated_by_the_harness(analog_dataset) -> None:
    root, row, _ = analog_dataset
    path = root / "captures" / "amdsb000.json"
    before = path.read_bytes()

    evaluate_capture(root, row, "iq")

    assert path.read_bytes() == before
    assert json.loads(before)["capture_kind"] == "analog"


# --- digital behaviour must be unchanged ----------------------------------------------


@pytest.mark.skipif(
    not (V1_CAPTURES / "QPSK_snr20dB_r000.json").exists(),
    reason="frozen V1 dataset not present",
)
def test_digital_capture_still_scores_a_real_ber() -> None:
    row = {
        "capture_id": "QPSK_snr20dB_r000",
        "ground_truth_file": "captures/QPSK_snr20dB_r000.json",
        "iq_file": "captures/QPSK_snr20dB_r000.iq",
        "wav_file": "captures/QPSK_snr20dB_r000.wav",
    }

    record = evaluate_capture(V1_CAPTURES.parent, row, "iq")

    assert record["expected_family"] == "PSK-like"
    assert record["expected_bits"] == 8_192
    assert record["ber_status"] == "ok"
    assert record["ber_strict"] == 0.0
    assert record["es_n0_db"] is not None
