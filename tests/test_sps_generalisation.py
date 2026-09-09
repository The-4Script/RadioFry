"""SPS-augmented training and the V3 production checkpoint (BANK.md Entry 040).

Entry 039 measured the production 11-class RadioML checkpoint at 35.4% fused accuracy on
RadioFry's own synthetic distribution, collapsing to 10.4% at 32 samples per symbol.
Entry 040 traced that to two causes - domain mismatch and single-SPS training - and
rejected the third hypothesis (that the 128-sample window is too short).
"""

import json
import numpy as np
import pytest
import torch

from radiofry import pipeline
from radiofry.contracts import UnifiedSignalContainer
from radiofry.dsp.preprocessing import preprocess
from radiofry.models.modulation_inference import metrics_path, predict_modulation
from radiofry.synthetic_gen.v1.channel import add_awgn
from radiofry.synthetic_gen.v1.config import SampleSpec
from radiofry.synthetic_gen.v1.modulation import modulate
from radiofry.training.train_v2_synthetic import (
    SAMPLES_PER_SYMBOL_SWEEP, build_capture_specs)

FS, N = 200_000.0, 8_192
CHECKPOINT = pipeline.DEFAULT_MODULATION_MODEL


def _capture(modulation, sps, snr_db=20.0, seed=7_000_001):
    spec = SampleSpec(modulation=modulation, num_symbols=N // sps,
                      samples_per_symbol=sps, sample_rate_hz=FS, snr_db=snr_db, seed=seed)
    bits = np.random.default_rng([seed, 1]).integers(0, 2, spec.num_bits, dtype=np.uint8)
    iq, _ = add_awgn(modulate(bits, spec), snr_db, np.random.default_rng([seed, 2]))
    return preprocess(UnifiedSignalContainer(iq, FS))


# --- the training change --------------------------------------------------------------


def test_training_sweeps_samples_per_symbol() -> None:
    assert SAMPLES_PER_SYMBOL_SWEEP == (4, 8, 16, 32)


def test_capture_specs_cover_every_oversampling_factor() -> None:
    specs = build_capture_specs(replicates=1, seed_base=0)
    seen = {s.samples_per_symbol for s in specs}

    assert seen == set(SAMPLES_PER_SYMBOL_SWEEP)


def test_the_sweep_is_balanced_across_oversampling() -> None:
    specs = build_capture_specs(replicates=1, seed_base=0)
    counts = {s: sum(1 for x in specs if x.samples_per_symbol == s)
              for s in SAMPLES_PER_SYMBOL_SWEEP}

    assert len(set(counts.values())) == 1, counts


def test_capture_ids_stay_unique_across_the_sweep() -> None:
    specs = build_capture_specs(replicates=2, seed_base=0)
    ids = [s.capture_id for s in specs]

    assert len(ids) == len(set(ids))
    assert any("sps32" in i for i in ids)


# --- the production checkpoint ------------------------------------------------------------


def test_the_default_checkpoint_is_the_sps_augmented_one() -> None:
    assert CHECKPOINT == "models_saved/modulation_cnn_v3_spsaug.pt"


def test_the_previous_baseline_checkpoint_is_retained_for_comparison() -> None:
    # Entry 039's baseline must stay reproducible.
    from pathlib import Path
    assert Path("models_saved/modulation_cnn.pt").is_file()


def test_the_checkpoint_records_its_training_sweep() -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    assert payload["samples_per_symbol_sweep"] == [4, 8, 16, 32]
    assert payload["sample_length"] == 128, "the 128-sample window was NOT changed"


def test_the_checkpoint_has_its_required_metrics_sibling() -> None:
    sibling = metrics_path(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    assert sibling.is_file()
    assert json.loads(sibling.read_text())["model_sha256"] == payload["model_sha256"]


def test_the_checkpoint_is_digital_only() -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    assert sorted(payload["labels"]) == ["8PSK", "BPSK", "CPFSK", "GFSK",
                                         "PAM4", "QAM16", "QAM64", "QPSK"]
    assert not {"AM-DSB", "AM-SSB", "WBFM"} & set(payload["labels"])


# --- what the change was for -------------------------------------------------------------------


@pytest.mark.parametrize("sps", [4, 8, 16, 32])
@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "16QAM"])
def test_classification_now_works_across_the_oversampling_range(modulation, sps) -> None:
    expected = {"BPSK": "BPSK", "QPSK": "QPSK", "16QAM": "QAM16"}[modulation]

    prediction = predict_modulation(_capture(modulation, sps), CHECKPOINT)

    assert prediction.available is not False
    assert prediction.label == expected, f"{modulation} at sps={sps} -> {prediction.label}"


def test_high_oversampling_no_longer_collapses() -> None:
    # Entry 039: CNN top-1 was 0/15 for BPSK/QPSK/8PSK/QAM16/QAM64 at sps 32.
    for modulation, expected in [("BPSK", "BPSK"), ("QPSK", "QPSK"), ("8PSK", "8PSK"),
                                 ("16QAM", "QAM16"), ("64QAM", "QAM64")]:
        prediction = predict_modulation(_capture(modulation, 32), CHECKPOINT)
        assert prediction.label == expected, f"{modulation} sps=32 -> {prediction.label}"


# --- analog must be untouched: the classical gate never needed the CNN --------------------------


@pytest.mark.parametrize("scheme,kw,expected", [
    ("am_dsb", {"carrier_offset_hz": 20_000.0}, "AM-DSB"),
    ("am_ssb", {"carrier_offset_hz": 20_000.0, "sideband": "upper"}, "AM-SSB"),
    ("wbfm", {"carrier_offset_hz": 20_000.0, "frequency_deviation_hz": 15_000.0}, "WBFM"),
])
def test_analog_still_routes_correctly_without_analog_cnn_classes(scheme, kw, expected) -> None:
    from radiofry.dsp.cyclostationary import estimate_modulation_family
    from radiofry.fusion.confidence_fusion import fuse_modulation
    from radiofry.synthetic_gen.v1.analog import (
        AnalogSampleSpec, generate_message, modulate_analog)

    spec = AnalogSampleSpec(sample_rate_hz=FS, num_samples=N, snr_db=20.0,
                            seed=7_000_002, scheme=scheme, **kw)
    iq, _ = add_awgn(modulate_analog(generate_message(spec)[0], spec), 20.0,
                     np.random.default_rng([7_000_002, 2]))
    signal = preprocess(UnifiedSignalContainer(iq, FS))
    classical = estimate_modulation_family(signal.iq)
    prediction = predict_modulation(signal, CHECKPOINT)

    fusion = fuse_modulation(prediction.label, prediction.confidence, classical.family,
                             classical_evidence=classical.evidence,
                             classical_confidence=classical.confidence)

    assert fusion.label == expected
    assert fusion.analog_route == "classical_subtype"


def test_a_digital_capture_is_not_routed_to_an_analog_demodulator() -> None:
    from radiofry.dsp.cyclostationary import estimate_modulation_family
    from radiofry.fusion.confidence_fusion import fuse_modulation

    signal = _capture("QPSK", 8)
    classical = estimate_modulation_family(signal.iq)
    prediction = predict_modulation(signal, CHECKPOINT)

    fusion = fuse_modulation(prediction.label, prediction.confidence, classical.family,
                             classical_evidence=classical.evidence,
                             classical_confidence=classical.confidence)

    assert fusion.label not in {"AM-DSB", "AM-SSB", "WBFM"}
