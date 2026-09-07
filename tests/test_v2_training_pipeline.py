"""Tests for the V2.0 synthetic training pipeline (BANK.md Entry 015)."""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.models.modulation_inference import _fixed_iq
from radiofry.models.signal_features import add_signal_features
from radiofry.synthetic_gen.v1 import MODULATIONS
from radiofry.training.train_v2_synthetic import (
    FRAME_LENGTH,
    GENERALISATION_SEED_BASE,
    TRAIN_SEED_BASE,
    build_arrays,
    build_capture_specs,
    frames_from_capture,
    render_capture,
    split_capture_specs,
)

LABELS = sorted({spec.radiofry_label for spec in MODULATIONS.values()})


def _specs(replicates=2, seed_base=TRAIN_SEED_BASE):
    return build_capture_specs(replicates=replicates, seed_base=seed_base)


# --- label mapping -----------------------------------------------------------------


def test_labels_follow_the_existing_project_conventions() -> None:
    assert LABELS == ["8PSK", "BPSK", "CPFSK", "QAM16", "QAM64", "QPSK"]
    assert {s.label for s in _specs()} == set(LABELS)


def test_bfsk_is_labelled_cpfsk_and_swept_over_both_indices() -> None:
    fsk = [s for s in _specs() if s.modulation == "BFSK"]

    assert {s.label for s in fsk} == {"CPFSK"}
    assert {s.fsk_modulation_index for s in fsk} == {0.5, 1.0}


def test_non_fsk_specs_carry_no_modulation_index() -> None:
    assert all(s.fsk_modulation_index is None for s in _specs() if s.modulation != "BFSK")


# --- deterministic, capture-level splitting -----------------------------------------


def test_split_is_deterministic_for_a_given_seed() -> None:
    specs = _specs(replicates=5)

    first = split_capture_specs(specs, seed=99)
    second = split_capture_specs(specs, seed=99)

    for bucket in ("train", "validation", "test"):
        assert [s.capture_id for s in first[bucket]] == [s.capture_id for s in second[bucket]]


def test_split_changes_with_the_seed() -> None:
    specs = _specs(replicates=5)

    a = {s.capture_id for s in split_capture_specs(specs, seed=1)["test"]}
    b = {s.capture_id for s in split_capture_specs(specs, seed=2)["test"]}

    assert a != b


def test_every_capture_lands_in_exactly_one_split() -> None:
    specs = _specs(replicates=5)

    split = split_capture_specs(specs, seed=99)
    buckets = [{s.capture_id for s in split[name]} for name in ("train", "validation", "test")]

    assert sum(len(b) for b in buckets) == len(specs)
    assert buckets[0] & buckets[1] == set()
    assert buckets[0] & buckets[2] == set()
    assert buckets[1] & buckets[2] == set()


def test_no_capture_seed_is_shared_between_splits() -> None:
    # Seed-level separation is what stops two splits rendering the same waveform.
    split = split_capture_specs(_specs(replicates=5), seed=99)
    seeds = {name: {s.seed for s in specs} for name, specs in split.items()}

    assert seeds["train"] & seeds["validation"] == set()
    assert seeds["train"] & seeds["test"] == set()
    assert seeds["validation"] & seeds["test"] == set()


def test_windows_of_one_capture_never_straddle_the_split() -> None:
    # 10 replicates gives 6/2/2 per stratum; below ~5 a stratum can leave a bucket empty.
    specs = _specs(replicates=10)
    split = split_capture_specs(specs, seed=99)
    ids = {name: {s.capture_id for s in members} for name, members in split.items()}

    built = {name: set(build_arrays(members[:2], LABELS)["capture_ids"]) for name, members in split.items()}
    assert built["train"] <= ids["train"]
    assert built["test"] <= ids["test"]
    assert built["train"] & built["test"] == set()


def test_training_and_generalisation_seed_ranges_are_disjoint() -> None:
    train = {s.seed for s in _specs(replicates=4, seed_base=TRAIN_SEED_BASE)}
    holdout = {s.seed for s in _specs(replicates=4, seed_base=GENERALISATION_SEED_BASE)}

    assert train & holdout == set()


# --- input representation matches inference ------------------------------------------


def test_training_frames_use_the_same_preprocessing_as_inference() -> None:
    # The whole point: no train/inference skew. A frame built by the training path must
    # equal the frame the runtime builds from that same 128-sample window.
    iq = render_capture(_specs()[0])
    frames = frames_from_capture(iq)

    window = UnifiedSignalContainer(iq[:FRAME_LENGTH], 200_000.0)
    expected = add_signal_features(_fixed_iq(window, FRAME_LENGTH), include_engineered=True)

    np.testing.assert_allclose(frames[0], expected, rtol=1e-5, atol=1e-6)


def test_frames_have_the_shape_the_model_expects() -> None:
    frames = frames_from_capture(render_capture(_specs()[0]))

    assert frames.ndim == 3
    assert frames.shape[1:] == (4, FRAME_LENGTH)
    assert frames.dtype == np.float32
    assert np.all(np.isfinite(frames))


def test_frames_are_non_overlapping_and_cover_the_capture() -> None:
    iq = render_capture(_specs()[0])

    frames = frames_from_capture(iq)

    assert frames.shape[0] == iq.size // FRAME_LENGTH


def test_build_arrays_aligns_targets_snrs_and_capture_ids() -> None:
    specs = _specs()[:3]

    data = build_arrays(specs, LABELS)

    assert data["frames"].shape[0] == data["targets"].shape[0] == data["snrs"].shape[0]
    assert data["frames"].shape[0] == len(data["capture_ids"])
    assert set(np.unique(data["capture_ids"])) == {s.capture_id for s in specs}
    for spec in specs:
        mask = data["capture_ids"] == spec.capture_id
        assert set(np.unique(data["targets"][mask])) == {LABELS.index(spec.label)}


def test_capture_rendering_is_reproducible() -> None:
    spec = _specs()[0]

    np.testing.assert_array_equal(render_capture(spec), render_capture(spec))
