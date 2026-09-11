"""Guards on the RadioML 2018.01A adapter.

The dangerous failures here are silent ones, in rough order of damage:

1. Using DeepSig's shipped `classes.txt`, which measurement showed is the WRONG ordering.
   Every label would be wrong and every number meaningless, with nothing crashing.
2. Mapping a class onto a RadioFry label that it is not - GMSK onto GFSK, or 4ASK onto PAM4 -
   which manufactures agreement.
3. Block arithmetic that drifts from the stored labels, so frames get the wrong class.
4. Splits that overlap, which would put training frames in the held-out set.

The dataset lives outside the repository, so tests needing it skip when it is absent.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py", reason="h5py is part of the optional dev/training extras")

from radiofry.datasets.radioml2018 import (  # noqa: E402
    CLASSES,
    FRAMES_PER_BLOCK,
    LABEL_MAP,
    SHIPPED_WRONG_ORDER,
    SNR_LEVELS,
    block_bounds,
    block_start,
    index_digest,
    inventory,
    load_indices,
    select_indices,
    split_manifest,
    verify_layout,
)

DATASET = Path.home() / "Documents" / "RadioFry" / "dataset 2"

requires_dataset = pytest.mark.skipif(
    not any(DATASET.glob("*.hdf5")) if DATASET.is_dir() else True,
    reason="RadioML 2018.01A is not present on this machine")


# --- 1. the class ordering ---------------------------------------------------------------------


def test_the_measured_ordering_is_used_not_the_shipped_one() -> None:
    """DeepSig's classes.txt disagrees with the data; using it mislabels every frame."""

    assert CLASSES != SHIPPED_WRONG_ORDER
    assert CLASSES[:4] == ("OOK", "4ASK", "8ASK", "BPSK")
    assert CLASSES[21] == "FM" and CLASSES[22] == "GMSK"


def test_the_ordering_has_the_right_shape() -> None:
    assert len(CLASSES) == 24
    assert len(set(CLASSES)) == 24
    assert set(CLASSES) == set(SHIPPED_WRONG_ORDER), (
        "the two orderings must contain the same 24 names, differing only in order")


# --- 2. the label mapping -----------------------------------------------------------------------


def test_only_exact_counterparts_are_mapped() -> None:
    assert LABEL_MAP == {"BPSK": "BPSK", "QPSK": "QPSK", "8PSK": "8PSK",
                         "16QAM": "QAM16", "64QAM": "QAM64"}


def test_gmsk_is_not_mapped_onto_gfsk() -> None:
    """GMSK is CPM with h=0.5; RadioFry's GFSK is a different generator configuration."""

    assert "GMSK" not in LABEL_MAP
    assert "GFSK" not in LABEL_MAP.values()


def test_4ask_is_not_mapped_onto_pam4() -> None:
    """Measured as unipolar (DC fraction 68%); PAM4 is bipolar. Unresolved, so unmapped."""

    assert "4ASK" not in LABEL_MAP
    assert "PAM4" not in LABEL_MAP.values()


def test_analog_classes_are_not_mapped_into_a_digital_label_set() -> None:
    for analog in ("FM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC"):
        assert analog not in LABEL_MAP


# --- 3. block arithmetic and splits --------------------------------------------------------------


def test_block_start_is_class_major_then_snr_ascending() -> None:
    assert block_start(0, -20) == 0
    assert block_start(0, -18) == FRAMES_PER_BLOCK
    assert block_start(1, -20) == len(SNR_LEVELS) * FRAMES_PER_BLOCK
    assert block_start(23, 30) == (24 * 26 - 1) * FRAMES_PER_BLOCK


def test_block_start_rejects_out_of_range_arguments() -> None:
    with pytest.raises(ValueError):
        block_start(24, 30)
    with pytest.raises(ValueError):
        block_start(0, 31)


def test_the_three_splits_tile_the_block_exactly_and_do_not_overlap() -> None:
    bounds = [block_bounds(name) for name in ("train", "val", "test")]

    assert bounds[0][0] == 0
    assert bounds[-1][1] == FRAMES_PER_BLOCK
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end == start, "splits must be contiguous with no gap and no overlap"
    assert sum(high - low for low, high in bounds) == FRAMES_PER_BLOCK


def test_an_unknown_split_is_rejected() -> None:
    with pytest.raises(ValueError):
        block_bounds("holdout")


def test_snr_levels_span_the_documented_range() -> None:
    assert SNR_LEVELS[0] == -20 and SNR_LEVELS[-1] == 30
    assert len(SNR_LEVELS) == 26
    assert all(b - a == 2 for a, b in zip(SNR_LEVELS, SNR_LEVELS[1:]))


def test_the_split_manifest_records_the_protocol() -> None:
    manifest = split_manifest(DATASET)

    assert "contiguous" in manifest
    assert "sealed" in manifest


def test_index_digest_is_content_addressed() -> None:
    a = np.array([1, 2, 3], dtype=np.int64)

    assert index_digest(a) == index_digest(np.array([1, 2, 3]))
    assert index_digest(a) != index_digest(np.array([1, 2, 4]))


# --- 4. against the real file ---------------------------------------------------------------------


@requires_dataset
def test_the_block_arithmetic_agrees_with_the_stored_labels() -> None:
    """If this drifts, every frame silently carries the wrong class."""

    assert verify_layout(DATASET, samples=40)


@requires_dataset
def test_the_dataset_is_balanced_and_complete() -> None:
    summary = inventory(DATASET)

    assert summary["frames"] == 24 * 26 * 4096
    assert summary["shape"] == (24 * 26 * 4096, 1024, 2)
    assert summary["configurations"] == 24 * 26
    assert summary["balanced"]
    assert set(summary["class_counts"].values()) == {26 * 4096}


@requires_dataset
def test_splits_are_disjoint_on_real_indices() -> None:
    kwargs = dict(classes=("BPSK",), snr_db=(30,), per_config=None, seed=0)
    train = set(select_indices(DATASET, "train", **kwargs).tolist())
    validation = set(select_indices(DATASET, "val", **kwargs).tolist())
    test = set(select_indices(DATASET, "test", **kwargs).tolist())

    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == FRAMES_PER_BLOCK


@requires_dataset
def test_selection_is_stratified_over_class_and_snr() -> None:
    indices = select_indices(DATASET, "val", classes=("BPSK", "QPSK", "16QAM"),
                             snr_db=(-20, 0, 30), per_config=7, seed=3)
    subset = load_indices(DATASET, "val", indices)

    assert len(subset) == 3 * 3 * 7
    for name in ("BPSK", "QPSK", "16QAM"):
        assert int((subset.modulation == name).sum()) == 3 * 7
    for snr in (-20, 0, 30):
        assert int((subset.snr_db == snr).sum()) == 3 * 7


@requires_dataset
def test_selection_is_reproducible() -> None:
    kwargs = dict(classes=("QPSK",), snr_db=(10,), per_config=20)
    first = select_indices(DATASET, "train", seed=11, **kwargs)
    again = select_indices(DATASET, "train", seed=11, **kwargs)
    other = select_indices(DATASET, "train", seed=12, **kwargs)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


@requires_dataset
def test_loaded_labels_match_what_selection_asked_for() -> None:
    indices = select_indices(DATASET, "test", classes=("8PSK",), snr_db=(20,),
                             per_config=6, seed=2)
    subset = load_indices(DATASET, "test", indices)

    assert set(subset.modulation.tolist()) == {"8PSK"}
    assert set(subset.snr_db.tolist()) == {20}
    assert set(subset.radiofry_label.tolist()) == {"8PSK"}
    assert subset.mapped.all()


@requires_dataset
def test_unmapped_classes_carry_an_empty_label_rather_than_a_guess() -> None:
    indices = select_indices(DATASET, "test", classes=("GMSK", "FM", "4ASK"),
                             snr_db=(30,), per_config=5, seed=2)
    subset = load_indices(DATASET, "test", indices)

    assert subset.mapped.sum() == 0
    assert all(label == "" for label in subset.radiofry_label)


@requires_dataset
def test_frames_load_as_complex_and_are_finite() -> None:
    indices = select_indices(DATASET, "val", classes=("QPSK",), snr_db=(30,),
                             per_config=8, seed=4)
    subset = load_indices(DATASET, "val", indices)

    frames = subset.complex_frames()
    assert frames.dtype == np.complex64
    assert frames.shape == (8, 1024)
    assert np.all(np.isfinite(frames))


@requires_dataset
def test_a_container_is_the_pipeline_signal_type() -> None:
    indices = select_indices(DATASET, "val", classes=("BPSK",), snr_db=(30,),
                             per_config=4, seed=5)
    subset = load_indices(DATASET, "val", indices)

    container = subset.container(0)

    assert container.iq.shape == (1024,)
    assert container.metadata["source"] == "radioml2018.01a"
    assert container.metadata["modulation"] == "BPSK"
