"""Guards on the real-data training path.

The risks here are not crashes. They are, in order of how much damage they would do:

1. Train/inference skew. The candidate checkpoint is meant to drop into the existing runtime
   unchanged, so the vectorised window builder must reproduce
   `modulation_inference._window_frames` + `signal_features.add_signal_features` exactly. A
   silent difference would make every reported number describe a pipeline that is not the one
   that ships.
2. Reading the sealed held-out split during training, which would void the whole experiment.
3. Overwriting the frozen production checkpoint.
4. Losing reproducibility - unfixed seeds, or a selection that cannot be replayed.

The dataset lives outside the repository, so tests needing it skip when it is absent.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py", reason="h5py is part of the optional dev/training extras")
pytest.importorskip("torch", reason="torch is part of the optional ml extras")

from radiofry.contracts import UnifiedSignalContainer  # noqa: E402
from radiofry.datasets.realworld_multipath import (  # noqa: E402
    DATASET_CLASSES,
    HELDOUT_SNR_DB,
    SEALED_SPLIT,
    TRAIN_SNR_DB,
    select_indices,
)
from radiofry.models.modulation_inference import _window_frames  # noqa: E402
from radiofry.models.signal_features import add_signal_features  # noqa: E402
from radiofry.training.train_realworld import (  # noqa: E402
    FRAME_LENGTH,
    PRODUCTION_CHECKPOINT,
    TrainingConfig,
    _forbid_sealed_split,
    build_windows,
    inference_window_starts,
    load_pool,
    train,
)

DATASET_ROOT = (Path.home() / "Documents" / "RadioFry"
                / "Real-World IQ Dataset for Automatic Radio Modulati")

requires_dataset = pytest.mark.skipif(
    not (DATASET_ROOT / "dataset" / "subset_val.h5").is_file(),
    reason="the real-world dataset is not present on this machine")


# --- 1. the input contract ------------------------------------------------------------------


def test_build_windows_reproduces_the_production_path_exactly() -> None:
    """The whole comparison against V3 rests on this being identical, not merely close."""

    rng = np.random.default_rng(3)
    frames = (rng.normal(size=(6, 1024)) + 1j * rng.normal(size=(6, 1024))).astype(np.complex64)
    starts = inference_window_starts()

    for start in starts:
        mine = build_windows(frames, np.full(len(frames), start))
        for row, frame in enumerate(frames):
            block = frame[start:start + FRAME_LENGTH]
            values = np.stack([block.real, block.imag]).astype(np.float32)
            power = np.sqrt(np.mean(values ** 2))
            reference = add_signal_features(values / power if power > 0 else values,
                                            include_engineered=True)
            assert np.array_equal(mine[row], reference)


def test_the_window_positions_are_the_ones_production_uses() -> None:
    """Evaluation must sit on production's window grid, not a convenient alternative."""

    rng = np.random.default_rng(4)
    iq = (rng.normal(size=1024) + 1j * rng.normal(size=1024)).astype(np.complex64)
    signal = UnifiedSignalContainer(iq, 2_000_000.0, "iq", {})

    production = _window_frames(signal, FRAME_LENGTH, 4)
    starts = inference_window_starts()

    assert len(starts) == len(production)
    for start, expected in zip(starts, production):
        block = iq[start:start + FRAME_LENGTH]
        values = np.stack([block.real, block.imag]).astype(np.float32)
        power = np.sqrt(np.mean(values ** 2))
        assert np.array_equal(values / power, expected)


def test_a_zero_power_window_does_not_produce_nan() -> None:
    frames = np.zeros((2, 1024), dtype=np.complex64)

    windows = build_windows(frames, np.zeros(2, dtype=int))

    assert np.all(np.isfinite(windows))


def test_build_windows_rejects_a_mismatched_start_count() -> None:
    frames = np.zeros((3, 1024), dtype=np.complex64)

    with pytest.raises(ValueError):
        build_windows(frames, np.zeros(2, dtype=int))


def test_windows_carry_the_four_production_channels() -> None:
    rng = np.random.default_rng(5)
    frames = (rng.normal(size=(4, 1024)) + 1j * rng.normal(size=(4, 1024))).astype(np.complex64)

    windows = build_windows(frames, np.zeros(4, dtype=int))

    assert windows.shape == (4, 4, FRAME_LENGTH)
    assert windows.dtype == np.float32


# --- 2. the sealed split ---------------------------------------------------------------------


def test_the_sealed_split_cannot_be_loaded_through_the_training_path() -> None:
    """A held-out set that training can reach is not held out."""

    with pytest.raises(PermissionError):
        _forbid_sealed_split(SEALED_SPLIT)


def test_the_other_splits_are_reachable() -> None:
    assert _forbid_sealed_split("train") == "train"
    assert _forbid_sealed_split("val") == "val"


@requires_dataset
def test_load_pool_refuses_the_sealed_split() -> None:
    with pytest.raises(PermissionError):
        load_pool(DATASET_ROOT, SEALED_SPLIT, classes=("BPSK",), snr_db=(20,),
                  per_config_limit=4, seed=0)


@requires_dataset
def test_the_sealed_split_is_reachable_only_by_asking_for_it_explicitly() -> None:
    """Evaluation still needs it; the guard is a deliberate step, not a locked door."""

    pool = load_pool(DATASET_ROOT, SEALED_SPLIT, classes=("BPSK",), snr_db=(30,),
                     per_config_limit=4, seed=0, allow_sealed=True)

    assert len(pool) > 0


# --- 3. the production checkpoint ------------------------------------------------------------


def test_training_refuses_to_overwrite_the_production_checkpoint(tmp_path) -> None:
    config = TrainingConfig(mode="scratch", dataset_root=str(DATASET_ROOT),
                            output=str(tmp_path / PRODUCTION_CHECKPOINT))

    with pytest.raises(PermissionError):
        train(config)


def test_an_output_path_is_required() -> None:
    with pytest.raises(ValueError):
        train(TrainingConfig(mode="scratch", dataset_root=str(DATASET_ROOT), output=""))


# --- 4. the protocol -------------------------------------------------------------------------


def test_the_training_and_heldout_snr_levels_are_disjoint() -> None:
    """The point of the protocol: evaluation frames come from different recordings."""

    assert set(TRAIN_SNR_DB).isdisjoint(HELDOUT_SNR_DB)
    assert set(TRAIN_SNR_DB) | set(HELDOUT_SNR_DB) == {20, 22, 24, 26, 28, 30}


def test_transfer_modes_require_a_starting_checkpoint() -> None:
    for mode in ("finetune", "linear_probe"):
        with pytest.raises(ValueError):
            TrainingConfig(mode=mode, initial_checkpoint=None)


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(mode="whatever")


def test_scratch_needs_no_checkpoint() -> None:
    assert TrainingConfig(mode="scratch").initial_checkpoint is None


@requires_dataset
def test_selection_is_reproducible_and_stratified() -> None:
    """Same seed, same frames; and every configuration contributes equally."""

    first = select_indices(DATASET_ROOT, "val", classes=("BPSK", "QPSK"),
                           channels=(0, 1), snr_db=TRAIN_SNR_DB,
                           per_config_limit=20, seed=11)
    again = select_indices(DATASET_ROOT, "val", classes=("BPSK", "QPSK"),
                           channels=(0, 1), snr_db=TRAIN_SNR_DB,
                           per_config_limit=20, seed=11)
    different = select_indices(DATASET_ROOT, "val", classes=("BPSK", "QPSK"),
                               channels=(0, 1), snr_db=TRAIN_SNR_DB,
                               per_config_limit=20, seed=12)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, different)
    # 2 classes x 2 channels x 3 SNR levels x 20 frames
    assert first.size == 2 * 2 * 3 * 20


@requires_dataset
def test_a_pool_holds_only_the_requested_snr_levels_and_is_class_balanced() -> None:
    pool = load_pool(DATASET_ROOT, "val", classes=("BPSK", "QPSK", "QAM"),
                     snr_db=TRAIN_SNR_DB, per_config_limit=15, seed=7)

    assert set(pool.snr_db.tolist()) <= set(TRAIN_SNR_DB)
    assert set(pool.snr_db.tolist()).isdisjoint(HELDOUT_SNR_DB)
    counts = {name: int((pool.modulation == name).sum())
              for name in ("BPSK", "QPSK", "QAM")}
    assert len(set(counts.values())) == 1, f"classes are not balanced: {counts}"


@requires_dataset
def test_pool_labels_index_the_declared_class_order() -> None:
    pool = load_pool(DATASET_ROOT, "val", classes=DATASET_CLASSES,
                     snr_db=(20,), per_config_limit=3, seed=9)

    for label, name in zip(pool.labels, pool.modulation):
        assert DATASET_CLASSES[label] == name


@requires_dataset
def test_frames_are_held_as_float16_to_bound_memory() -> None:
    pool = load_pool(DATASET_ROOT, "val", classes=("BPSK",), snr_db=(20,),
                     per_config_limit=8, seed=3)

    assert pool.raw.dtype == np.float16
    assert pool.complex_frames(np.arange(len(pool))).dtype == np.complex64
