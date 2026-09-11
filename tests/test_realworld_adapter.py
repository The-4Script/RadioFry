"""The real-world dataset adapter reads the archive read-only and labels it honestly.

The risks this guards are not crashes. They are: silently mis-labelling frames by assuming
a class order; manufacturing agreement by mapping GMSK onto GFSK; and modifying or
extracting the raw archive. Each has a test.

The archive lives outside the repository, so every test skips when it is absent.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py", reason="h5py is part of the optional dev/training extras")

from radiofry.datasets.realworld_multipath import (  # noqa: E402
    CARRIER_HZ,
    FRAME_SAMPLES,
    LABEL_MAP,
    SAMPLE_RATE_HZ,
    UNMAPPED,
    inventory,
    load_subset,
)

ARCHIVE = (Path.home() / "Downloads"
           / "Real-World IQ Dataset for Automatic Radio Modulati.zip")

requires_archive = pytest.mark.skipif(
    not ARCHIVE.is_file(),
    reason="the real-world dataset archive is not present on this machine")


# --- label handling is the dangerous part -------------------------------------------------


def test_only_exact_counterparts_are_mapped() -> None:
    """A mapping that is nearly right is worse than one that abstains."""
    assert LABEL_MAP == {"BPSK": "BPSK", "QPSK": "QPSK", "QAM": "QAM16"}
    assert set(UNMAPPED) == {"GMSK", "OFDM", "NBFM", "WBFM"}


def test_gmsk_is_not_mapped_onto_gfsk() -> None:
    """Both use a Gaussian pulse at BT=0.3, but they are not the same modulation.

    Mapping them would manufacture agreement and make any accuracy figure meaningless.
    """
    assert "GMSK" not in LABEL_MAP
    assert "GFSK" not in LABEL_MAP.values()


def test_analog_classes_are_not_mapped_into_a_digital_label_set() -> None:
    """The production checkpoint is 8-class digital-only; NBFM/WBFM have no output."""
    for analog in ("NBFM", "WBFM"):
        assert analog not in LABEL_MAP


def test_the_dataset_constants_match_the_published_configuration() -> None:
    assert SAMPLE_RATE_HZ == 2_000_000.0
    assert CARRIER_HZ == 2.4e9
    assert FRAME_SAMPLES == 1024


# --- reading the archive -----------------------------------------------------------------------


@requires_archive
def test_the_label_mapping_is_read_from_the_file_not_assumed() -> None:
    """A differently-encoded release must not be silently mis-labelled."""
    summary = inventory(ARCHIVE)

    mapping = summary["test"]["mapping"]
    assert mapping == {"BPSK": 0, "QPSK": 1, "QAM": 2, "GMSK": 3,
                       "OFDM": 4, "NBFM": 5, "WBFM": 6}


@requires_archive
def test_the_inventory_reports_the_documented_shape() -> None:
    summary = inventory(ARCHIVE)

    assert set(summary) == {"train", "val", "test"}
    assert summary["train"]["frames"] == 400_000
    assert summary["val"]["frames"] == 80_000
    assert summary["test"]["frames"] == 80_000
    for split in summary.values():
        assert split["frame_samples"] == FRAME_SAMPLES
        assert split["dtype"] == "float16"
        assert split["snr_values"] == [20, 22, 24, 26, 28, 30]
        assert set(split["channels"]) == {0, 1}


@requires_archive
def test_loading_returns_complex_frames_at_the_dataset_sample_rate() -> None:
    subset = load_subset(ARCHIVE, "test", limit=64, seed=1)

    assert subset.iq.dtype == np.complex64
    assert subset.iq.shape == (64, FRAME_SAMPLES)
    assert subset.sample_rate_hz == SAMPLE_RATE_HZ
    assert np.all(np.isfinite(subset.iq))


@requires_archive
def test_every_frame_carries_its_own_metadata() -> None:
    subset = load_subset(ARCHIVE, "test", limit=32, seed=1)

    assert len(subset.modulation) == len(subset)
    assert len(subset.channel) == len(subset)
    assert len(subset.snr_db) == len(subset)
    assert set(subset.channel.tolist()) <= {0, 1}
    assert set(subset.snr_db.tolist()) <= {20, 22, 24, 26, 28, 30}


@requires_archive
def test_unmapped_classes_get_an_empty_label_rather_than_a_guess() -> None:
    subset = load_subset(ARCHIVE, "test", classes=UNMAPPED, limit=48, seed=2)

    assert subset.mapped.sum() == 0, "no unmapped class may acquire a RadioFry label"
    assert all(label == "" for label in subset.radiofry_label)


@requires_archive
def test_mapped_classes_get_their_exact_counterpart() -> None:
    subset = load_subset(ARCHIVE, "test", classes=("BPSK", "QPSK", "QAM"),
                         limit=96, seed=2)

    assert subset.mapped.all()
    for dataset_name, radiofry_name in zip(subset.modulation, subset.radiofry_label):
        assert radiofry_name == LABEL_MAP[dataset_name]


@requires_archive
def test_a_container_is_the_pipeline_signal_type() -> None:
    subset = load_subset(ARCHIVE, "test", limit=8, seed=3)

    container = subset.container(0)

    assert container.iq.shape == (FRAME_SAMPLES,)
    assert container.sample_rate == SAMPLE_RATE_HZ
    assert container.metadata["center_frequency_hz"] == CARRIER_HZ
    assert container.metadata["source"] == "realworld_multipath"


@requires_archive
def test_sampling_is_deterministic_and_not_a_prefix() -> None:
    """The file is ordered by configuration, so a prefix would be one class at one SNR."""
    first = load_subset(ARCHIVE, "test", limit=200, seed=5)
    again = load_subset(ARCHIVE, "test", limit=200, seed=5)
    different = load_subset(ARCHIVE, "test", limit=200, seed=6)

    assert np.array_equal(first.iq, again.iq), "same seed must give the same frames"
    assert not np.array_equal(first.iq, different.iq)
    assert len(set(first.modulation.tolist())) > 1, "a sample must span classes"


@requires_archive
def test_an_unknown_subset_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_subset(ARCHIVE, "not_a_split")


# --- the raw archive is never touched -----------------------------------------------------------


@requires_archive
def test_loading_does_not_modify_or_extract_the_archive() -> None:
    """Raw data stays raw: same bytes, same mtime, nothing written beside it."""
    before_size = ARCHIVE.stat().st_size
    before_mtime = ARCHIVE.stat().st_mtime
    siblings_before = {p.name for p in ARCHIVE.parent.iterdir()}

    load_subset(ARCHIVE, "test", limit=16, seed=4)
    inventory(ARCHIVE)

    assert ARCHIVE.stat().st_size == before_size
    assert ARCHIVE.stat().st_mtime == before_mtime
    assert {p.name for p in ARCHIVE.parent.iterdir()} == siblings_before, (
        "nothing may be extracted next to the archive")


@requires_archive
def test_the_archive_is_opened_read_only() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.mode == "r"
