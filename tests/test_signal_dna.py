"""Signal-DNA reads the production CNN; it must not change or misrepresent it.

Four failures would make this feature actively misleading, and each has a test that
would catch it:

* using the softmax output as the "embedding" - then the picture would show the decision,
  not the representation, and overlap would be circular evidence;
* drifting off the production preprocessing path - then the space shown would not be the
  space production classifies in;
* mutating the model while reading it (a left-behind hook, training mode, a gradient) -
  then looking at the CNN would change what it predicts;
* promoting the CNN's own prediction into the ground-truth field for an unlabelled
  capture - then the plot would confirm itself.

The reference dataset is regenerated from the V1 generator rather than shipped, so one
test asserts that the regeneration reproduces the frozen captures.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional ml extra")

from radiofry.contracts import UnifiedSignalContainer  # noqa: E402
from radiofry.models.embedding import (  # noqa: E402
    DEFAULT_LAYER,
    EMBEDDING_LAYERS,
    PENULTIMATE,
    POOLED,
    SOURCE_REAL,
    SOURCE_SYNTHETIC,
    UNLABELLED,
    EmbeddingRecord,
    aggregate_windows,
    build_set,
    embed_signal,
    load_embedding_model,
)
from radiofry.models.embedding_dataset import (  # noqa: E402
    GENERATOR_TO_LABEL,
    centroid_distances,
    class_separation,
    collect_benchmark_grid,
    collect_v1_captures,
    filter_set,
    fit_pca,
    generate_grid_capture,
    nearest_class_centroids,
)
from radiofry.pipeline import DEFAULT_MODULATION_MODEL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / DEFAULT_MODULATION_MODEL
V1_CAPTURES = PROJECT_ROOT / "data" / "synthetic_v1" / "captures"

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason="production checkpoint not present")
requires_v1 = pytest.mark.skipif(not V1_CAPTURES.exists(),
                                 reason="frozen V1 dataset not present")


@pytest.fixture(scope="module")
def loaded():
    model, reason = load_embedding_model(CHECKPOINT)
    if model is None:
        pytest.skip(f"checkpoint unavailable: {reason}")
    return model


def _v1_signal(stem: str = "QPSK_snr20dB_r000"):
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import load_capture

    truth = json.loads((V1_CAPTURES / f"{stem}.json").read_text(encoding="utf-8"))
    entry = next(f for f in truth["files"] if f["file_format"] == "iq")
    signal = load_capture(V1_CAPTURES / f"{stem}.iq",
                          sample_rate=truth["signal"]["sample_rate_hz"],
                          iq_format=IQFormat(entry["dtype"], entry["byte_order"]))
    return signal, truth


def _noise_signal(samples: int = 8_192, seed: int = 4):
    rng = np.random.default_rng(seed)
    iq = ((rng.normal(size=samples) + 1j * rng.normal(size=samples))
          / np.sqrt(2)).astype(np.complex64)
    return UnifiedSignalContainer(iq, 200_000.0, "iq")


# --- loading the production checkpoint --------------------------------------------------


@requires_checkpoint
def test_the_production_checkpoint_loads_in_eval_mode(loaded) -> None:
    assert loaded.labels == ("8PSK", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16",
                             "QAM64", "QPSK")
    assert loaded.input_channels == 4 and loaded.engineered, "production uses iqap"
    assert loaded.sample_length == 128
    assert loaded.model.training is False, "eval mode is what makes this deterministic"
    assert loaded.checkpoint_sha256


def test_a_missing_checkpoint_is_reported_not_raised() -> None:
    model, reason = load_embedding_model(Path("does") / "not" / "exist.pt")

    assert model is None
    assert "not found" in reason.lower()


# --- the embedding is an internal representation, never the softmax ------------------------


@requires_checkpoint
@requires_v1
@pytest.mark.parametrize("layer,width", [(PENULTIMATE, 256), (POOLED, 128)])
def test_each_layer_returns_its_documented_width(loaded, layer, width) -> None:
    signal, _ = _v1_signal()

    records = embed_signal(loaded, signal, capture_id="x", layer=layer)

    assert all(r.vector.shape == (width,) for r in records)
    assert loaded.width(layer) == width


@requires_checkpoint
@requires_v1
def test_no_layer_returns_the_softmax_dimensionality(loaded) -> None:
    """The guard against the whole point being lost.

    An 8-vector on a simplex describes the decision, not the representation; overlap in
    it would be circular evidence for a confusion.
    """
    signal, _ = _v1_signal()

    for layer in EMBEDDING_LAYERS:
        records = embed_signal(loaded, signal, capture_id="x", layer=layer)
        width = records[0].vector.shape[0]
        assert width != len(loaded.labels), f"{layer} returned num_classes dimensions"
        assert width > len(loaded.labels)


@requires_checkpoint
@requires_v1
def test_the_embedding_is_not_a_probability_vector(loaded) -> None:
    signal, _ = _v1_signal()

    vector = embed_signal(loaded, signal, capture_id="x")[0].vector

    assert not np.isclose(vector.sum(), 1.0), "a penultimate ReLU is not a distribution"
    assert vector.max() > 1.0 or vector.min() < 0.0 or vector.sum() > 1.5


@requires_checkpoint
def test_an_unknown_layer_is_rejected(loaded) -> None:
    with pytest.raises(ValueError):
        embed_signal(loaded, _noise_signal(), capture_id="x", layer="softmax")


# --- it uses the production path, and does not disturb it ------------------------------------


@requires_checkpoint
@requires_v1
def test_the_embedding_path_reproduces_the_production_prediction(loaded) -> None:
    """Same preprocessing, same model, same window reduction - so the same answer."""
    from radiofry.models.modulation_inference import predict_modulation

    for stem in ("BPSK_snr20dB_r000", "QPSK_snr10dB_r000", "16QAM_snr0dB_r000"):
        signal, _ = _v1_signal(stem)
        production = predict_modulation(signal, CHECKPOINT)
        records = embed_signal(loaded, signal, capture_id=stem)

        assert records[0].predicted_label == production.label, stem
        assert records[0].confidence == pytest.approx(production.confidence, abs=1e-6)


@requires_checkpoint
@requires_v1
def test_extracting_embeddings_does_not_change_what_the_model_predicts(loaded) -> None:
    """No lingering hook, no training mode, no mutated state."""
    from radiofry.models.modulation_inference import predict_modulation

    signal, _ = _v1_signal()
    before = predict_modulation(signal, CHECKPOINT)

    for layer in EMBEDDING_LAYERS:
        embed_signal(loaded, signal, capture_id="x", layer=layer)

    after = predict_modulation(signal, CHECKPOINT)
    assert after.label == before.label
    assert after.confidence == pytest.approx(before.confidence, abs=1e-9)
    assert loaded.model.training is False
    assert not list(loaded.model._forward_hooks.values()), "no hook may be left behind"


@requires_checkpoint
@requires_v1
def test_embeddings_are_deterministic(loaded) -> None:
    signal, _ = _v1_signal()

    first = embed_signal(loaded, signal, capture_id="x")
    second = embed_signal(loaded, signal, capture_id="x")

    for left, right in zip(first, second):
        assert np.array_equal(left.vector, right.vector)


@requires_checkpoint
@requires_v1
def test_the_window_count_controls_how_many_records_are_returned(loaded) -> None:
    signal, _ = _v1_signal()

    assert len(embed_signal(loaded, signal, capture_id="x", windows=4)) == 4
    assert len(embed_signal(loaded, signal, capture_id="x", windows=1)) == 1


@requires_checkpoint
@requires_v1
def test_a_capture_embedded_alone_matches_its_place_in_a_batch(loaded) -> None:
    """Batching must not change a vector - otherwise the plot depends on the dataset."""
    signal, truth = _v1_signal()
    alone = embed_signal(loaded, signal, capture_id="QPSK_snr20dB_r000")

    collected = collect_v1_captures(loaded, root=V1_CAPTURES)
    matching = [r for r in collected.records
                if r.capture_id == "QPSK_snr20dB_r000"]

    assert matching, "the capture should be in the collected set"
    for single, batched in zip(alone, sorted(matching, key=lambda r: r.window_index)):
        assert np.allclose(single.vector, batched.vector, atol=1e-6)


# --- metadata is carried, never invented -------------------------------------------------------


@requires_checkpoint
@requires_v1
def test_metadata_is_preserved_through_extraction(loaded) -> None:
    signal, truth = _v1_signal("BPSK_snr20dB_r000")

    records = embed_signal(
        loaded, signal, capture_id="BPSK_snr20dB_r000", source=SOURCE_SYNTHETIC,
        true_label="BPSK", samples_per_symbol=8, snr_db=20.0, seed=1993804628)

    for index, record in enumerate(records):
        assert record.capture_id == "BPSK_snr20dB_r000"
        assert record.window_index == index
        assert record.true_label == "BPSK"
        assert record.samples_per_symbol == 8
        assert record.snr_db == 20.0
        assert record.seed == 1993804628
        assert record.sample_rate_hz == signal.sample_rate
        assert record.source == SOURCE_SYNTHETIC


@requires_checkpoint
def test_an_unlabelled_capture_never_inherits_the_prediction_as_truth(loaded) -> None:
    """The distinction the brief calls mandatory."""
    records = embed_signal(loaded, _noise_signal(), capture_id="unknown",
                           source=SOURCE_REAL)

    for record in records:
        assert record.true_label is None
        assert record.labelled is False
        assert record.correct is None, "correctness is undefined without ground truth"
        assert record.display_true_label == UNLABELLED
        assert record.predicted_label != record.display_true_label


@requires_checkpoint
def test_an_unrecognised_source_falls_back_to_unknown(loaded) -> None:
    records = embed_signal(loaded, _noise_signal(), capture_id="x", source="marketing")

    assert records[0].source == "unknown"


@requires_checkpoint
def test_an_empty_capture_produces_no_records(loaded) -> None:
    empty = UnifiedSignalContainer(np.empty(0, dtype=np.complex64), 200_000.0, "iq")

    assert embed_signal(loaded, empty, capture_id="x") == []


@requires_checkpoint
def test_a_capture_shorter_than_one_frame_still_embeds(loaded) -> None:
    """`_window_frames` interpolates short captures; the embedding must follow it."""
    short = UnifiedSignalContainer(
        np.ones(16, dtype=np.complex64), 200_000.0, "iq")

    records = embed_signal(loaded, short, capture_id="x")

    assert len(records) == 1
    assert records[0].vector.shape == (256,)
    assert np.all(np.isfinite(records[0].vector))


# --- aggregation ---------------------------------------------------------------------------------


@requires_checkpoint
@requires_v1
def test_window_aggregation_is_the_mean_and_is_marked_as_aggregated(loaded) -> None:
    signal, _ = _v1_signal()
    records = embed_signal(loaded, signal, capture_id="x", true_label="QPSK")

    aggregated = aggregate_windows(records)

    assert len(aggregated) == 1
    assert aggregated[0].window_index == -1, "aggregated records are marked, not faked"
    assert np.allclose(aggregated[0].vector,
                       np.mean([r.vector for r in records], axis=0), atol=1e-6)
    assert aggregated[0].true_label == "QPSK"


# --- the reference dataset is the generator's own output ------------------------------------------


@requires_v1
def test_regeneration_reproduces_a_frozen_v1_capture() -> None:
    """The claim that the grid is the benchmark's data, not a new dataset.

    The stored file is quantised int16, so the comparison is normalised correlation
    rather than equality.
    """
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import load_capture
    from radiofry.synthetic_gen.v1.channel import add_awgn
    from radiofry.synthetic_gen.v1.config import SampleSpec
    from radiofry.synthetic_gen.v1.modulation import generate_source_bits, modulate

    truth = json.loads(
        (V1_CAPTURES / "BPSK_snr20dB_r000.json").read_text(encoding="utf-8"))
    meta, seeds = truth["signal"], truth["seeds"]
    spec = SampleSpec(modulation=truth["modulation"]["name"],
                      num_symbols=meta["num_symbols"],
                      samples_per_symbol=meta["samples_per_symbol"],
                      sample_rate_hz=meta["sample_rate_hz"],
                      snr_db=truth["noise"]["target_snr_db"],
                      seed=seeds["seed"], bits_seed=seeds["bits_seed"])
    regenerated, _ = add_awgn(modulate(generate_source_bits(spec), spec), spec.snr_db,
                              np.random.default_rng([spec.seed, 2]))

    entry = next(f for f in truth["files"] if f["file_format"] == "iq")
    stored = load_capture(V1_CAPTURES / "BPSK_snr20dB_r000.iq",
                          sample_rate=meta["sample_rate_hz"],
                          iq_format=IQFormat(entry["dtype"], entry["byte_order"])).iq

    regenerated = np.asarray(regenerated)
    count = min(regenerated.size, stored.size)
    correlation = abs(np.vdot(regenerated[:count] / np.linalg.norm(regenerated[:count]),
                              stored[:count] / np.linalg.norm(stored[:count])))
    assert correlation > 0.9999, f"regeneration diverged: |corr| = {correlation}"


def test_grid_generation_is_deterministic() -> None:
    first = generate_grid_capture("BPSK", 8, 10.0, 601)
    second = generate_grid_capture("BPSK", 8, 10.0, 601)

    assert np.array_equal(first.iq, second.iq)
    assert not np.array_equal(generate_grid_capture("BPSK", 8, 10.0, 607).iq, first.iq)


@requires_checkpoint
def test_the_grid_covers_every_production_class(loaded) -> None:
    collected = collect_benchmark_grid(
        loaded, samples_per_symbol=(8,), snr_db=(20.0,), seeds=(601,))

    assert set(collected.column("true_label")) == set(GENERATOR_TO_LABEL.values())
    assert len(collected) == len(GENERATOR_TO_LABEL) * 4, "4 windows per capture"


@requires_checkpoint
@requires_v1
def test_v1_collection_carries_ground_truth_from_the_sidecars(loaded) -> None:
    collected = collect_v1_captures(loaded, root=V1_CAPTURES)

    assert collected.ok
    assert all(r.source == SOURCE_SYNTHETIC for r in collected.records)
    assert all(r.true_label is not None for r in collected.records)
    assert all(r.samples_per_symbol == 8 for r in collected.records)


def test_an_absent_capture_directory_is_reported_not_raised(tmp_path) -> None:
    class _Stub:
        checkpoint_path, checkpoint_sha256 = "stub", "0" * 8

    collected = collect_v1_captures(_Stub(), root=tmp_path)

    assert not collected.ok
    assert collected.warnings


# --- projection --------------------------------------------------------------------------------------


def test_pca_is_deterministic_and_reports_its_variance() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(120, 32))

    first = fit_pca(vectors, 3)
    second = fit_pca(vectors, 3)

    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.coordinates.shape == (120, 3)
    assert first.explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < first.shown_variance <= 1.0
    assert first.components_for(0.90) >= 3


def test_pca_transform_places_new_points_on_the_same_axes() -> None:
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(60, 16))
    projection = fit_pca(vectors, 3)

    reprojected = projection.transform(vectors)

    assert np.allclose(reprojected, projection.coordinates, atol=1e-9)
    assert projection.transform(vectors[0]).shape == (1, 3)


def test_pca_refuses_degenerate_input() -> None:
    assert fit_pca(np.empty((0, 0))) is None
    assert fit_pca(np.ones((1, 8))) is None
    assert fit_pca(np.ones((5,))) is None


def test_pca_reports_the_component_count_it_could_have_used() -> None:
    rng = np.random.default_rng(2)
    projection = fit_pca(rng.normal(size=(50, 20)), 3)

    assert projection.total_components == 20
    assert projection.explained_variance_ratio.size == 20
    assert projection.components.shape == (3, 20)


# --- filtering ------------------------------------------------------------------------------------------


def _records(count: int = 6):
    rng = np.random.default_rng(3)
    made = []
    for index in range(count):
        made.append(EmbeddingRecord(
            vector=rng.normal(size=8).astype(np.float32),
            capture_id=f"c{index}", window_index=0,
            predicted_label="QPSK" if index % 2 else "BPSK",
            confidence=0.9, source=SOURCE_SYNTHETIC if index < 4 else SOURCE_REAL,
            true_label="QPSK" if index < 3 else ("BPSK" if index < 5 else None),
            samples_per_symbol=8 if index < 3 else 32, snr_db=20.0 if index < 2 else 0.0))
    return made


class _FakeModel:
    checkpoint_path, checkpoint_sha256 = "fake", "abc123"

    def width(self, layer):
        return 8


def _set():
    return build_set(_records(), _FakeModel(), DEFAULT_LAYER, 1)


def test_filtering_by_class_keeps_vectors_and_metadata_aligned() -> None:
    filtered = filter_set(_set(), true_label="QPSK")

    assert len(filtered) == 3
    assert filtered.vectors.shape[0] == 3
    assert all(r.true_label == "QPSK" for r in filtered.records)


def test_filtering_by_correctness_excludes_unlabelled_captures() -> None:
    """`correct` is None without ground truth, so such records match neither filter."""
    source = _set()

    correct = filter_set(source, correct=True)
    incorrect = filter_set(source, correct=False)

    assert all(r.true_label is not None for r in correct.records)
    assert all(r.true_label is not None for r in incorrect.records)
    assert len(correct) + len(incorrect) < len(source)


def test_filtering_by_source_separates_real_from_synthetic() -> None:
    assert len(filter_set(_set(), source=SOURCE_REAL)) == 2
    assert len(filter_set(_set(), source=SOURCE_SYNTHETIC)) == 4


def test_filtering_accepts_a_collection_of_values() -> None:
    assert len(filter_set(_set(), samples_per_symbol={8, 32})) == 6
    assert len(filter_set(_set(), samples_per_symbol={32})) == 3


def test_a_filter_matching_nothing_returns_an_explained_empty_set() -> None:
    filtered = filter_set(_set(), true_label="NOT-A-CLASS")

    assert not filtered.ok
    assert filtered.warnings


# --- geometry helpers -------------------------------------------------------------------------------------


def test_class_separation_reports_overlap_and_needs_two_classes() -> None:
    measured = class_separation(_set(), "QPSK", "BPSK")

    assert measured is not None
    assert measured.count_left == 3 and measured.count_right == 2
    assert 0.0 <= measured.overlap <= 1.0
    assert measured.centroid_distance >= 0.0
    assert class_separation(_set(), "QPSK", "NOT-A-CLASS") is None


def test_centroids_ignore_unlabelled_records() -> None:
    centroids = nearest_class_centroids(_set())

    assert set(centroids) == {"QPSK", "BPSK"}
    assert all(centre.shape == (8,) for centre in centroids.values())


def test_prototype_distances_are_sorted_and_named() -> None:
    source = _set()
    centroids = nearest_class_centroids(source)

    distances = centroid_distances(source, source.vectors[0], centroids)

    assert [name for name, _ in distances] == sorted(
        centroids, key=lambda k: float(np.linalg.norm(
            source.vectors[0].astype(np.float64) - centroids[k].astype(np.float64))))
    assert distances == sorted(distances, key=lambda item: item[1])
    assert centroid_distances(source, source.vectors[0], {}) == []


# --- caching identity ---------------------------------------------------------------------------------------


def test_the_fingerprint_changes_with_the_things_that_matter() -> None:
    base = _set()
    other_layer = build_set(_records(), _FakeModel(), POOLED, 1)
    other_windows = build_set(_records(), _FakeModel(), DEFAULT_LAYER, 4)
    fewer = build_set(_records(3), _FakeModel(), DEFAULT_LAYER, 1)

    assert base.fingerprint() == _set().fingerprint(), "same inputs, same identity"
    assert base.fingerprint() != other_layer.fingerprint()
    assert base.fingerprint() != other_windows.fingerprint()
    assert base.fingerprint() != fewer.fingerprint()


def test_an_empty_set_is_safe_to_inspect() -> None:
    empty = build_set([], _FakeModel(), DEFAULT_LAYER, 4)

    assert not empty.ok
    assert len(empty) == 0
    assert empty.width == 0
    assert empty.warnings
    assert filter_set(empty, true_label="QPSK") is empty
