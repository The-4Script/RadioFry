"""Collect CNN embeddings over known data, and project them for viewing.

Two collection sources, both reproducible:

* **Frozen V1 captures** - the 40 `.iq` files under `data/synthetic_v1/captures`, read
  from disk with their ground-truth sidecars. Validated and byte-frozen, but every capture
  is at samples-per-symbol 8, so this set cannot answer questions about SPS.
* **The benchmark grid** - regenerated in memory from the V1 generator using the Entry 039
  seeds (601 / 607 / 613) across samples-per-symbol 4/8/16/32 and SNR 20/15/10/5/0. This
  is not a new dataset: the same generator functions reproduce the frozen captures to
  `|corr| = 1.0` (the residual is the int16 quantisation of the stored files), which is
  asserted by a test. Regenerating is what makes SPS and SNR available as axes.

Projection
----------
PCA only, by SVD, with the sign convention fixed so repeated runs are bit-identical. UMAP
was considered and deliberately not added: `umap-learn` is a heavy dependency (numba,
llvmlite) that is not declared anywhere in this project, its default embedding is
stochastic, and nothing here needs a nonlinear projection to answer the questions asked.

PCA is a **linear projection fitted to the data being shown**. It is a view, not a model:
it is not trained, it predicts nothing, and no classifier is involved. Distances in the
3-D projection are distances in a 3-D shadow of a 256-D space - useful for seeing whether
classes separate at all, and not a metric for how similar two signals "really" are.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radiofry.models.embedding import (
    DEFAULT_LAYER,
    DEFAULT_WINDOWS,
    SOURCE_SYNTHETIC,
    EmbeddingModel,
    EmbeddingRecord,
    EmbeddingSet,
    build_set,
    embed_signal,
)

V1_CAPTURES = Path("data/synthetic_v1/captures")

# The Entry 039 benchmark grid, recorded in reports/benchmark_v039/config.json.
BENCHMARK_SEEDS = (601, 607, 613)
BENCHMARK_SPS = (4, 8, 16, 32)
BENCHMARK_SNR_DB = (20.0, 15.0, 10.0, 5.0, 0.0)
BENCHMARK_SAMPLES = 8192
BENCHMARK_SAMPLE_RATE = 200_000.0

# generator name -> production label. BFSK and CPFSK are the same signal (Entry 039).
GENERATOR_TO_LABEL = {
    "BPSK": "BPSK", "QPSK": "QPSK", "8PSK": "8PSK", "BFSK": "CPFSK",
    "GFSK": "GFSK", "PAM4": "PAM4", "16QAM": "QAM16", "64QAM": "QAM64",
}


@dataclass(frozen=True)
class PCAProjection:
    """A linear projection of an embedding set, plus everything needed to judge it."""

    coordinates: np.ndarray              # (n, components)
    explained_variance_ratio: np.ndarray  # per retained component
    mean: np.ndarray
    components: np.ndarray               # (components, width)
    fitted_on: str
    total_components: int

    @property
    def shown_variance(self) -> float:
        return float(np.sum(self.explained_variance_ratio[:3]))

    def components_for(self, fraction: float) -> int:
        """How many components would be needed to reach `fraction` of the variance."""
        cumulative = np.cumsum(self.explained_variance_ratio)
        reached = np.searchsorted(cumulative, fraction) + 1
        return int(min(reached, self.explained_variance_ratio.size))

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the *same* fitted projection to new vectors, so views stay comparable."""
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        return (values - self.mean) @ self.components.T


def fit_pca(vectors: np.ndarray, components: int = 3,
            fitted_on: str = "the displayed embedding set") -> PCAProjection | None:
    """Deterministic PCA by SVD. Returns None when there is nothing to fit.

    The sign of each component is fixed by forcing the largest-magnitude loading positive.
    Without that, SVD may return either sign run to run and the plot would mirror itself
    for no reason.
    """

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        return None
    wanted = int(min(components, values.shape[0] - 1, values.shape[1]))
    if wanted < 1:
        return None

    mean = values.mean(axis=0)
    centred = values - mean
    _, singular, right = np.linalg.svd(centred, full_matrices=False)

    # Deterministic sign convention (the same one scikit-learn's svd_flip applies).
    for row in range(right.shape[0]):
        largest = int(np.argmax(np.abs(right[row])))
        if right[row, largest] < 0:
            right[row] *= -1.0

    variance = singular ** 2
    total = float(variance.sum())
    ratio = variance / total if total > 0 else np.zeros_like(variance)

    kept = right[:wanted]
    return PCAProjection(
        coordinates=centred @ kept.T,
        explained_variance_ratio=ratio,
        mean=mean,
        components=kept,
        fitted_on=fitted_on,
        total_components=int(singular.size),
    )


def _load_v1_truth(stem: str) -> dict | None:
    import json

    path = V1_CAPTURES / f"{stem}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_v1_captures(
    loaded: EmbeddingModel,
    *,
    layer: str = DEFAULT_LAYER,
    windows: int = DEFAULT_WINDOWS,
    root: Path | None = None,
    limit: int | None = None,
) -> EmbeddingSet:
    """Embed the frozen V1 dataset, reading the captures and their truth from disk."""

    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import load_capture

    captures = sorted((root or V1_CAPTURES).glob("*.iq"))
    if limit is not None:
        captures = captures[:limit]
    if not captures:
        return build_set([], loaded, layer, windows,
                         ("The frozen V1 capture set is not present in this checkout.",))

    records: list[EmbeddingRecord] = []
    warnings: list[str] = []
    for path in captures:
        truth = _load_v1_truth(path.stem)
        if truth is None:
            warnings.append(f"{path.stem}: no ground-truth sidecar; skipped.")
            continue
        signal_meta = truth.get("signal", {})
        entry = next((f for f in truth.get("files", [])
                      if f.get("file_format") == "iq"), None)
        if entry is None:
            warnings.append(f"{path.stem}: sidecar lists no IQ file; skipped.")
            continue
        try:
            signal = load_capture(
                path, sample_rate=signal_meta.get("sample_rate_hz"),
                iq_format=IQFormat(entry["dtype"], entry["byte_order"]))
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{path.stem}: could not be read ({error}); skipped.")
            continue
        records.extend(embed_signal(
            loaded, signal, capture_id=path.stem, layer=layer, windows=windows,
            source=SOURCE_SYNTHETIC,
            true_label=truth.get("modulation", {}).get("radiofry_label"),
            samples_per_symbol=signal_meta.get("samples_per_symbol"),
            snr_db=truth.get("noise", {}).get("target_snr_db"),
            seed=truth.get("seeds", {}).get("seed")))
    return build_set(records, loaded, layer, windows, tuple(warnings))


def generate_grid_capture(generator_name: str, samples_per_symbol: int,
                          snr_db: float, seed: int,
                          samples: int = BENCHMARK_SAMPLES,
                          sample_rate_hz: float = BENCHMARK_SAMPLE_RATE):
    """One benchmark-grid capture, in memory, by the same path that wrote frozen V1.

    `generate_source_bits` -> `modulate` -> `add_awgn` with the noise stream seeded as
    `default_rng([seed, 2])` is exactly what `generate_sample` does before it writes files.
    """

    from radiofry.contracts import UnifiedSignalContainer
    from radiofry.synthetic_gen.v1.channel import add_awgn
    from radiofry.synthetic_gen.v1.config import SampleSpec
    from radiofry.synthetic_gen.v1.modulation import generate_source_bits, modulate

    spec = SampleSpec(
        modulation=generator_name,
        num_symbols=max(1, samples // samples_per_symbol),
        samples_per_symbol=samples_per_symbol,
        sample_rate_hz=sample_rate_hz,
        snr_db=float(snr_db),
        seed=int(seed))
    bits = generate_source_bits(spec)
    clean = modulate(bits, spec)
    noisy, _ = add_awgn(clean, spec.snr_db, np.random.default_rng([spec.seed, 2]))
    return UnifiedSignalContainer(np.asarray(noisy), sample_rate_hz, "iq")


def collect_benchmark_grid(
    loaded: EmbeddingModel,
    *,
    layer: str = DEFAULT_LAYER,
    windows: int = DEFAULT_WINDOWS,
    generators: tuple[str, ...] = tuple(GENERATOR_TO_LABEL),
    samples_per_symbol: tuple[int, ...] = BENCHMARK_SPS,
    snr_db: tuple[float, ...] = BENCHMARK_SNR_DB,
    seeds: tuple[int, ...] = BENCHMARK_SEEDS,
) -> EmbeddingSet:
    """Embed the regenerated Entry 039 grid.

    Deterministic: the same arguments always produce the same vectors, because the
    generator is seeded and the model runs in eval/inference mode.
    """

    records: list[EmbeddingRecord] = []
    warnings: list[str] = []
    for generator_name in generators:
        label = GENERATOR_TO_LABEL.get(generator_name)
        if label is None:
            warnings.append(f"{generator_name}: not a benchmark class; skipped.")
            continue
        for sps in samples_per_symbol:
            for snr in snr_db:
                for seed in seeds:
                    capture_id = f"{generator_name}_sps{sps}_snr{snr:g}dB_s{seed}"
                    try:
                        signal = generate_grid_capture(generator_name, sps, snr, seed)
                    except (ValueError, KeyError) as error:
                        warnings.append(f"{capture_id}: {error}; skipped.")
                        continue
                    records.extend(embed_signal(
                        loaded, signal, capture_id=capture_id, layer=layer,
                        windows=windows, source=SOURCE_SYNTHETIC, true_label=label,
                        samples_per_symbol=sps, snr_db=float(snr), seed=seed))
    return build_set(records, loaded, layer, windows, tuple(warnings))


def filter_set(embeddings: EmbeddingSet, **criteria) -> EmbeddingSet:
    """Subset by metadata, keeping vectors and records aligned.

    Supported keys: `true_label`, `predicted_label`, `samples_per_symbol`, `snr_db`,
    `source`, `correct`, `window_index`. Each takes a value or a collection of values.

    The positional parameter is `embeddings`, not `source`, because `source` is itself
    one of the filterable fields - naming them the same made filtering by source raise
    "multiple values for argument".
    """

    if not embeddings.ok:
        return embeddings

    def matches(record: EmbeddingRecord) -> bool:
        for name, wanted in criteria.items():
            if wanted is None:
                continue
            actual = getattr(record, name, None) if name != "correct" else record.correct
            if isinstance(wanted, (set, frozenset, list, tuple)):
                if actual not in wanted:
                    return False
            elif actual != wanted:
                return False
        return True

    kept = [record for record in embeddings.records if matches(record)]
    if not kept:
        return EmbeddingSet(np.empty((0, 0), dtype=np.float32), (), embeddings.layer,
                            embeddings.checkpoint_path, embeddings.checkpoint_sha256,
                            embeddings.windows,
                            ("No embeddings match the selected filters.",))
    return EmbeddingSet(
        np.stack([record.vector for record in kept]).astype(np.float32),
        tuple(kept), embeddings.layer, embeddings.checkpoint_path,
        embeddings.checkpoint_sha256, embeddings.windows, embeddings.warnings)


@dataclass(frozen=True)
class ClassSeparation:
    """How far apart two classes sit in the raw embedding space.

    Reported in the **raw** space, not the PCA projection, so it does not depend on how
    many components happen to be displayed. `overlap` is the fraction of one class's
    members whose nearest class centroid is the other class - a concrete, checkable
    statement, not a probability.
    """

    left: str
    right: str
    centroid_distance: float
    left_spread: float
    right_spread: float
    overlap: float
    count_left: int
    count_right: int

    @property
    def separation_ratio(self) -> float:
        """Centroid distance relative to the classes' own spread. Below ~1 is overlap."""
        spread = (self.left_spread + self.right_spread) / 2.0
        return float(self.centroid_distance / spread) if spread > 0 else float("inf")


def class_separation(source: EmbeddingSet, left: str, right: str,
                     use_true_label: bool = True) -> ClassSeparation | None:
    """Measure the geometry between two labelled classes in the raw embedding space."""

    field_name = "true_label" if use_true_label else "predicted_label"
    left_mask = np.array([getattr(r, field_name) == left for r in source.records])
    right_mask = np.array([getattr(r, field_name) == right for r in source.records])
    if left_mask.sum() < 2 or right_mask.sum() < 2:
        return None

    left_points = source.vectors[left_mask].astype(np.float64)
    right_points = source.vectors[right_mask].astype(np.float64)
    left_centre = left_points.mean(axis=0)
    right_centre = right_points.mean(axis=0)
    distance = float(np.linalg.norm(left_centre - right_centre))

    def spread(points, centre):
        return float(np.mean(np.linalg.norm(points - centre, axis=1)))

    misassigned = 0
    for points, own, other in ((left_points, left_centre, right_centre),
                               (right_points, right_centre, left_centre)):
        own_distance = np.linalg.norm(points - own, axis=1)
        other_distance = np.linalg.norm(points - other, axis=1)
        misassigned += int(np.count_nonzero(other_distance < own_distance))

    return ClassSeparation(
        left=left, right=right, centroid_distance=distance,
        left_spread=spread(left_points, left_centre),
        right_spread=spread(right_points, right_centre),
        overlap=misassigned / float(len(left_points) + len(right_points)),
        count_left=int(len(left_points)), count_right=int(len(right_points)))


def nearest_class_centroids(source: EmbeddingSet) -> dict[str, np.ndarray]:
    """Centroid of each labelled class in the raw embedding space."""

    groups: dict[str, list[np.ndarray]] = {}
    for index, record in enumerate(source.records):
        if record.true_label is None:
            continue
        groups.setdefault(record.true_label, []).append(source.vectors[index])
    return {label: np.mean(np.stack(points), axis=0)
            for label, points in groups.items() if points}


def centroid_distances(source: EmbeddingSet, vector: np.ndarray,
                       centroids: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """Euclidean distance from one embedding to each class centroid, nearest first.

    This is a *prototype distance in the raw embedding space* and nothing more. It is not
    an out-of-distribution probability and has not been validated as a detector; a large
    distance says the vector sits far from every class centre this set contains, which is
    a prompt to look, not a verdict.
    """

    if not centroids:
        return []
    point = np.asarray(vector, dtype=np.float64)
    scored = [(label, float(np.linalg.norm(point - centre.astype(np.float64))))
              for label, centre in centroids.items()]
    return sorted(scored, key=lambda item: item[1])
