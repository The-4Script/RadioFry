"""Read the production CNN's internal representation without changing what it predicts.

"Signal-DNA" is the investigator-facing name. Technically this extracts an **internal
activation of the current production checkpoint** for a capture, so that captures can be
compared in the space the classifier actually uses rather than by their final labels.

Which tensor, and why
---------------------
`ModulationCNN` is:

    features   = [Conv 4->64, Conv 64->128, Conv 128->128, AdaptiveAvgPool1d(1)]
    classifier = [Flatten, Linear(128->256), ReLU, Dropout(0.5), Linear(256->num_classes)]

Two internal representations are available, and both are offered:

* **`penultimate` (256-d, the default)** - the output of `classifier[:4]`, i.e. after
  `Linear(128->256)` and its ReLU. This is *literally the vector the final linear layer
  reads*: the logits are an affine map of it. Two classes that overlap here must be
  confusable, because no linear boundary can separate what overlaps. That direct link to
  the decision is why it is the default for diagnosing confusions.
* **`pooled` (128-d)** - the output of `features`, flattened. The convolutional stack's
  own global-average-pooled summary, before any classification-specific projection. Less
  class-specialised, and useful when the question is about the signal rather than about
  the decision boundary.

The softmax output is deliberately NOT available as an embedding. It is 8 numbers
constrained to a simplex; distances in it describe the decision, not the representation,
and a test asserts no layer returns `num_classes` dimensions.

Temporal aggregation
--------------------
No aggregation is invented here. The model already reduces time itself, with
`AdaptiveAvgPool1d(1)` at the end of the convolutional stack, so both representations are
fixed-dimensional for any input length. Production classifies four evenly spaced 128-sample
windows and averages their *softmax* vectors; this module returns **one record per window**
so that averaging is the caller's explicit choice. `aggregate_windows` offers the mean of
window embeddings, which is a different operation from production's mean-softmax and is
labelled as such.

Honesty rules
-------------
* The checkpoint is loaded with the same integrity checks production uses, and the same
  preprocessing path (`_window_frames` + `add_signal_features`) is called - not a copy of
  it. A test asserts that re-deriving softmax from these logits reproduces
  `predict_modulation` exactly.
* Metadata is carried, never inferred. A capture with no ground truth is labelled
  `None`, and the UI shows "Unknown / unlabelled"; the CNN's own prediction is never
  promoted into the ground-truth field.
* Extraction is deterministic: `eval()` disables dropout and batch-norm updates, and
  `inference_mode` disables autograd.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from radiofry.contracts import UnifiedSignalContainer

# name -> (how to build it, expected width for the production checkpoint)
PENULTIMATE = "penultimate"
POOLED = "pooled"
EMBEDDING_LAYERS = (PENULTIMATE, POOLED)

DEFAULT_LAYER = PENULTIMATE
DEFAULT_WINDOWS = 4

SOURCE_SYNTHETIC = "synthetic"
SOURCE_REAL = "real"
SOURCE_UNKNOWN = "unknown"
SOURCES = (SOURCE_SYNTHETIC, SOURCE_REAL, SOURCE_UNKNOWN)

UNLABELLED = "Unknown / unlabelled"


@dataclass(frozen=True)
class EmbeddingModel:
    """A loaded production checkpoint, held open so it is not re-read per capture."""

    model: Any
    labels: tuple[str, ...]
    sample_length: int
    engineered: bool
    input_channels: int
    checkpoint_path: str
    checkpoint_sha256: str

    def width(self, layer: str) -> int:
        return 256 if layer == PENULTIMATE else 128


@dataclass(frozen=True)
class EmbeddingRecord:
    """One window of one capture, in the CNN's internal space, plus its provenance."""

    vector: np.ndarray
    capture_id: str
    window_index: int
    predicted_label: str
    confidence: float
    source: str = SOURCE_UNKNOWN
    true_label: str | None = None
    samples_per_symbol: int | None = None
    snr_db: float | None = None
    sample_rate_hz: float | None = None
    seed: int | None = None

    @property
    def labelled(self) -> bool:
        return self.true_label is not None

    @property
    def correct(self) -> bool | None:
        """None when there is no ground truth - never guessed from the prediction."""
        if self.true_label is None:
            return None
        return self.predicted_label == self.true_label

    @property
    def display_true_label(self) -> str:
        return self.true_label if self.true_label is not None else UNLABELLED


@dataclass(frozen=True)
class EmbeddingSet:
    """A collected embedding dataset: vectors plus aligned metadata."""

    vectors: np.ndarray                  # (n, width)
    records: tuple[EmbeddingRecord, ...]
    layer: str
    checkpoint_path: str
    checkpoint_sha256: str
    windows: int
    warnings: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return self.vectors.size > 0

    def __len__(self) -> int:
        return len(self.records)

    @property
    def width(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0

    def column(self, name: str) -> list:
        return [getattr(record, name) for record in self.records]

    def fingerprint(self) -> str:
        """Identity of (checkpoint, layer, windows, dataset), for caching."""
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.checkpoint_sha256.encode())
        digest.update(self.layer.encode())
        digest.update(str(self.windows).encode())
        for record in self.records:
            digest.update(record.capture_id.encode())
            digest.update(str(record.window_index).encode())
        return digest.hexdigest()[:16]


def load_embedding_model(
    checkpoint_path: str | Path,
) -> tuple[EmbeddingModel | None, str]:
    """Load the production checkpoint the way production loads it.

    Returns `(model, "")` on success or `(None, reason)` on failure - never raises, so a
    missing or mismatched checkpoint degrades the feature instead of the application.
    """

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None, f"Checkpoint not found: {checkpoint}"
    try:
        import torch

        from radiofry.models.artifact_integrity import hash_state_dict_contents
        from radiofry.models.modulation_cnn import ModulationCNN

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actual_hash = hash_state_dict_contents(payload["state_dict"])
        labels = tuple(str(name) for name in payload["labels"])
        model = ModulationCNN(int(payload.get("input_channels", 2)), len(labels))
        model.load_state_dict(payload["state_dict"])
        # eval() is what makes this deterministic: dropout becomes identity and the
        # batch-norm layers use their stored running statistics rather than the batch.
        model.eval()
        return EmbeddingModel(
            model=model,
            labels=labels,
            sample_length=int(payload.get("sample_length", 128)),
            engineered=payload.get("features", "iq") == "iqap",
            input_channels=int(payload.get("input_channels", 2)),
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=str(actual_hash),
        ), ""
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        return None, f"Could not load checkpoint: {error}"


def _forward(loaded: EmbeddingModel, batch: np.ndarray, layer: str):
    """Internal activation and logits for a batch, in one pass, without autograd.

    The submodules are called directly rather than through a hook: the same objects the
    prediction path uses, read in the same order, with nothing registered on the model
    that could outlive this call and change inference elsewhere.
    """

    import torch

    if layer not in EMBEDDING_LAYERS:
        raise ValueError(f"layer must be one of {EMBEDDING_LAYERS}")
    model = loaded.model
    with torch.inference_mode():
        tensor = torch.from_numpy(batch)
        pooled = model.features(tensor)                 # (n, 128, 1)
        logits = model.classifier(pooled)               # (n, num_classes)
        if layer == POOLED:
            activation = torch.flatten(pooled, 1)       # (n, 128)
        else:
            # classifier[:4] = Flatten, Linear(128->256), ReLU, Dropout(identity in eval)
            activation = model.classifier[:4](pooled)   # (n, 256)
        return activation.numpy().copy(), logits.numpy().copy()


def embed_signal(
    loaded: EmbeddingModel,
    signal: UnifiedSignalContainer,
    *,
    capture_id: str,
    layer: str = DEFAULT_LAYER,
    windows: int = DEFAULT_WINDOWS,
    source: str = SOURCE_UNKNOWN,
    true_label: str | None = None,
    samples_per_symbol: int | None = None,
    snr_db: float | None = None,
    seed: int | None = None,
) -> list[EmbeddingRecord]:
    """One `EmbeddingRecord` per inference window, using the production feature path.

    `predicted_label` and `confidence` are derived from the same logits the embedding came
    from, averaged over windows exactly as `predict_modulation` averages softmax - so the
    label attached to every window of a capture is the capture's production decision, not
    a per-window guess.
    """

    from radiofry.models.modulation_inference import _window_frames
    from radiofry.models.signal_features import add_signal_features

    if signal.iq.size == 0:
        return []
    frames = _window_frames(signal, loaded.sample_length, windows)
    batch = np.stack([add_signal_features(frame, include_engineered=loaded.engineered)
                      for frame in frames])
    activation, logits = _forward(loaded, batch, layer)

    # Same reduction production uses: softmax per window, then mean across windows.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = (exponentiated / exponentiated.sum(axis=1, keepdims=True)).mean(axis=0)
    best = int(np.argmax(probabilities))

    return [
        EmbeddingRecord(
            vector=activation[index].astype(np.float32),
            capture_id=capture_id,
            window_index=index,
            predicted_label=loaded.labels[best],
            confidence=float(probabilities[best]),
            source=source if source in SOURCES else SOURCE_UNKNOWN,
            true_label=true_label,
            samples_per_symbol=samples_per_symbol,
            snr_db=snr_db,
            sample_rate_hz=signal.sample_rate,
            seed=seed,
        )
        for index in range(activation.shape[0])
    ]


def aggregate_windows(records: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
    """Collapse each capture's windows to their mean embedding.

    This is the mean of the window *embeddings*, which is not what production does - it
    averages softmax vectors. The metadata is carried from the first window and the
    window index becomes -1 to mark the record as aggregated rather than observed.
    """

    by_capture: dict[str, list[EmbeddingRecord]] = {}
    for record in records:
        by_capture.setdefault(record.capture_id, []).append(record)

    aggregated = []
    for capture_id, group in by_capture.items():
        mean = np.mean(np.stack([item.vector for item in group]), axis=0)
        first = group[0]
        aggregated.append(EmbeddingRecord(
            vector=mean.astype(np.float32),
            capture_id=capture_id,
            window_index=-1,
            predicted_label=first.predicted_label,
            confidence=first.confidence,
            source=first.source,
            true_label=first.true_label,
            samples_per_symbol=first.samples_per_symbol,
            snr_db=first.snr_db,
            sample_rate_hz=first.sample_rate_hz,
            seed=first.seed,
        ))
    return aggregated


def build_set(records: list[EmbeddingRecord], loaded: EmbeddingModel, layer: str,
              windows: int, warnings: tuple[str, ...] = ()) -> EmbeddingSet:
    """Stack records into an `EmbeddingSet`, keeping vectors and metadata aligned."""

    if not records:
        return EmbeddingSet(np.empty((0, 0), dtype=np.float32), (), layer,
                            loaded.checkpoint_path, loaded.checkpoint_sha256, windows,
                            warnings or ("No embeddings were produced.",))
    vectors = np.stack([record.vector for record in records]).astype(np.float32)
    return EmbeddingSet(vectors, tuple(records), layer, loaded.checkpoint_path,
                        loaded.checkpoint_sha256, windows, warnings)
