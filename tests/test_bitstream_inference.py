from pathlib import Path
import json
import pickle

import numpy as np

from radiofry.models.bitstream_inference import predict_bitstream
from radiofry.models.artifact_integrity import hash_bytes, serialize_sklearn_model
from radiofry.synthetic_gen.generate_dataset import generate_interleaver_dataset


def test_saved_interleaver_classifier_predicts_a_label() -> None:
    model_path = Path("models_saved/interleaver_classifier.pkl")
    if not model_path.exists():
        return
    bits = np.resize(np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8), 128)

    result = predict_bitstream(bits, model_path)

    assert result.available
    assert result.label in {"none", "block", "convolutional", "diagonal", "pseudo_random"}
    assert 0 <= result.confidence <= 1


def test_interleaver_generator_includes_all_architecture_classes(tmp_path: Path) -> None:
    manifest = generate_interleaver_dataset(tmp_path, examples_per_class=2, length=128)
    labels = {row.split(",")[1] for row in manifest.read_text(encoding="utf-8").splitlines()[1:]}

    assert labels == {"none", "block", "convolutional", "diagonal", "pseudo_random"}


def test_bitstream_inference_reports_feature_mismatch(tmp_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=1, random_state=7).fit(
        np.zeros((4, 999), dtype=np.float32), ["none", "none", "block", "block"]
    )

    model_path = tmp_path / "classifier.pkl"
    model_bytes = serialize_sklearn_model(model)
    model_hash = hash_bytes(model_bytes)
    with model_path.open("wb") as handle:
        pickle.dump({"model_bytes": model_bytes, "model_sha256": model_hash}, handle)
    model_path.with_name("classifier_metrics.json").write_text(json.dumps({"model_sha256": model_hash}), encoding="utf-8")

    result = predict_bitstream(np.zeros(128, dtype=np.uint8), model_path)

    assert not result.available
    assert "feature mismatch" in result.message