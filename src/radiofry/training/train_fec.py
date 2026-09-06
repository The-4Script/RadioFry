"""Train the explainable FEC-scheme classifier from a manifest."""

import csv
import json
import pickle
from pathlib import Path

import numpy as np

from radiofry.synthetic_gen.features import bit_features
from radiofry.models.artifact_integrity import hash_bytes, serialize_sklearn_model


def train_fec(manifest_path: str | Path, output_path: str | Path, *, seed: int = 7) -> dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    manifest = Path(manifest_path)
    rows = [row for row in csv.DictReader(manifest.open(encoding="utf-8")) if row.get("filename")]
    if not rows:
        raise ValueError(f"Manifest contains no usable samples: {manifest}")
    features = np.vstack([bit_features(np.load(manifest.parent / row["filename"])) for row in rows])
    labels = np.array([row["fec_type"] for row in rows])
    class_names = sorted(np.unique(labels).tolist())
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=seed, stratify=labels)
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(train_x, train_y)
    predictions = model.predict(test_x)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1), features, labels, cv=folds, scoring="accuracy", n_jobs=1)
    model_bytes = serialize_sklearn_model(model)
    model_sha256 = hash_bytes(model_bytes)
    metrics: dict[str, object] = {
        "model_sha256": model_sha256,
        "samples": int(len(rows)),
        "classes": class_names,
        "feature_count": int(features.shape[1]),
        "test_accuracy": float(accuracy_score(test_y, predictions)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=class_names).tolist(),
        "feature_importances": model.feature_importances_.tolist(),
        "seed": seed,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump({"model_bytes": model_bytes, "model_sha256": model_sha256}, handle)
    output.with_name(f"{output.stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the RadioFry FEC classifier")
    parser.add_argument("--manifest", default="data/synthetic/fec/fec_manifest.csv")
    parser.add_argument("--output", default="models_saved/fec_classifier.pkl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(train_fec(args.manifest, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
