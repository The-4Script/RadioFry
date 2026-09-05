"""Train the explainable FEC-scheme classifier from a manifest."""

import csv
import pickle
from pathlib import Path

import numpy as np

from radiofry.synthetic_gen.features import bit_features


def train_fec(manifest_path: str | Path, output_path: str | Path) -> dict[str, float]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    manifest = Path(manifest_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    features = np.vstack([bit_features(np.load(manifest.parent / row["filename"])) for row in rows])
    labels = np.array([row["fec_type"] for row in rows])
    test_fraction = max(0.2, len(np.unique(labels)) / len(labels))
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=test_fraction, random_state=7, stratify=labels)
    model = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=-1).fit(train_x, train_y)
    accuracy = float(accuracy_score(test_y, model.predict(test_x)))
    with Path(output_path).open("wb") as handle:
        pickle.dump(model, handle)
    return {"accuracy": accuracy, "samples": float(len(rows))}
