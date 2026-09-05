from pathlib import Path

import numpy as np

from radiofry.models.bitstream_inference import predict_bitstream


def test_saved_interleaver_classifier_predicts_a_label() -> None:
    model_path = Path("models_saved/interleaver_classifier.pkl")
    if not model_path.exists():
        return
    bits = np.resize(np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8), 128)

    result = predict_bitstream(bits, model_path)

    assert result.available
    assert result.label in {"none", "block", "diagonal", "pseudo_random"}
    assert 0 <= result.confidence <= 1