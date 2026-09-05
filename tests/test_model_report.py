import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from radiofry.models.modulation_cnn import ModulationCNN
from radiofry.reporting.report_builder import build_report, report_json


def test_modulation_cnn_output_shape() -> None:
    model = ModulationCNN(input_channels=2, num_classes=11)
    output = model(torch.zeros(4, 2, 128))

    assert output.shape == (4, 11)


def test_report_serializes_dataclasses_and_numpy_values() -> None:
    report = build_report(source={"samples": np.int64(8)}, stages={"scores": np.array([0.2, 0.8])})

    loaded = json.loads(report_json(report))
    assert loaded["source"]["samples"] == 8
    assert loaded["stages"]["scores"] == [0.2, 0.8]