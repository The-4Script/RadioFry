"""Trainable model definitions with optional ML dependencies."""

from .modulation_inference import ModulationPrediction, predict_modulation
from .bitstream_inference import BitstreamPrediction, predict_bitstream

__all__ = ["ModulationPrediction", "predict_modulation", "BitstreamPrediction", "predict_bitstream"]
