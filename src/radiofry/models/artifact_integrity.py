"""Deterministic identity hashes for trusted inference artifacts."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import pickle
from pathlib import Path
from typing import Any


def hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def hash_torch_state_dict(state_dict: dict[str, Any]) -> str:
    import torch

    buffer = BytesIO()
    torch.save(state_dict, buffer)
    return hash_bytes(buffer.getvalue())


def hash_sklearn_model(model: Any) -> str:
    return hash_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def serialize_sklearn_model(model: Any) -> bytes:
    return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)


def metrics_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_name(f"{path.stem}_metrics.json")