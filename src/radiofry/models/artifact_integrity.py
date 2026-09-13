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
    """Hash of the bytes `torch.save` produces for this state dict.

    **This is not portable across environments.** It hashes the serialised container, so
    the result depends on the torch version and the pickle protocol doing the serialising,
    not only on the tensor values. Two machines holding a byte-identical checkpoint file
    can disagree on this hash - observed directly in CI, where the checkout was
    byte-identical to the local file yet this returned `0365780e...` against the local
    `a7b02533...`.

    Kept because existing checkpoints and metrics files store values produced by it. For
    an identity that survives a change of environment use `hash_state_dict_contents`.
    """

    import torch

    buffer = BytesIO()
    torch.save(state_dict, buffer)
    return hash_bytes(buffer.getvalue())


def hash_state_dict_contents(state_dict: dict[str, Any]) -> str:
    """Serialisation-independent identity of the tensor values themselves.

    Hashes sorted parameter names with their dtype, shape and raw little-endian bytes, so
    the result depends on the weights and nothing else. Unchanged by the torch version,
    the pickle protocol, or a save/load round trip - which is what makes it usable as a
    freeze assertion that holds on any machine.
    """

    import sys

    import torch

    digest = sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        array = tensor.contiguous().numpy()
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype.str).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def hash_sklearn_model(model: Any) -> str:
    return hash_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def serialize_sklearn_model(model: Any) -> bytes:
    return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)


def metrics_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_name(f"{path.stem}_metrics.json")