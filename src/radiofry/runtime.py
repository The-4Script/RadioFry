"""Runtime asset checks for the inference-only application."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuntimeArtifact:
    name: str
    path: str
    required: bool
    available: bool
    size_bytes: int
    message: str = ""


def resolve_runtime_path(path: str | Path) -> Path:
    """Resolve a relative runtime asset path from the repository root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def check_runtime_artifacts(paths: Mapping[str, str | Path]) -> dict[str, object]:
    """Return JSON-friendly availability diagnostics for inference artifacts."""

    artifacts: list[RuntimeArtifact] = []
    for name, configured_path in paths.items():
        path = resolve_runtime_path(configured_path)
        required = name in {"modulation", "interleaver", "fec"}
        if not path.is_file():
            artifacts.append(RuntimeArtifact(name, str(path), required, False, 0, "file not found"))
            continue
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            artifacts.append(RuntimeArtifact(name, str(path), required, False, 0, "file is empty"))
            continue
        artifacts.append(RuntimeArtifact(name, str(path), required, True, size_bytes))

    required_ready = all(item.available for item in artifacts if item.required)
    return {
        "ready": required_ready,
        "message": "Required inference artifacts are available." if required_ready else "One or more required inference artifacts are unavailable.",
        "artifacts": artifacts,
    }