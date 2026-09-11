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


# Which optional dependency each FEC scheme needs. Declared in pyproject as the `fec`
# extra and included in requirements.txt; this maps the scheme a caller asks for onto the
# import that has to succeed for it to actually decode (BANK.md Entry 044).
FEC_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "convolutional": ("commpy",),
    "reed_solomon": ("reedsolo",),
    "concatenated": ("commpy", "reedsolo"),
    "ldpc": (),          # pure numpy; needs a parity-check matrix, not a package
    "none": (),
}

FEC_EXTRA_HINT = 'pip install -e ".[fec]"'


def check_fec_support() -> dict[str, object]:
    """Report which FEC schemes can actually decode in this environment.

    Without the `fec` extra, `decode_fec` returns `success=False` with a per-call message
    and nothing else surfaces it - a deployment could report `ready: true` while three of
    the four schemes silently produced no decoding at all. This makes that condition
    visible at report level. It imports nothing eagerly beyond the check itself and
    changes no decoding behaviour.
    """

    import importlib.util

    present: dict[str, bool] = {}
    for scheme, modules in FEC_REQUIREMENTS.items():
        present[scheme] = all(
            importlib.util.find_spec(module) is not None for module in modules)

    missing_modules = sorted({
        module
        for modules in FEC_REQUIREMENTS.values()
        for module in modules
        if importlib.util.find_spec(module) is None
    })
    unavailable = sorted(name for name, ok in present.items() if not ok)

    if not missing_modules:
        message = "All FEC decoders are available."
    else:
        message = (
            f"FEC decoding unavailable for {', '.join(unavailable)}: missing "
            f"{', '.join(missing_modules)}. Install the declared extra with "
            f"{FEC_EXTRA_HINT}.")

    return {
        "available": not missing_modules,
        "schemes": present,
        "missing_modules": missing_modules,
        "message": message,
        # LDPC needs code parameters rather than a package; stated so a reader does not
        # mistake "package present" for "can decode a captured LDPC stream blind".
        "notes": ("LDPC decoding additionally requires the code's parity-check matrix; "
                  "pseudo-random de-interleaving requires the generator seed. Neither is "
                  "recoverable from a captured bitstream alone."),
    }
