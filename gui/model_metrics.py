"""Load measured model metrics for honest GUI labels."""

import json
from pathlib import Path
from typing import Any


def load_model_metrics(model_path: str | Path) -> dict[str, Any] | None:
    path = Path(model_path)
    metrics_path = path.with_name(f"{path.stem}_metrics.json")
    try:
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None