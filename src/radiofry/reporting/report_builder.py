"""JSON-safe aggregation of pipeline results."""

from datetime import datetime, timezone
import json
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {name: _json_safe(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_report(*, source: dict[str, Any], stages: dict[str, Any]) -> dict[str, Any]:
    """Build a stable report envelope while preserving stage-level results."""

    return {
        "schema_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _json_safe(source),
        "stages": _json_safe(stages),
    }


def report_json(report: dict[str, Any]) -> str:
    """Serialize a report for Streamlit download or filesystem export."""

    return json.dumps(_json_safe(report), indent=2, sort_keys=True)
