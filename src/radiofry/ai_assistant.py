"""Optional Groq-powered interpretation of an evidence report."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Iterator

from radiofry.reporting.report_builder import _json_safe

GPT_MODEL = "openai/gpt-oss-120b"
REVIEW_MODEL = "qwen/qwen3.8-27b"


def _load_local_key() -> str | None:
    """Read a local env file without overriding an explicitly exported key."""

    if os.getenv("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    root = Path(__file__).resolve().parents[2]
    for filename in (".env.local", ".env"):
        path = root / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "GROQ_API_KEY":
                return value.strip().strip("\"'")
    return None


def groq_available() -> bool:
    """Return whether the optional SDK and an API key are available."""

    try:
        import groq  # noqa: F401
    except ImportError:
        return False
    return bool(_load_local_key())


def _client() -> Any:
    try:
        from groq import Groq
    except ImportError as error:
        raise RuntimeError("Install the optional Groq dependency to use AI analysis.") from error
    api_key = _load_local_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it to the environment or .env.local.")
    return Groq(api_key=api_key)


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Keep descriptive evidence while excluding waveform/bitstream-sized arrays."""

    if depth > 5:
        return "[nested evidence omitted]"
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, depth=depth + 1)
            for key, item in value.items()
            if str(key) not in {"bits", "symbols", "samples", "waveform", "iq", "spectrum"}
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 12:
            return [_compact_value(item, depth=depth + 1) for item in value[:12]] + [
                f"[{len(value) - 12} items omitted]"
            ]
        return [_compact_value(item, depth=depth + 1) for item in value]
    return value


def _ai_report_context(report: dict[str, Any], limit: int = 8000) -> str:
    """Build a bounded prompt context; the complete report remains available locally."""

    compact = _compact_value(_json_safe(report))
    encoded = json.dumps(compact, separators=(",", ":"), sort_keys=True)
    return encoded[:limit] + ("...[context truncated]" if len(encoded) > limit else "")


def _stream(model: str, prompt: str, *, temperature: float, top_p: float, reasoning_effort: str) -> Iterator[str]:
    from groq import AuthenticationError, GroqError, RateLimitError

    try:
        completion = _client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=2048,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            stream=True,
            stop=None,
        )
        for chunk in completion:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    except RateLimitError as error:
        raise RuntimeError(
            "Groq rejected the request because it exceeded the model or account limit. "
            "The evidence prompt was bounded; wait briefly and try again."
        ) from error
    except AuthenticationError as error:
        raise RuntimeError(
            "Groq rejected the request: GROQ_API_KEY is missing, invalid, or revoked."
        ) from error
    except GroqError as error:
        # Every other Groq failure (connection drop, timeout, malformed request, a 5xx
        # from Groq, or a disconnect partway through the stream) is a GroqError subclass,
        # not a RuntimeError/OSError/ValueError. It was previously uncaught here, so it
        # skipped the friendly banner gui/pages/8_report.py renders for RuntimeError and
        # crashed the report page with a raw traceback instead.
        raise RuntimeError(f"Groq request failed: {error}") from error


def stream_executive_brief(report: dict[str, Any]) -> Iterator[str]:
    """Stream a concise, evidence-grounded technical brief."""

    prompt = (
        "You are RadioFry's senior RF analyst. Analyze the JSON evidence below. "
        "Do not invent measurements, protocols, or certainty. Write a concise report with "
        "headings: Assessment, Evidence, Caveats, and Next action. Explain what the current "
        "result means for an engineer and explicitly distinguish model confidence from proof.\n\n"
        f"REPORT JSON (bounded evidence context):\n{_ai_report_context(report)}"
    )
    return _stream(GPT_MODEL, prompt, temperature=1, top_p=1, reasoning_effort="medium")


def stream_critical_review(report: dict[str, Any], brief: str) -> Iterator[str]:
    """Stream an independent second-model challenge of the first interpretation."""

    prompt = (
        "You are a skeptical RF verification reviewer. Review the machine-generated brief "
        "against the original evidence. Identify unsupported claims, missing evidence, and "
        "the smallest useful follow-up measurement. Be specific and concise. Never upgrade "
        "an unavailable or reviewable stage into a confirmed result.\n\n"
        f"ORIGINAL REPORT (bounded evidence context):\n{_ai_report_context(report, 6000)}\n\n"
        f"FIRST ANALYST BRIEF:\n{brief[:5000]}"
    )
    return _stream(REVIEW_MODEL, prompt, temperature=0.6, top_p=0.95, reasoning_effort="default")
