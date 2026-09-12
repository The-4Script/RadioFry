"""Optional Groq-powered interpretation of an evidence report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from radiofry.reporting.report_builder import report_json

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


def _stream(model: str, prompt: str, *, temperature: float, top_p: float, reasoning_effort: str) -> Iterator[str]:
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


def stream_executive_brief(report: dict[str, Any]) -> Iterator[str]:
    """Stream a concise, evidence-grounded technical brief."""

    prompt = (
        "You are RadioFry's senior RF analyst. Analyze the JSON evidence below. "
        "Do not invent measurements, protocols, or certainty. Write a concise report with "
        "headings: Assessment, Evidence, Caveats, and Next action. Explain what the current "
        "result means for an engineer and explicitly distinguish model confidence from proof.\n\n"
        f"REPORT JSON:\n{report_json(report)[:24000]}"
    )
    return _stream(GPT_MODEL, prompt, temperature=1, top_p=1, reasoning_effort="medium")


def stream_critical_review(report: dict[str, Any], brief: str) -> Iterator[str]:
    """Stream an independent second-model challenge of the first interpretation."""

    prompt = (
        "You are a skeptical RF verification reviewer. Review the machine-generated brief "
        "against the original evidence. Identify unsupported claims, missing evidence, and "
        "the smallest useful follow-up measurement. Be specific and concise. Never upgrade "
        "an unavailable or reviewable stage into a confirmed result.\n\n"
        f"ORIGINAL REPORT:\n{report_json(report)[:18000]}\n\n"
        f"FIRST ANALYST BRIEF:\n{brief[:10000]}"
    )
    return _stream(REVIEW_MODEL, prompt, temperature=0.6, top_p=0.95, reasoning_effort="default")
