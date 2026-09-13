"""Every Groq SDK failure must surface as a RuntimeError.

groq.GroqError and all of its subclasses (AuthenticationError, APIConnectionError,
APITimeoutError, InternalServerError, RateLimitError, ...) inherit only from the
stdlib Exception - never from RuntimeError, OSError, or ValueError. `_stream` used to
catch RateLimitError alone, and gui/pages/8_report.py catches
`(RuntimeError, OSError, ValueError)` around the streaming calls, so an expired API
key, a dropped connection, a Groq-side 5xx, or a disconnect partway through the
stream was not a RuntimeError/OSError/ValueError either and reached neither except
clause: it propagated uncaught and crashed the report page instead of showing the
"AI analysis could not be completed" banner every other failure mode gets.

These tests fake the Groq client so they need no network access and no API key.
"""

from unittest.mock import MagicMock

import httpx
import pytest
from groq import APIConnectionError, AuthenticationError, InternalServerError, RateLimitError

from radiofry import ai_assistant


def _chunk(text: str):
    piece = MagicMock()
    piece.choices = [MagicMock(delta=MagicMock(content=text))]
    return piece


def _status_error(cls, message: str):
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.groq.com/x"))
    return cls(message, response=response, body=None)


@pytest.mark.parametrize(
    "error",
    [
        _status_error(AuthenticationError, "invalid api key"),
        _status_error(RateLimitError, "too many requests"),
        _status_error(InternalServerError, "server error"),
        APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x")),
    ],
    ids=["authentication", "rate_limit", "server_5xx", "connection"],
)
def test_every_groq_failure_becomes_a_runtime_error(monkeypatch, error) -> None:
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = error
    monkeypatch.setattr(ai_assistant, "_client", lambda: fake_client)

    with pytest.raises(RuntimeError):
        list(ai_assistant.stream_executive_brief({"stages": {}}))


def test_a_failure_partway_through_the_stream_is_also_caught(monkeypatch) -> None:
    """The original code only wrapped the initial `.create()` call; a break in the
    connection while iterating the stream itself was not covered at all."""

    def interrupted_stream():
        yield _chunk("partial evidence, then the connection drops")
        raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = interrupted_stream()
    monkeypatch.setattr(ai_assistant, "_client", lambda: fake_client)

    received = []
    with pytest.raises(RuntimeError):
        for part in ai_assistant.stream_executive_brief({"stages": {}}):
            received.append(part)

    # The chunks that arrived before the drop are not lost.
    assert received == ["partial evidence, then the connection drops"]


def test_a_successful_stream_is_unaffected(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(
        [_chunk("Assessment: "), _chunk("evidence supports QPSK.")]
    )
    monkeypatch.setattr(ai_assistant, "_client", lambda: fake_client)

    assert "".join(ai_assistant.stream_executive_brief({"stages": {}})) == (
        "Assessment: evidence supports QPSK."
    )


def test_the_authentication_message_names_the_actual_problem(monkeypatch) -> None:
    """A bad key and a rate limit need different next actions from whoever is
    running the demo, so the two must not collapse into one generic message."""

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _status_error(
        AuthenticationError, "invalid api key"
    )
    monkeypatch.setattr(ai_assistant, "_client", lambda: fake_client)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        list(ai_assistant.stream_executive_brief({"stages": {}}))
