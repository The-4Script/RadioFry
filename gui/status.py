"""Shared status decisions for evidence stages."""


CORRELATION_MATCH_THRESHOLD = 0.75


def correlation_status(correlation: object) -> tuple[str, str]:
    if not correlation:
        return "Waiting", "muted"
    found = correlation.get("sync_pattern") is not None or correlation.get("sync_match_score", 0.0) >= CORRELATION_MATCH_THRESHOLD
    return ("Available", "ready") if found else ("Needs review", "review")