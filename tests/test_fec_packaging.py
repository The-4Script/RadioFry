"""FEC decoder availability must be visible, not discovered at decode time.

`reedsolo` and `scikit-commpy` are the declared `fec` extra (pyproject) and are included
in `requirements.txt` as `-e .[ml,gui,fec]`, so the *declarations* were already correct.
What was missing was visibility: without the extra installed, `check_runtime_artifacts`
still reported `ready: true` - it checks model FILES - while three of the four FEC schemes
silently returned `success=False` with the reason buried in a per-call message.

These tests pin that the condition is now reported at report level, that the message names
the remedy, and that nothing about decoding behaviour changed (BANK.md Entry 044).
"""

import importlib.util

import numpy as np
import pytest

from radiofry.runtime import FEC_REQUIREMENTS, check_fec_support


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def test_every_dispatchable_scheme_has_a_declared_requirement() -> None:
    """The map must not drift from what `decode_fec` actually accepts."""
    from radiofry.decoding.fec import dispatch

    source = dispatch.decode_fec.__doc__ or ""
    assert set(FEC_REQUIREMENTS) >= {
        "convolutional", "reed_solomon", "concatenated", "ldpc", "none"}
    assert source, "decode_fec should document its parameters"


def test_the_report_states_which_schemes_can_decode() -> None:
    support = check_fec_support()

    assert set(support["schemes"]) == set(FEC_REQUIREMENTS)
    for scheme, modules in FEC_REQUIREMENTS.items():
        expected = all(_installed(module) for module in modules)
        assert support["schemes"][scheme] is expected, scheme


def test_availability_matches_the_installed_modules() -> None:
    every_module_present = all(
        _installed(module)
        for modules in FEC_REQUIREMENTS.values() for module in modules)

    assert check_fec_support()["available"] is every_module_present


def test_ldpc_needs_no_package_but_still_needs_its_code() -> None:
    """A present package must not be read as "can decode a captured LDPC stream"."""
    support = check_fec_support()

    assert FEC_REQUIREMENTS["ldpc"] == (), "LDPC decoding is pure numpy"
    assert support["schemes"]["ldpc"] is True
    assert "parity-check matrix" in support["notes"]
    assert "seed" in support["notes"], "the pseudo-random caveat belongs here too"


def test_a_missing_extra_is_reported_with_the_exact_remedy(monkeypatch) -> None:
    """The message has to be actionable, not merely true."""
    real = importlib.util.find_spec

    def absent(name, *args, **kwargs):
        return None if name in {"reedsolo", "commpy"} else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", absent)
    support = check_fec_support()

    assert support["available"] is False
    assert set(support["missing_modules"]) == {"commpy", "reedsolo"}
    assert support["schemes"] == {
        "convolutional": False, "reed_solomon": False, "concatenated": False,
        "ldpc": True, "none": True}
    assert '.[fec]' in support["message"], "name the install that fixes it"
    for scheme in ("convolutional", "reed_solomon", "concatenated"):
        assert scheme in support["message"]


def test_the_capability_check_is_serialisable_and_cheap() -> None:
    """It lands in the JSON report, and runs on every analysis."""
    import json
    import time

    started = time.perf_counter()
    support = check_fec_support()
    elapsed = time.perf_counter() - started

    assert json.dumps(support)
    assert elapsed < 0.5, "must not add meaningful cost to every analysis"


def test_the_pipeline_surfaces_fec_support_in_the_runtime_stage() -> None:
    from radiofry.contracts import UnifiedSignalContainer
    from radiofry.pipeline import analyze_capture

    rng = np.random.default_rng(11)
    iq = ((rng.normal(size=4_096) + 1j * rng.normal(size=4_096))
          / np.sqrt(2)).astype(np.complex64)

    report = analyze_capture(UnifiedSignalContainer(iq, 200_000.0, "iq"))

    support = report["stages"]["runtime"]["fec_support"]
    assert "available" in support and "schemes" in support and "message" in support


def test_artifact_readiness_and_fec_support_stay_separate() -> None:
    """Model files being present says nothing about decoder packages, and vice versa."""
    from radiofry.runtime import check_runtime_artifacts

    artifacts = check_runtime_artifacts({"modulation": "models_saved/does_not_exist.pt"})

    assert artifacts["ready"] is False
    assert "fec_support" not in artifacts, (
        "the two conditions are reported separately and must not be conflated")
