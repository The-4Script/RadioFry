"""Prerequisite safety guards for analog support (BANK.md Entry 019).

Analog generation is NOT implemented. These cover the two latent defects Entry 018
measured, plus a guard locking the symbol-rate experiment to digital modulations.
"""

import numpy as np
import pytest

from radiofry.evaluation import symbol_rate_experiment
from radiofry.evaluation.metrics import expected_family_for
from radiofry.fusion.confidence_fusion import _FAMILY_BY_LABEL
from radiofry.synthetic_gen.v1 import MODULATIONS, SampleSpec
from radiofry.synthetic_gen.v1.generator import _eb_n0_db

DIGITAL_MODULATIONS = ["BPSK", "QPSK", "8PSK", "BFSK", "16QAM", "64QAM"]


# --- Task 1: family lookup must not crash the harness -------------------------------


@pytest.mark.parametrize(
    "family,expected",
    [("psk", "PSK-like"), ("qam", "QAM-like"), ("fsk", "FSK-like")],
)
def test_existing_digital_family_mappings_are_unchanged(family: str, expected: str) -> None:
    assert expected_family_for(family) == expected


def test_analog_family_resolves_instead_of_raising() -> None:
    assert expected_family_for("analog") == "analog-like"


def test_pam_family_resolves_instead_of_raising() -> None:
    # PAM4 already exists in the registry (Entry 016), so this crash was live, not
    # hypothetical. Fusion maps PAM4 to QAM-like, so the vocabularies agree.
    assert expected_family_for("pam") == "QAM-like"


def test_every_family_in_the_registry_resolves() -> None:
    for spec in MODULATIONS.values():
        assert expected_family_for(spec.family)


def test_family_vocabulary_matches_the_fusion_vocabulary() -> None:
    resolved = {expected_family_for(spec.family) for spec in MODULATIONS.values()}
    resolved.add(expected_family_for("analog"))

    assert resolved <= set(_FAMILY_BY_LABEL.values())


def test_unknown_family_still_raises() -> None:
    with pytest.raises(KeyError):
        expected_family_for("not_a_family")


# --- Task 2: no Eb/N0 for a bit-less capture ----------------------------------------


def test_eb_n0_is_computed_normally_for_digital_captures() -> None:
    assert _eb_n0_db(19.03, 2) == pytest.approx(19.03 - 10 * np.log10(2))
    assert _eb_n0_db(19.03, 1) == pytest.approx(19.03)


def test_eb_n0_is_none_when_there_are_no_bits_per_symbol() -> None:
    # Previously produced -inf, which would have poisoned analog ground truth.
    assert _eb_n0_db(19.03, 0) is None


def test_eb_n0_is_none_when_es_n0_is_none() -> None:
    assert _eb_n0_db(None, 4) is None


def test_eb_n0_is_never_infinite() -> None:
    for bits in (0, 1, 2, 4, 6):
        value = _eb_n0_db(19.03, bits)
        assert value is None or np.isfinite(value)


@pytest.mark.parametrize("modulation", DIGITAL_MODULATIONS + ["PAM4", "GFSK"])
def test_every_current_modulation_still_gets_a_finite_eb_n0(modulation: str) -> None:
    spec = SampleSpec(modulation=modulation, num_symbols=8, samples_per_symbol=8,
                      sample_rate_hz=200_000.0, snr_db=10.0, seed=1)
    es_n0 = 10.0 + 10 * np.log10(spec.samples_per_symbol)

    value = _eb_n0_db(es_n0, spec.bits_per_symbol)

    assert value is not None and np.isfinite(value)


# --- Task 3: the symbol-rate experiment stays digital --------------------------------


def test_symbol_rate_experiment_uses_its_own_digital_modulation_list() -> None:
    assert symbol_rate_experiment.MODULATIONS == DIGITAL_MODULATIONS


def test_symbol_rate_experiment_excludes_the_newer_classes() -> None:
    # It must not silently widen when the registry grows - PAM4 and GFSK are already in
    # the registry and must stay out of this experiment.
    assert "PAM4" not in symbol_rate_experiment.MODULATIONS
    assert "GFSK" not in symbol_rate_experiment.MODULATIONS


def test_symbol_rate_experiment_is_independent_of_the_registry() -> None:
    # Guards the Entry 018 concern: the default must not be tuple(MODULATIONS).
    assert set(symbol_rate_experiment.MODULATIONS) != set(MODULATIONS)
    assert set(symbol_rate_experiment.MODULATIONS) < set(MODULATIONS)


def test_symbol_rate_experiment_default_matches_its_pinned_list() -> None:
    import inspect

    default = inspect.signature(symbol_rate_experiment.run_experiment).parameters["modulations"].default

    assert list(default) == DIGITAL_MODULATIONS


def test_symbol_rate_experiment_would_reject_an_analog_modulation() -> None:
    # build_signal delegates to the generator, which has no analog entry yet.
    with pytest.raises((ValueError, KeyError)):
        symbol_rate_experiment.build_signal("AM-DSB", pulse="rect", snr_db=None, seed=1)
