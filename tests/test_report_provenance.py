import json
import math

import numpy as np

from radiofry.correlation.bitstream_correlation import correlate_bitstream
from radiofry.reporting.report_builder import build_report, report_json
from radiofry.runtime import check_runtime_environment


def test_report_marks_protocol_and_field_accuracy_as_unverified() -> None:
    marker = np.array([0, 1, 0, 1], dtype=np.uint8)
    report = build_report(
        source={"format": "iq", "sample_rate": None, "samples": 4},
        stages={
            "fusion": {"label": "BPSK", "review_recommended": True},
            "bitstream_analysis": {"verification": {"protocol_verified": False}},
        },
    )

    assert report["schema_version"] == "0.3"
    assert report["interpretation"]["human_review_required"] is True
    assert report["interpretation"]["protocol_verified"] is False
    assert report["interpretation"]["confidence_is_not_field_accuracy"] is True
    report_json(report)
    assert correlate_bitstream(marker).protocol_verified is False


def test_runtime_health_has_explicit_python_and_dependency_sections() -> None:
    result = check_runtime_environment()

    assert {"python", "dependencies", "ready"} <= result.keys()
    assert isinstance(result["dependencies"], dict)


def test_non_finite_values_serialise_as_null_not_as_bare_nan_tokens() -> None:
    """A bare NaN/Infinity is accepted by Python's json module (non-standard) but is
    not valid JSON per RFC 8259, so a strict downstream parser - or anything other
    than Python re-reading its own export - would reject the whole report over one
    number a stage happened not to be able to compute. It must become `null`, and
    only the affected leaf: everything else in the same report is unaffected."""

    report = build_report(
        source={"format": "iq", "sample_rate": None, "samples": 4},
        stages={
            "parameters": {
                "snr_db": float("nan"),
                "carrier_frequency_hz": float("inf"),
                "occupied_bandwidth_hz": float("-inf"),
                "symbol_rate_hz": 1234.5,
            },
            "fusion": {"label": "BPSK", "review_recommended": True},
        },
    )

    encoded = report_json(report)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded

    reloaded = json.loads(encoded)  # fails outright if any non-standard token slipped through
    parameters = reloaded["stages"]["parameters"]
    assert parameters["snr_db"] is None
    assert parameters["carrier_frequency_hz"] is None
    assert parameters["occupied_bandwidth_hz"] is None
    assert parameters["symbol_rate_hz"] == 1234.5


def test_a_complex_value_with_a_non_finite_component_is_also_sanitised() -> None:
    report = build_report(
        source={"format": "iq", "sample_rate": None, "samples": 4},
        stages={"parameters": {"estimate": complex(float("nan"), 1.0)}},
    )

    encoded = report_json(report)
    reloaded = json.loads(encoded)

    assert reloaded["stages"]["parameters"]["estimate"] == {"real": None, "imag": 1.0}


def test_a_finite_numpy_scalar_still_serialises_to_its_plain_value() -> None:
    report = build_report(
        source={"format": "iq", "sample_rate": None, "samples": 4},
        stages={"parameters": {"snr_db": np.float32(12.5)}},
    )

    reloaded = json.loads(report_json(report))

    assert math.isclose(reloaded["stages"]["parameters"]["snr_db"], 12.5, rel_tol=1e-4)
