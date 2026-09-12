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
