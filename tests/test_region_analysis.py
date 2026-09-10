"""Region select and re-analyze through the existing pipeline (Time Machine V2).

The dangerous failure here is not a crash - it is analysing the wrong samples and
presenting the result as if it described the region the analyst chose. These tests
concentrate on that: exact sample bounds, exact container contents, honest metadata, and
the real `analyze_capture` being the thing that runs.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.explore.time_machine import (
    MIN_ANALYSIS_SAMPLES,
    WindowSelection,
    compare_reports,
    region_container,
    select_region,
)

FS = 200_000.0
N = 40_000
V1_CAPTURES = Path(__file__).resolve().parents[1] / "data" / "synthetic_v1" / "captures"


def _ramp(n: int = N) -> np.ndarray:
    """Samples that encode their own index, so a slice proves which region it is."""
    return (np.arange(n, dtype=np.float64)
            + 1j * np.arange(n, dtype=np.float64)).astype(np.complex64)


# --- timestamp to sample mapping ---------------------------------------------------------


def test_a_region_maps_to_exact_sample_bounds() -> None:
    result = select_region(N, FS, 0.05, 0.10)

    assert result.ok
    assert result.selection.start_sample == 10_000
    assert result.selection.end_sample == 20_000
    assert result.selection.length_samples == 10_000
    assert result.selection.duration_seconds == pytest.approx(0.05)


def test_the_region_reuses_the_window_selection_type() -> None:
    # One selection concept in the feature, not two.
    assert isinstance(select_region(N, FS, 0.0, 0.05).selection, WindowSelection)


def test_a_region_running_past_the_end_is_clipped_to_the_recording() -> None:
    result = select_region(N, FS, 0.15, 5.0)

    assert result.ok
    assert result.selection.end_sample == N


# --- validation refuses rather than guesses -------------------------------------------------


def test_a_reversed_region_is_refused_not_silently_swapped() -> None:
    result = select_region(N, FS, 0.10, 0.05)

    assert not result.ok
    assert result.selection is None
    assert "before it starts" in result.reason


def test_an_empty_region_is_refused() -> None:
    result = select_region(N, FS, 0.05, 0.05)

    assert not result.ok


def test_a_region_starting_past_the_recording_is_refused() -> None:
    result = select_region(N, FS, 5.0, 6.0)

    assert not result.ok
    assert "past the end" in result.reason


def test_a_region_ending_before_the_recording_is_refused() -> None:
    assert not select_region(N, FS, -2.0, -1.0).ok


def test_a_region_too_short_to_analyse_is_refused_with_the_reason() -> None:
    result = select_region(N, FS, 0.0, MIN_ANALYSIS_SAMPLES / FS / 2)

    assert not result.ok
    assert str(MIN_ANALYSIS_SAMPLES) in result.reason


def test_a_region_exactly_at_the_minimum_is_accepted() -> None:
    result = select_region(N, FS, 0.0, MIN_ANALYSIS_SAMPLES / FS)

    assert result.ok
    assert result.selection.length_samples == MIN_ANALYSIS_SAMPLES


def test_an_empty_capture_is_refused() -> None:
    assert not select_region(0, FS, 0.0, 0.1).ok


def test_a_capture_without_a_sample_rate_is_refused_with_an_explanation() -> None:
    result = select_region(N, None, 0.0, 0.1)

    assert not result.ok
    assert "sample rate" in result.reason


def test_a_very_short_recording_does_not_crash() -> None:
    result = select_region(10, FS, 0.0, 1.0)

    assert not result.ok
    assert result.selection is None


# --- the container holds exactly the selected samples -------------------------------------------


def test_the_container_holds_exactly_the_selected_source_samples() -> None:
    signal = UnifiedSignalContainer(_ramp(), FS, "iq")
    selection = select_region(N, FS, 0.05, 0.10).selection

    region = region_container(signal, selection)

    assert region.iq.size == 10_000
    assert np.array_equal(region.iq, signal.iq[10_000:20_000])
    assert region.iq[0].real == pytest.approx(10_000.0), "wrong region extracted"


def test_the_container_preserves_sample_rate_and_source_format() -> None:
    signal = UnifiedSignalContainer(_ramp(), FS, "wav")

    region = region_container(signal, select_region(N, FS, 0.0, 0.05).selection)

    assert region.sample_rate == FS
    assert region.source_format == "wav"


def test_the_container_records_which_part_of_the_capture_it_is() -> None:
    signal = UnifiedSignalContainer(_ramp(), FS, "iq", {"path": "capture.iq"})

    region = region_container(signal, select_region(N, FS, 0.05, 0.10).selection)

    assert region.metadata["region_of_capture"] is True
    assert region.metadata["region_start_sample"] == 10_000
    assert region.metadata["region_end_sample"] == 20_000
    assert region.metadata["region_start_seconds"] == pytest.approx(0.05)
    # The original path still names the real source and must not be dropped, but the
    # region bounds above stop it from implying the whole file.
    assert region.metadata["path"] == "capture.iq"


def test_legitimate_metadata_carries_over() -> None:
    signal = UnifiedSignalContainer(
        _ramp(), FS, "iq", {"center_frequency_hz": 433_000_000.0, "dtype": "int16"})

    region = region_container(signal, select_region(N, FS, 0.0, 0.05).selection)

    assert region.metadata["center_frequency_hz"] == 433_000_000.0
    assert region.metadata["dtype"] == "int16"


def test_absent_metadata_is_not_invented() -> None:
    signal = UnifiedSignalContainer(_ramp(), FS, "iq")

    region = region_container(signal, select_region(N, FS, 0.0, 0.05).selection)

    assert "center_frequency_hz" not in region.metadata


def test_the_container_never_over_reads_a_shorter_array() -> None:
    signal = UnifiedSignalContainer(_ramp(500), FS, "iq")
    selection = WindowSelection(0, 40_000, 40_000, FS)  # built for a longer capture

    assert region_container(signal, selection).iq.size == 500


# --- the existing pipeline is what runs ------------------------------------------------------------


@pytest.mark.skipif(not V1_CAPTURES.exists(), reason="frozen V1 dataset not present")
def test_analyze_capture_runs_on_the_selected_region_and_sees_those_samples() -> None:
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import analyze_capture, load_capture

    truth = json.loads((V1_CAPTURES / "16QAM_snr20dB_r000.json").read_text(encoding="utf-8"))
    entry = next(f for f in truth["files"] if f["file_format"] == "iq")
    signal = load_capture(V1_CAPTURES / "16QAM_snr20dB_r000.iq",
                          sample_rate=truth["signal"]["sample_rate_hz"],
                          iq_format=IQFormat(entry["dtype"], entry["byte_order"]))
    selection = select_region(signal.iq.size, signal.sample_rate, 0.02, 0.10).selection
    region = region_container(signal, selection)

    report = analyze_capture(region)

    # The report describes the region, not the whole capture.
    assert report["source"]["samples"] == selection.length_samples
    assert report["source"]["samples"] < signal.iq.size
    assert report["stages"]["fusion"]["label"]


@pytest.mark.skipif(not V1_CAPTURES.exists(), reason="frozen V1 dataset not present")
def test_a_region_analysis_is_not_the_whole_capture_analysis() -> None:
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import analyze_capture, load_capture

    truth = json.loads((V1_CAPTURES / "16QAM_snr20dB_r000.json").read_text(encoding="utf-8"))
    entry = next(f for f in truth["files"] if f["file_format"] == "iq")
    signal = load_capture(V1_CAPTURES / "16QAM_snr20dB_r000.iq",
                          sample_rate=truth["signal"]["sample_rate_hz"],
                          iq_format=IQFormat(entry["dtype"], entry["byte_order"]))
    whole = analyze_capture(signal)
    selection = select_region(signal.iq.size, signal.sample_rate, 0.05, 0.09).selection

    region = analyze_capture(region_container(signal, selection))

    assert region["source"]["samples"] != whole["source"]["samples"], (
        "the selection analysis must not be a re-run of the whole capture")


# --- comparison table -----------------------------------------------------------------------------


def _report(**parameters) -> dict:
    return {"stages": {"parameters": parameters,
                       "fusion": {"label": "QPSK", "trust_score": 0.9},
                       "classical_modulation": {"family": "PSK-like"},
                       "cnn_modulation": {"label": "QPSK", "confidence": 0.88}}}


def test_the_comparison_pairs_fields_present_in_both_reports() -> None:
    rows = compare_reports(_report(snr_db=14.2), _report(snr_db=9.8))

    labels = {row[0] for row in rows}
    assert "Modulation" in labels and "Estimated SNR" in labels
    snr = next(row for row in rows if row[0] == "Estimated SNR")
    assert snr[1] == "14.2 dB" and snr[2] == "9.8 dB"


def test_a_field_missing_from_either_side_is_dropped_not_padded() -> None:
    rows = compare_reports(_report(snr_db=14.2), _report())

    assert "Estimated SNR" not in {row[0] for row in rows}


def test_the_comparison_is_empty_without_both_reports() -> None:
    assert compare_reports(None, _report()) == []
    assert compare_reports(_report(), None) == []


def test_the_comparison_never_invents_a_measurement() -> None:
    rows = compare_reports(_report(), _report())

    for _, left, right in rows:
        assert left and right
        assert "None" not in (left, right)


# --- visual (drag) selection maps to the same region as the numeric form ---------------------


def test_a_drag_maps_to_the_same_region_as_typed_bounds() -> None:
    from radiofry.explore.time_machine import region_from_drag

    dragged = region_from_drag(N, FS, 0.05, 0.10)
    typed = select_region(N, FS, 0.05, 0.10)

    assert dragged.ok and typed.ok
    assert dragged.selection == typed.selection


def test_a_right_to_left_drag_is_ordered_not_rejected() -> None:
    # Dragging backwards is a normal gesture; typing an end before a start is an error.
    from radiofry.explore.time_machine import region_from_drag

    backwards = region_from_drag(N, FS, 0.10, 0.05)

    assert backwards.ok
    assert backwards.selection.start_sample == 10_000
    assert backwards.selection.end_sample == 20_000


def test_a_drag_past_the_end_is_clamped_to_the_recording() -> None:
    from radiofry.explore.time_machine import region_from_drag

    result = region_from_drag(N, FS, 0.15, 99.0)

    assert result.ok
    assert result.selection.end_sample == N


def test_a_drag_starting_before_zero_is_clamped() -> None:
    from radiofry.explore.time_machine import region_from_drag

    result = region_from_drag(N, FS, -3.0, 0.05)

    assert result.ok
    assert result.selection.start_sample == 0


def test_a_zero_width_drag_is_refused() -> None:
    from radiofry.explore.time_machine import region_from_drag

    assert not region_from_drag(N, FS, 0.05, 0.05).ok


def test_a_tiny_drag_is_refused_by_the_same_minimum_as_the_form() -> None:
    from radiofry.explore.time_machine import region_from_drag

    result = region_from_drag(N, FS, 0.05, 0.05 + MIN_ANALYSIS_SAMPLES / FS / 4)

    assert not result.ok
    assert str(MIN_ANALYSIS_SAMPLES) in result.reason


def test_a_drag_with_no_range_is_refused() -> None:
    from radiofry.explore.time_machine import region_from_drag

    assert not region_from_drag(N, FS, None, 0.1).ok
    assert not region_from_drag(N, FS, 0.1, None).ok


def test_a_drag_without_a_sample_rate_is_refused() -> None:
    from radiofry.explore.time_machine import region_from_drag

    assert not region_from_drag(N, None, 0.0, 0.1).ok


def test_the_overview_spans_the_whole_capture_at_bounded_cost() -> None:
    from radiofry.explore.time_machine import capture_overview

    times, frequencies, power = capture_overview(_ramp(400_000), FS, rows=64)

    assert power.shape[0] <= 64, "row budget caps the work regardless of capture length"
    assert times.size == power.shape[0]
    assert frequencies.size == power.shape[1]
    assert times[-1] > 0.9 * (400_000 / FS), "the overview must reach the recording end"


def test_the_overview_is_empty_for_an_unusable_capture() -> None:
    from radiofry.explore.time_machine import capture_overview

    assert capture_overview(np.empty(0, dtype=np.complex64), FS)[0].size == 0
    assert capture_overview(_ramp(1_000), None)[0].size == 0


# --- locked-region helpers (Time Machine V4) --------------------------------------------


def test_timestamps_are_shown_as_clock_positions() -> None:
    from radiofry.explore.time_machine import format_timestamp

    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(10.25) == "00:10.250"
    assert format_timestamp(14.8) == "00:14.800"
    assert format_timestamp(None) == "n/a"


def test_a_timestamp_past_an_hour_gains_an_hours_field() -> None:
    from radiofry.explore.time_machine import format_timestamp

    assert format_timestamp(3661.5) == "1:01:01.500"


def test_a_timestamp_is_truncated_never_rounded_up() -> None:
    """A displayed bound must not point past the sample it describes."""
    from radiofry.explore.time_machine import format_timestamp

    assert format_timestamp(42.8009) == "00:42.800"
    assert format_timestamp(0.9999) == "00:00.999"


def test_a_negative_timestamp_is_floored_at_zero() -> None:
    from radiofry.explore.time_machine import format_timestamp

    assert format_timestamp(-5.0) == "00:00.000"


def test_an_analysis_is_claimed_only_for_the_exact_samples_it_ran_on() -> None:
    from radiofry.explore.time_machine import analysis_for_region

    ran = {"start_sample": 2_000, "end_sample": 12_000, "report": {"tag": "exact"}}

    assert analysis_for_region([ran], 2_000, 12_000) is ran
    assert analysis_for_region([ran], 2_000, 12_001) is None, "one sample off is a "\
        "different region"
    assert analysis_for_region([ran], 1_999, 12_000) is None
    assert analysis_for_region([ran], 4_000, 8_000) is None, "an overlapping region is "\
        "not this region"


def test_the_most_recent_analysis_of_a_region_is_the_one_returned() -> None:
    from radiofry.explore.time_machine import analysis_for_region

    first = {"start_sample": 0, "end_sample": 100, "report": {"run": 1}}
    second = {"start_sample": 0, "end_sample": 100, "report": {"run": 2}}

    assert analysis_for_region([first, second], 0, 100) is second


def test_no_investigations_means_no_analysis() -> None:
    from radiofry.explore.time_machine import analysis_for_region

    assert analysis_for_region([], 0, 100) is None
    assert analysis_for_region(None, 0, 100) is None


def test_the_end_timestamp_comes_from_the_end_sample_not_start_plus_duration() -> None:
    """start + duration lands a half-ULP low, which costs a whole displayed millisecond.

    6000/fs + 11000/fs == 0.08499999999999999, so truncating it to milliseconds shows a
    region ending at 0.085 s as 00:00.084. Reading the boundary off the sample it
    actually is removes the error at source.
    """
    from radiofry.explore.time_machine import format_timestamp

    held = WindowSelection(6_000, 11_000, N, FS)

    assert held.start_seconds + held.duration_seconds < held.end_seconds, (
        "the artifact this property exists to avoid")
    assert held.end_seconds == 0.085
    assert format_timestamp(held.end_seconds) == "00:00.085"


def test_end_seconds_agrees_with_end_sample_across_the_capture() -> None:
    for start, length in ((0, 128), (6_000, 11_000), (1, 39_999), (24_000, 16_000)):
        held = WindowSelection(start, length, N, FS)
        assert held.end_seconds == held.end_sample / FS


def test_end_seconds_is_unknown_without_a_sample_rate() -> None:
    assert WindowSelection(0, 100, N, None).end_seconds is None


def test_formatting_absorbs_representation_error_without_rounding_up() -> None:
    """The slack is a nanosecond: enough for binary noise, never a real millisecond."""
    from radiofry.explore.time_machine import format_timestamp

    assert format_timestamp(0.03 + 0.055) == "00:00.085", "binary noise absorbed"
    assert format_timestamp(42.8009) == "00:42.800", "a real 0.9 ms is still truncated"
    assert format_timestamp(0.0849) == "00:00.084"


# --- investigation names (label only, never analysis input) ------------------------------


def test_an_empty_name_falls_back_to_a_numbered_default() -> None:
    from radiofry.explore.time_machine import investigation_name

    assert investigation_name("", 1) == "Investigation 1"
    assert investigation_name(None, 4) == "Investigation 4"


def test_a_whitespace_only_name_counts_as_empty() -> None:
    """A stray space must not become the label."""
    from radiofry.explore.time_machine import investigation_name

    assert investigation_name("   ", 2) == "Investigation 2"
    assert investigation_name("\t\n", 2) == "Investigation 2"


def test_a_supplied_name_is_kept_and_trimmed() -> None:
    from radiofry.explore.time_machine import investigation_name

    assert investigation_name("Burst 3", 1) == "Burst 3"
    assert investigation_name("  Possible frequency shift  ", 1) == \
        "Possible frequency shift"


def test_naming_does_not_touch_region_matching() -> None:
    """`analysis_for_region` matches on samples, so a name can never redirect it."""
    from radiofry.explore.time_machine import analysis_for_region

    named = {"name": "Burst 3", "start_sample": 2_000, "end_sample": 12_000}
    renamed = dict(named, name="Something else entirely")

    assert analysis_for_region([named], 2_000, 12_000) is named
    assert analysis_for_region([renamed], 2_000, 12_000) is renamed
    assert analysis_for_region([renamed], 2_000, 12_001) is None


def test_an_entry_stored_before_names_existed_still_resolves() -> None:
    from radiofry.explore.time_machine import investigation_name

    legacy = {"start_sample": 0, "end_sample": 100}

    assert investigation_name(legacy.get("name"), 1) == "Investigation 1"
