"""Signal Time Machine window controller and view data.

The controller is the single source of truth for which part of the recording is being
looked at, so these tests concentrate on the two things that would make the feature lie:
a window that does not correspond to the requested time, and view data that is not the
real signal.
"""

import numpy as np
import pytest

from radiofry.contracts import UnifiedSignalContainer
from radiofry.explore.time_machine import (
    DEFAULT_WINDOW_SECONDS,
    advance_cursor,
    downsample_for_display,
    extract_window,
    select_window,
    symbol_rate_decimated,
    total_duration_seconds,
    window_measurements,
    window_spectrogram,
    window_spectrum,
)

FS = 200_000.0
N = 40_000


def _ramp(n: int = N) -> np.ndarray:
    """Samples whose value encodes their index, so a slice proves its own position."""
    return (np.arange(n, dtype=np.float64)
            + 1j * np.arange(n, dtype=np.float64)).astype(np.complex64)


# --- timeline position maps to sample indices ------------------------------------------


def test_duration_follows_sample_count_and_rate() -> None:
    assert total_duration_seconds(N, FS) == pytest.approx(0.2)


def test_duration_is_unknown_without_a_sample_rate() -> None:
    assert total_duration_seconds(N, None) is None
    assert total_duration_seconds(N, 0.0) is None


def test_cursor_maps_to_the_expected_start_sample() -> None:
    selection = select_window(N, FS, cursor_seconds=0.05, window_seconds=0.01)

    assert selection.start_sample == 10_000
    assert selection.length_samples == 2_000
    assert selection.start_seconds == pytest.approx(0.05)
    assert selection.duration_seconds == pytest.approx(0.01)


def test_moving_the_cursor_moves_the_extracted_samples() -> None:
    iq = _ramp()

    first = extract_window(iq, select_window(N, FS, 0.00, 0.01))
    second = extract_window(iq, select_window(N, FS, 0.05, 0.01))

    assert first[0].real == pytest.approx(0.0)
    assert second[0].real == pytest.approx(10_000.0)
    assert not np.array_equal(first, second)


def test_every_view_derives_from_the_same_selection() -> None:
    # The point of the feature: one window feeds all views.
    iq = _ramp()
    selection = select_window(N, FS, 0.05, 0.01)
    window = extract_window(iq, selection)

    _, displayed = downsample_for_display(window)
    frequencies, magnitude = window_spectrum(window, FS)
    times, _, power = window_spectrogram(window, FS)

    assert displayed.size > 0 and frequencies.size > 0 and times.size > 0
    assert np.isin(displayed, window).all(), "plotted points must be real samples"
    assert power.shape[0] == times.size


# --- window boundaries are safe -----------------------------------------------------------


def test_a_cursor_at_the_end_yields_an_in_bounds_window() -> None:
    selection = select_window(N, FS, cursor_seconds=0.199, window_seconds=0.01)

    assert selection.end_sample <= N
    assert selection.start_sample >= 0
    assert selection.length_samples > 0


def test_a_cursor_past_the_end_is_clamped() -> None:
    selection = select_window(N, FS, cursor_seconds=999.0, window_seconds=0.01)

    assert selection.end_sample == N


def test_a_negative_cursor_is_clamped_to_the_start() -> None:
    assert select_window(N, FS, cursor_seconds=-5.0, window_seconds=0.01).start_sample == 0


def test_a_window_longer_than_the_capture_becomes_the_whole_capture() -> None:
    selection = select_window(N, FS, cursor_seconds=0.0, window_seconds=10.0)

    assert selection.start_sample == 0
    assert selection.length_samples == N


def test_extraction_never_exceeds_the_available_samples() -> None:
    iq = _ramp(100)
    # A selection built for a longer capture must not over-read a shorter array.
    selection = select_window(N, FS, 0.05, 0.01)

    assert extract_window(iq, selection).size <= 100


# --- empty and short signals must not crash -------------------------------------------------


def test_an_empty_capture_produces_an_empty_window() -> None:
    selection = select_window(0, FS, 0.0, 0.01)

    assert selection.length_samples == 0
    assert extract_window(np.empty(0, dtype=np.complex64), selection).size == 0


@pytest.mark.parametrize("size", [0, 1, 2, 4, 7])
def test_short_captures_do_not_crash_any_view(size: int) -> None:
    iq = _ramp(size)
    selection = select_window(size, FS, 0.0, 0.01)
    window = extract_window(iq, selection)

    downsample_for_display(window)
    window_spectrum(window, FS)
    window_spectrogram(window, FS)
    window_measurements(window, FS)


def test_a_missing_sample_rate_falls_back_to_the_whole_capture() -> None:
    selection = select_window(N, None, 0.05, 0.01)

    assert selection.start_sample == 0
    assert selection.length_samples == N
    assert selection.duration_seconds is None


def test_views_return_empty_rather_than_inventing_data_without_a_rate() -> None:
    window = _ramp(1_000)

    assert window_spectrum(window, None)[0].size == 0
    assert window_spectrogram(window, None)[0].size == 0


# --- visualisation receives real samples ------------------------------------------------------


def test_downsampling_decimates_and_does_not_interpolate() -> None:
    values = _ramp(10_000)

    indices, displayed = downsample_for_display(values, max_points=500)

    assert displayed.size <= 500
    assert np.array_equal(displayed, values[indices]), "values must be untouched samples"


def test_a_small_window_is_returned_untouched() -> None:
    values = _ramp(50)

    indices, displayed = downsample_for_display(values, max_points=500)

    assert np.array_equal(displayed, values)
    assert np.array_equal(indices, np.arange(50))


def test_the_spectrum_finds_a_real_tone_at_the_right_offset() -> None:
    time = np.arange(4_096) / FS
    tone = np.exp(2j * np.pi * 25_000.0 * time).astype(np.complex64)

    frequencies, magnitude = window_spectrum(tone, FS)

    assert frequencies[int(np.argmax(magnitude))] == pytest.approx(25_000.0, abs=200.0)


def test_the_spectrogram_axes_describe_the_selection() -> None:
    time = np.arange(8_192) / FS
    tone = np.exp(2j * np.pi * 10_000.0 * time).astype(np.complex64)

    times, frequencies, power = window_spectrogram(tone, FS)

    assert times.size == power.shape[0]
    assert frequencies.size == power.shape[1]
    assert times.min() >= 0.0
    assert power.max() == pytest.approx(0.0, abs=1e-6), "normalised to a 0 dB peak"


# --- measurements must not claim what was not measured ------------------------------------------


def test_window_measurements_report_only_window_quantities() -> None:
    measurements = window_measurements(_ramp(2_000), FS)

    assert {"samples", "rms_amplitude", "peak_amplitude"} <= set(measurements)
    for forbidden in ("modulation", "symbol_rate_hz", "snr_db", "carrier_frequency_hz",
                      "occupied_bandwidth_hz"):
        assert forbidden not in measurements, (
            f"{forbidden} is a whole-capture pipeline result and must not be "
            "recomputed per window")


def test_measurements_are_empty_for_an_empty_window() -> None:
    assert window_measurements(np.empty(0, dtype=np.complex64), FS) == {}


# --- IQ scatter honesty ---------------------------------------------------------------------------


def test_symbol_decimation_needs_a_symbol_rate() -> None:
    window = _ramp(1_000)

    assert symbol_rate_decimated(window, FS, None).size == 0
    assert symbol_rate_decimated(window, FS, 0.0).size == 0


def test_symbol_decimation_uses_the_reported_rate() -> None:
    window = _ramp(1_000)

    decimated = symbol_rate_decimated(window, FS, 25_000.0)  # 8 samples per symbol

    assert decimated.size == 125
    assert np.array_equal(decimated, window[::8])


# --- playback is timeline movement ------------------------------------------------------------------


def test_the_cursor_advances_by_step_times_speed() -> None:
    assert advance_cursor(0.10, 1.0, step_seconds=0.01, speed=2.0) == pytest.approx(0.12)


def test_playback_loops_at_the_end() -> None:
    assert advance_cursor(0.99, 1.0, step_seconds=0.05, speed=1.0, loop=True) == 0.0


def test_playback_can_stop_at_the_end_instead_of_looping() -> None:
    assert advance_cursor(0.99, 1.0, step_seconds=0.05, loop=False) == pytest.approx(1.0)


def test_playback_is_inert_without_a_duration() -> None:
    assert advance_cursor(0.5, None, step_seconds=0.01) == 0.0


# --- real ingestion containers ------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["iq", "wav"])
def test_a_unified_signal_container_drives_the_controller(source: str) -> None:
    signal = UnifiedSignalContainer(_ramp(5_000), FS, source)

    selection = select_window(signal.iq.size, signal.sample_rate, 0.01,
                              DEFAULT_WINDOW_SECONDS)
    window = extract_window(signal.iq, selection)

    assert window.size > 0
    assert np.array_equal(window,
                          signal.iq[selection.start_sample:selection.end_sample])


# --- axis formatting & readouts --------------------------------------------------------


def test_axis_units_scale_intelligently() -> None:
    from radiofry.explore.time_machine import (
        format_frequency_axis_unit,
        format_smart_frequency,
        format_smart_time,
        format_time_axis_unit,
    )

    assert format_time_axis_unit(2.5) == (1.0, "s")
    assert format_time_axis_unit(0.02) == (1000.0, "ms")
    assert format_time_axis_unit(0.0005) == (1_000_000.0, "µs")

    assert format_frequency_axis_unit(5_000_000.0) == (1e-6, "MHz")
    assert format_frequency_axis_unit(25_000.0) == (1e-3, "kHz")
    assert format_frequency_axis_unit(500.0) == (1.0, "Hz")

    assert format_smart_time(0.025, span_seconds=0.025) == "25.00 ms"
    assert format_smart_time(1.2345, span_seconds=2.0) == "1.2345 s"
    assert format_smart_frequency(12_400.0) == "+12.40 kHz"
    assert format_smart_frequency(-50_000.0) == "-50.00 kHz"
    assert format_smart_frequency(0.0) == "0.0 Hz"


def test_window_measurements_reuses_precomputed_spectrum() -> None:
    window = _ramp(2_000)
    freqs, mag = window_spectrum(window, FS)
    meas = window_measurements(window, FS, spectrum_data=(freqs, mag))

    assert "peak_offset_hz" in meas
    assert meas["samples"] == 2000.0


# --- markers & delta measurements ------------------------------------------------------


def test_calculate_marker_delta() -> None:
    from radiofry.explore.time_machine import calculate_marker_delta

    delta = calculate_marker_delta(
        t1=0.010, f1=10_000.0, p1=-30.0,
        t2=0.020, f2=25_000.0, p2=-20.0,
    )

    assert delta["delta_time_s"] == pytest.approx(0.010)
    assert delta["delta_time_str"] == "10.00 ms"
    assert delta["pri_freq_hz"] == pytest.approx(100.0)
    assert delta["delta_freq_hz"] == pytest.approx(15_000.0)
    assert delta["delta_freq_str"] == "15.00 kHz"
    assert delta["delta_power_db"] == pytest.approx(10.0)
    assert delta["delta_power_str"] == "+10.0 dB"


# --- candidate burst detector ----------------------------------------------------------


def test_detect_candidate_bursts_finds_active_regions() -> None:
    from radiofry.explore.time_machine import detect_candidate_bursts

    # Create signal with quiet noise and two high-power bursts
    np.random.seed(42)
    sig = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.01  # Noise floor
    # Burst 1: samples 5000 to 10000 (25 ms)
    sig[5_000:10_000] += (np.random.randn(5_000) + 1j * np.random.randn(5_000)) * 1.0
    # Burst 2: samples 20000 to 26000 (30 ms)
    sig[20_000:26_000] += (np.random.randn(6_000) + 1j * np.random.randn(6_000)) * 1.0

    bursts = detect_candidate_bursts(sig, FS, threshold_db_above_noise=6.0, min_duration_s=0.01)

    assert len(bursts) >= 2
    b1 = bursts[0]
    assert b1["start_seconds"] == pytest.approx(5_000 / FS, abs=0.005)
    assert b1["end_seconds"] == pytest.approx(10_000 / FS, abs=0.005)
    assert b1["peak_snr_db"] > 10.0


# --- two-region comparison -------------------------------------------------------------


def test_compare_two_regions() -> None:
    from radiofry.explore.time_machine import compare_two_regions

    sig = UnifiedSignalContainer(_ramp(10_000), FS, "iq")
    sel_a = select_window(10_000, FS, 0.01, 0.01)
    sel_b = select_window(10_000, FS, 0.03, 0.01)

    rows = compare_two_regions(sig, sel_a, sel_b)

    assert len(rows) >= 4
    field_names = [r[0] for r in rows]
    assert "Duration" in field_names
    assert "RMS Amplitude" in field_names
    assert "Peak Amplitude" in field_names


def test_compare_two_regions_with_investigation_dicts() -> None:
    from radiofry.explore.time_machine import compare_two_regions

    inv_a = {
        "name": "Burst Alpha",
        "start_s": 0.01,
        "end_s": 0.05,
        "samples": 8_000,
        "report": {
            "stages": {
                "fusion": {"label": "QPSK", "trust_score": 0.88},
                "parameters": {"snr_db": 18.2, "occupied_bandwidth_hz": 25_000.0}
            }
        }
    }
    inv_b = {
        "name": "Burst Beta",
        "start_s": 0.10,
        "end_s": 0.16,
        "samples": 12_000,
        "report": {
            "stages": {
                "fusion": {"label": "16QAM", "trust_score": 0.75},
                "parameters": {"snr_db": 15.0, "occupied_bandwidth_hz": 30_000.0}
            }
        }
    }

    rows = compare_two_regions(inv_a, inv_b)
    names = [r[0] for r in rows]
    assert "Duration" in names
    assert "Samples" in names
    assert "Modulation" in names
    assert "Modulation Trust" in names
    assert "Estimated SNR" in names
    assert "Occupied Bandwidth" in names


def test_calculate_marker_delta_full_measurements() -> None:
    from radiofry.explore.time_machine import calculate_marker_delta

    deltas = calculate_marker_delta(
        time_a=0.010, time_b=0.025,
        freq_a=-5_000.0, freq_b=15_000.0,
        power_a=-30.0, power_b=-22.5
    )

    assert deltas["delta_time_s"] == pytest.approx(0.015)
    assert deltas["time_frequency_hz"] == pytest.approx(1.0 / 0.015)
    assert deltas["delta_freq_hz"] == pytest.approx(20_000.0)
    assert deltas["delta_power_db"] == pytest.approx(7.5)


def test_axis_formatters_handle_arrays_and_scalars() -> None:
    from radiofry.explore.time_machine import (
        format_frequency_axis_unit, format_time_axis_unit,
        format_smart_time, format_smart_frequency
    )

    freqs = np.array([-50_000.0, 0.0, 50_000.0])
    mult, unit = format_frequency_axis_unit(freqs)
    assert unit == "kHz"
    assert mult == pytest.approx(1e-3)

    mult_mhz, unit_mhz = format_frequency_axis_unit(2_500_000.0)
    assert unit_mhz == "MHz"
    assert mult_mhz == pytest.approx(1e-6)

    t_mult, t_unit = format_time_axis_unit(0.005)
    assert t_unit == "ms"
    assert t_mult == pytest.approx(1000.0)

    assert format_smart_time(0.0125) == "12.50 ms"
    assert format_smart_frequency(4_820.0) == "+4.82 kHz"
    assert format_smart_frequency(-12_500.0) == "-12.50 kHz"


