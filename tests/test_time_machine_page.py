"""The Signal Time Machine page actually renders, with and without a capture.

These drive the real Streamlit script through `AppTest`, so they catch the failures unit
tests on the controller cannot: an exception during render, a control that does not move
the cursor, or a freeze that does not hold the window.

Streamlit is a `gui` extra, so the whole module skips when it is not installed.
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="streamlit is an optional gui extra")

from streamlit.testing.v1 import AppTest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE = PROJECT_ROOT / "gui" / "pages" / "9_time_machine.py"
V1_CAPTURES = PROJECT_ROOT / "data" / "synthetic_v1" / "captures"

# Pages import `gui.theme`, which normally resolves because gui/app.py puts the project
# root on sys.path before any page runs. AppTest loads a page directly, so do it here.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _capture(extension: str):
    from radiofry.ingestion.iq_parser import IQFormat
    from radiofry.pipeline import load_capture

    truth = json.loads((V1_CAPTURES / "QPSK_snr20dB_r000.json").read_text(encoding="utf-8"))
    entry = next(f for f in truth["files"] if f["file_format"] == "iq")
    path = V1_CAPTURES / f"QPSK_snr20dB_r000{extension}"
    if extension == ".iq":
        return load_capture(path, sample_rate=truth["signal"]["sample_rate_hz"],
                            iq_format=IQFormat(entry["dtype"], entry["byte_order"]))
    return load_capture(path)


def _page(signal=None, report=None) -> "AppTest":
    app = AppTest.from_file(str(PAGE), default_timeout=180)
    if signal is not None:
        app.session_state["signal"] = signal
        app.session_state["report"] = report or {}
    return app.run()


requires_v1 = pytest.mark.skipif(not V1_CAPTURES.exists(),
                                 reason="frozen V1 dataset not present")


# --- renders at all -----------------------------------------------------------------------


def test_the_page_renders_without_a_capture() -> None:
    app = _page()

    assert not app.exception, [e.value for e in app.exception]


@requires_v1
@pytest.mark.parametrize("extension", [".iq", ".wav"])
def test_the_page_renders_for_iq_and_wav_captures(extension: str) -> None:
    app = _page(_capture(extension))

    assert not app.exception, [e.value for e in app.exception]


@requires_v1
def test_the_timeline_controls_are_present() -> None:
    app = _page(_capture(".iq"))

    assert len(app.button) >= 2, "play and pause"
    assert len(app.slider) >= 1, "position scrubber"
    assert len(app.toggle) >= 1, "freeze"
    assert {"Cursor", "Window", "Window samples", "Total duration"} <= {
        m.label for m in app.metric}


# --- the controls actually move the window ---------------------------------------------------


@requires_v1
def test_scrubbing_moves_the_cursor() -> None:
    app = _page(_capture(".iq"))
    before = [m.value for m in app.metric if m.label == "Cursor"][0]

    app.slider[0].set_value(0.08).run()

    after = [m.value for m in app.metric if m.label == "Cursor"][0]
    assert not app.exception, [e.value for e in app.exception]
    assert after != before
    assert after.startswith("0.0800")


@requires_v1
def test_freezing_holds_the_window_and_locks_the_scrubber() -> None:
    """Freeze disables the position controls, so a browser user cannot move the window.

    AppTest refuses to drive a disabled widget, which is exactly the guarantee wanted
    here: the assertion is that the control is unusable while frozen and the cursor is
    unchanged, not that moving it silently has no effect.
    """
    app = _page(_capture(".iq"))
    app.slider[0].set_value(0.05).run()
    frozen_cursor = [m.value for m in app.metric if m.label == "Cursor"][0]

    app.toggle[0].set_value(True).run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_frozen"] is True
    assert app.slider[0].disabled, "the scrubber must be locked while frozen"
    held = [m.value for m in app.metric if m.label == "Cursor"][0]
    assert held == frozen_cursor, "a frozen window must keep its position"


@requires_v1
def test_unfreezing_releases_the_scrubber() -> None:
    app = _page(_capture(".iq"))
    app.toggle[0].set_value(True).run()

    app.toggle[0].set_value(False).run()

    assert app.session_state["stm_frozen"] is False
    assert not app.slider[0].disabled


@requires_v1
def test_freeze_is_visually_announced() -> None:
    app = _page(_capture(".iq"))

    app.toggle[0].set_value(True).run()

    assert any("FROZEN" in str(item.value) for item in app.info)


@requires_v1
def test_playback_advances_the_cursor_and_stops_at_the_end() -> None:
    """Playback is bounded: it runs to the end of the recording and stops.

    The cursor is parked near the end first so this needs only one advance. An
    unbounded loop here would hang the test - and would equally spin the browser.
    """
    app = _page(_capture(".iq"))
    duration = app.session_state["stm_capture_key"][0] / app.session_state[
        "stm_capture_key"][1]
    app.slider[0].set_value(round(duration * 0.99, 4)).run()

    next(b for b in app.button if "Play" in b.label).click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_playing"] is False, "playback must stop at the end"
    assert app.session_state["stm_cursor"] == pytest.approx(duration, rel=1e-3)


# --- honesty: whole-capture values are labelled as such -------------------------------------------


@requires_v1
def test_the_page_separates_whole_capture_results_from_the_window() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))

    rendered = " ".join(str(block.value) for block in app.markdown)
    assert "Whole-capture analysis" in rendered
    assert "Selected window" in rendered
    assert "not</b> recomputed" in rendered or "not recomputed" in rendered


@requires_v1
def test_the_iq_scatter_is_not_called_a_decoded_constellation() -> None:
    signal = _capture(".iq")
    app = _page(signal, {})

    rendered = " ".join(str(block.value) for block in app.markdown)
    captions = " ".join(str(c.value) for c in app.caption)
    assert "IQ Scatter" in rendered
    assert "not a decoded constellation" in (rendered + captions).lower() or \
           "not a decoded constellation" in captions.lower()


# --- region investigation (Time Machine V2) ---------------------------------------------------


@requires_v1
def test_the_region_controls_and_analyze_button_render() -> None:
    app = _page(_capture(".iq"))

    labels = {b.label for b in app.button}
    numbers = {n.label for n in app.number_input}
    assert "Analyze Selection" in labels
    assert {"Region start (s)", "Region end (s)"} <= numbers


@requires_v1
def test_a_reversed_region_warns_and_hides_the_analyze_button() -> None:
    app = _page(_capture(".iq"))

    app.number_input(key="stm_region_start").set_value(0.10).run()
    app.number_input(key="stm_region_end").set_value(0.02).run()

    assert not app.exception, [e.value for e in app.exception]
    assert any("before it starts" in str(w.value) for w in app.warning)
    assert "Analyze Selection" not in {b.label for b in app.button}, (
        "an invalid region must not be analysable")


@requires_v1
def test_a_region_shorter_than_one_frame_is_refused() -> None:
    app = _page(_capture(".iq"))

    app.number_input(key="stm_region_start").set_value(0.0100).run()
    app.number_input(key="stm_region_end").set_value(0.0102).run()

    assert not app.exception, [e.value for e in app.exception]
    assert any("128" in str(w.value) for w in app.warning)


@requires_v1
def test_analyzing_a_region_runs_the_pipeline_and_shows_a_comparison() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app.number_input(key="stm_region_start").set_value(0.02).run()
    app.number_input(key="stm_region_end").set_value(0.09).run()

    next(b for b in app.button if b.label == "Analyze Selection").click().run()

    assert not app.exception, [e.value for e in app.exception]
    investigations = app.session_state["stm_investigations"]
    assert len(investigations) == 1
    stored = investigations[0]
    assert stored["report"]["source"]["samples"] == stored["samples"]
    assert stored["samples"] < signal.iq.size, "the region, not the whole capture"
    rendered = " ".join(str(block.value) for block in app.markdown)
    assert "Whole capture vs selected region" in rendered


@requires_v1
def test_the_whole_capture_panel_survives_a_region_analysis() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app.number_input(key="stm_region_start").set_value(0.02).run()
    app.number_input(key="stm_region_end").set_value(0.09).run()

    next(b for b in app.button if b.label == "Analyze Selection").click().run()

    rendered = " ".join(str(block.value) for block in app.markdown)
    assert "Whole-capture analysis" in rendered, "global results must stay visible"
    assert "Selected window" in rendered


@requires_v1
def test_two_regions_can_be_investigated_in_turn() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))

    for start, end in ((0.01, 0.05), (0.09, 0.14)):
        app.number_input(key="stm_region_start").set_value(start).run()
        app.number_input(key="stm_region_end").set_value(end).run()
        next(b for b in app.button if b.label == "Analyze Selection").click().run()

    assert not app.exception, [e.value for e in app.exception]
    investigations = app.session_state["stm_investigations"]
    assert len(investigations) == 2
    assert investigations[0]["start_sample"] != investigations[1]["start_sample"]


@requires_v1
def test_scrubbing_alone_never_triggers_an_analysis() -> None:
    """Timeline movement is visualisation only; analysis is explicit."""
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))

    app.slider[0].set_value(0.05).run()
    app.number_input(key="stm_region_start").set_value(0.01).run()

    assert app.session_state["stm_investigations"] == []


# --- visual region selection (Time Machine V3) --------------------------------------------


def _drag(app: "AppTest", start: float, end: float) -> "AppTest":
    """Post a box-selection event the way the overview chart delivers one.

    `AppTest` cannot drive a Plotly chart, so the event payload is written into the
    chart's session_state slot in the shape Streamlit stores it. That is the same input
    the page reads on a real drag, so the page-side conversion is exercised for real.
    """
    app.session_state["stm_overview_chart"] = {
        "selection": {"points": [], "box": [{"x": [start, end], "y": [-1.0, 1.0]}],
                      "lasso": []}}
    return app.run()


@requires_v1
def test_the_overview_selection_surface_renders() -> None:
    app = _page(_capture(".iq"))

    rendered = " ".join(str(block.value) for block in app.markdown)
    captions = " ".join(str(c.value) for c in app.caption)
    assert not app.exception, [e.value for e in app.exception]
    assert "Select a region" in rendered
    assert "Drag horizontally" in captions


@requires_v1
def test_dragging_sets_the_numeric_region_controls() -> None:
    """The drag and the numbers are one selection: the drag must move the numbers."""
    app = _page(_capture(".iq"))

    app = _drag(app, 0.030, 0.085)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.030, abs=1e-4)
    assert app.session_state["stm_region_end"] == pytest.approx(0.085, abs=1e-4)
    assert app.number_input(key="stm_region_start").value == pytest.approx(0.030,
                                                                          abs=1e-4)
    assert app.number_input(key="stm_region_end").value == pytest.approx(0.085,
                                                                         abs=1e-4)


@requires_v1
def test_a_right_to_left_drag_is_ordered_not_rejected() -> None:
    app = _page(_capture(".iq"))

    app = _drag(app, 0.090, 0.020)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.020, abs=1e-4)
    assert app.session_state["stm_region_end"] == pytest.approx(0.090, abs=1e-4)
    assert not any("before it starts" in str(w.value) for w in app.warning)


@requires_v1
def test_a_drag_past_the_end_is_clamped_to_the_capture() -> None:
    app = _page(_capture(".iq"))
    duration = app.session_state["stm_capture_key"][0] / app.session_state[
        "stm_capture_key"][1]

    app = _drag(app, 0.05, duration + 5.0)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_end"] <= duration + 1e-9
    assert "Analyze Selection" in {b.label for b in app.button}


@requires_v1
def test_a_too_short_drag_is_refused_the_same_way_as_typed_bounds() -> None:
    """A flick too small to analyse is refused, and says so, instead of silently
    producing a region the pipeline cannot use. 20 us is 4 samples at 200 kHz, far
    under the 128 a frame needs - the identical bound typed bounds are held to.

    The previous selection deliberately survives: the refusal is announced and the
    highlight keeps marking the region that is still selected, so the Analyze button
    continues to describe what is actually on screen rather than the rejected flick.
    """
    app = _page(_capture(".iq"))
    before = (app.session_state["stm_region_start"],
              app.session_state["stm_region_end"])

    app = _drag(app, 0.05000, 0.05002)

    assert not app.exception, [e.value for e in app.exception]
    assert any("could not be used" in str(w.value) for w in app.warning), (
        "an unusable drag must be explained, not ignored")
    assert (app.session_state["stm_region_start"],
            app.session_state["stm_region_end"]) == before, (
        "a refused drag must leave the existing selection untouched")
    assert app.session_state["stm_investigations"] == []


@requires_v1
def test_dragging_never_runs_an_analysis() -> None:
    """The hard rule: a selection gesture must not invoke the pipeline."""
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))

    app = _drag(app, 0.02, 0.09)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_investigations"] == [], (
        "dragging must only move the selection, never analyse it")


@requires_v1
def test_a_dragged_region_can_then_be_analysed_by_the_button() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.02, 0.09)

    next(b for b in app.button if b.label == "Analyze Selection").click().run()

    assert not app.exception, [e.value for e in app.exception]
    investigations = app.session_state["stm_investigations"]
    assert len(investigations) == 1
    assert investigations[0]["start_s"] == pytest.approx(0.02, abs=1e-4)
    assert investigations[0]["samples"] < signal.iq.size


@requires_v1
def test_typing_bounds_after_a_drag_wins() -> None:
    """A stale selection event must not be re-applied over what the analyst typed.

    Streamlit keeps the chart's selection in session_state across reruns, so without a
    guard the drag would overwrite the numeric controls on every subsequent rerun - the
    same class of widget-state fight as the earlier freeze bug.
    """
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)

    app.number_input(key="stm_region_start").set_value(0.005).run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.005, abs=1e-4)
    assert app.session_state["stm_region_end"] == pytest.approx(0.085, abs=1e-4)


@requires_v1
def test_using_the_current_window_still_overrides_a_previous_drag() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app.slider[0].set_value(0.12).run()

    next(b for b in app.button if b.label == "Use current window").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.12, abs=2e-3)


@requires_v1
def test_the_waveform_reports_the_same_region_the_overview_selected() -> None:
    """Both views must describe one region, and say which samples that is.

    The window shown at cursor 0 spans 0-20 ms, so a drag inside it must be reported as
    covering that window - by the same sample numbers the region controls resolved.
    """
    app = _page(_capture(".iq"))

    app = _drag(app, 0.005, 0.015)

    captions = " ".join(str(c.value) for c in app.caption)
    assert not app.exception, [e.value for e in app.exception]
    assert "Shaded: investigation region" in captions
    assert "1,000-3,000" in captions, (
        "the waveform must name the same samples the drag resolved to")


@requires_v1
def test_a_region_outside_the_current_window_is_said_to_be_outside() -> None:
    """The playback window and the investigation region stay distinguishable."""
    app = _page(_capture(".iq"))

    app = _drag(app, 0.100, 0.140)

    captions = " ".join(str(c.value) for c in app.caption)
    assert not app.exception, [e.value for e in app.exception]
    assert "lies outside this window" in captions
    assert "Shaded: investigation region" not in captions


# --- signal lock / focus (Time Machine V4) --------------------------------------------


def _labels(app: "AppTest") -> set:
    return {b.label for b in app.button}


def _lock(app: "AppTest") -> "AppTest":
    return next(b for b in app.button if b.label == "Lock Signal").click().run()


def _unlock(app: "AppTest") -> "AppTest":
    return next(b for b in app.button if b.label == "Unlock Signal").click().run()


def _analyze(app: "AppTest") -> "AppTest":
    return next(b for b in app.button if b.label == "Analyze Selection").click().run()


def _banner(app: "AppTest") -> str:
    return str(next(b for b in app.markdown if "Signal locked" in str(b.value)).value)


@requires_v1
def test_a_valid_region_can_be_locked() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    assert "Lock Signal" in _labels(app)

    app = _lock(app)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_locked"] is True
    assert "Unlock Signal" in _labels(app)
    assert "Lock Signal" not in _labels(app), "lock and unlock are one control"


@requires_v1
def test_an_invalid_region_offers_no_lock() -> None:
    """Never lock a region the pipeline could not analyse."""
    app = _page(_capture(".iq"))

    app.number_input(key="stm_region_start").set_value(0.10).run()
    app.number_input(key="stm_region_end").set_value(0.02).run()

    assert not app.exception, [e.value for e in app.exception]
    assert "Lock Signal" not in _labels(app)
    assert app.session_state["stm_locked"] is False


@requires_v1
def test_the_lock_survives_reruns() -> None:
    """The investigation target must not evaporate on the next interaction."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    bounds = (app.session_state["stm_region_start"],
              app.session_state["stm_region_end"])

    for _ in range(3):
        app = app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_locked"] is True
    assert (app.session_state["stm_region_start"],
            app.session_state["stm_region_end"]) == bounds


@requires_v1
def test_unlocking_releases_the_lock_and_keeps_the_region() -> None:
    """Unlock frees the controls; it does not throw the region away."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _unlock(app)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_locked"] is False
    assert app.session_state["stm_region_start"] == pytest.approx(0.030, abs=1e-4)
    assert "Lock Signal" in _labels(app)


@requires_v1
def test_the_locked_region_shows_its_bounds_as_timestamps() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)

    app = _lock(app)

    rendered = _banner(app)
    assert "00:00.030" in rendered and "00:00.085" in rendered
    assert "6,000" in rendered and "17,000" in rendered, "sample bounds must be shown"


# --- the lock protects the region --------------------------------------------------------


@requires_v1
def test_scrubbing_does_not_move_the_locked_region() -> None:
    """A lock is a focus mechanism: the timeline still moves, the target does not."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app.slider[0].set_value(0.13).run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_cursor"] == pytest.approx(0.13, abs=2e-3), (
        "the playback window must still move while locked")
    assert app.session_state["stm_region_start"] == pytest.approx(0.030, abs=1e-4)
    assert app.session_state["stm_region_end"] == pytest.approx(0.085, abs=1e-4)
    assert app.session_state["stm_locked"] is True


@requires_v1
def test_dragging_does_not_replace_a_locked_region() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _drag(app, 0.120, 0.150)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.030, abs=1e-4)
    assert app.session_state["stm_region_end"] == pytest.approx(0.085, abs=1e-4)
    assert any("locked" in str(w.value) for w in app.warning), (
        "a refused drag must explain why")


@requires_v1
def test_a_drag_refused_by_the_lock_does_not_spring_back_after_unlocking() -> None:
    """The stale-state failure mode: a gesture refused while locked stays refused."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _drag(app, 0.120, 0.150)

    app = _unlock(app)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.030, abs=1e-4), (
        "unlocking must not apply the drag that was refused while locked")
    assert app.session_state["stm_region_end"] == pytest.approx(0.085, abs=1e-4)


@requires_v1
def test_the_numeric_controls_are_locked_too() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)

    app = _lock(app)

    assert app.number_input(key="stm_region_start").disabled
    assert app.number_input(key="stm_region_end").disabled
    assert next(b for b in app.button if b.label == "Use current window").disabled


@requires_v1
def test_unlocking_allows_a_new_region_to_be_selected() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _unlock(app)

    app = _drag(app, 0.120, 0.150)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.120, abs=1e-4)
    assert app.session_state["stm_region_end"] == pytest.approx(0.150, abs=1e-4)


@requires_v1
def test_numeric_editing_still_syncs_after_an_unlock() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _unlock(app)

    app.number_input(key="stm_region_start").set_value(0.012).run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_start"] == pytest.approx(0.012, abs=1e-4)


# --- the lock is cheap ---------------------------------------------------------------------


@requires_v1
def test_locking_and_unlocking_never_run_an_analysis() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)

    app = _lock(app)
    app.slider[0].set_value(0.13).run()
    app = _unlock(app)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_investigations"] == [], (
        "lock, scrub and unlock must not invoke the pipeline")


@requires_v1
def test_an_unanalysed_locked_region_claims_nothing() -> None:
    """No borrowed identity: the target has no modulation until one is measured."""
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)

    app = _lock(app)

    assert "Not analyzed yet" in _banner(app)


# --- the lock and the pipeline -------------------------------------------------------------


@requires_v1
def test_analyze_selection_runs_on_exactly_the_locked_samples() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    fs = signal.sample_rate

    app = _analyze(app)

    assert not app.exception, [e.value for e in app.exception]
    stored = app.session_state["stm_investigations"][-1]
    assert stored["start_sample"] == int(round(0.030 * fs))
    assert stored["end_sample"] == int(round(0.085 * fs))
    assert stored["report"]["source"]["samples"] == stored["samples"]
    assert app.session_state["stm_locked"] is True, "analysis must not release the lock"


@requires_v1
def test_the_locked_banner_reports_the_analysis_once_it_has_run() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _analyze(app)

    banner = _banner(app)
    assert "Not analyzed yet" not in banner
    assert "Analyzed" in banner


@requires_v1
def test_an_analysis_of_a_different_region_is_not_claimed_by_the_lock() -> None:
    """Analysing region A then locking region B must not label B with A's result."""
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.010, 0.050)
    app = _analyze(app)

    app = _drag(app, 0.090, 0.140)
    app = _lock(app)

    assert "Not analyzed yet" in _banner(app), (
        "a different region's analysis must not be attributed to this one")


# --- focus, and the rest of the page still working -------------------------------------------


@requires_v1
def test_focus_moves_the_playback_window_onto_the_region_without_changing_it() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.090, 0.140)
    app = _lock(app)

    app = next(b for b in app.button if b.label == "Focus").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_cursor"] == pytest.approx(0.090, abs=2e-3)
    assert app.session_state["stm_window_ms"] == 50.0, (
        "the smallest offered window that covers a 50 ms region")
    assert app.session_state["stm_region_start"] == pytest.approx(0.090, abs=1e-4), (
        "focus moves the view, not the region")
    assert app.session_state["stm_locked"] is True


@requires_v1
def test_playback_still_runs_while_a_signal_is_locked() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    duration = app.session_state["stm_capture_key"][0] / app.session_state[
        "stm_capture_key"][1]
    app.slider[0].set_value(round(duration * 0.99, 4)).run()

    app = next(b for b in app.button if "Play" in b.label).click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_playing"] is False, "playback still stops at the end"
    assert app.session_state["stm_locked"] is True, "playback must not release the lock"
    assert app.session_state["stm_region_start"] == pytest.approx(0.030, abs=1e-4)


@requires_v1
def test_freeze_still_works_while_a_signal_is_locked() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app.toggle[0].set_value(True).run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_frozen"] is True
    assert app.slider[0].disabled
    assert app.session_state["stm_locked"] is True


# --- lock safety: degenerate captures and capture swaps ---------------------------------


def _synthetic(samples: int, sample_rate):
    import numpy as np

    from radiofry.contracts import UnifiedSignalContainer

    iq = (np.arange(samples, dtype=np.float64)
          + 1j * np.arange(samples, dtype=np.float64)).astype(np.complex64)
    return UnifiedSignalContainer(iq, sample_rate, "iq")


def test_a_capture_without_a_sample_rate_offers_no_lock() -> None:
    """No time axis means no region, and therefore nothing that could be locked."""
    app = _page(_synthetic(4_096, None))

    assert not app.exception, [e.value for e in app.exception]
    assert "Lock Signal" not in _labels(app)
    assert app.session_state["stm_locked"] is False


def test_a_capture_too_short_for_a_region_offers_no_lock() -> None:
    app = _page(_synthetic(64, 200_000.0))

    assert not app.exception, [e.value for e in app.exception]
    assert "Lock Signal" not in _labels(app)
    assert app.session_state["stm_locked"] is False


@requires_v1
def test_loading_a_different_capture_releases_the_lock() -> None:
    """A lock names samples in one recording; it must not survive into another."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    assert app.session_state["stm_locked"] is True

    app.session_state["signal"] = _synthetic(8_192, 48_000.0)
    app = app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_locked"] is False, (
        "a lock must not carry over to a different recording")


@requires_v1
def test_a_lock_can_be_released_after_the_region_was_analysed() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _analyze(app)

    app = _unlock(app)

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_locked"] is False
    assert len(app.session_state["stm_investigations"]) == 1, (
        "unlocking must not discard a completed analysis")


# --- naming a locked investigation ---------------------------------------------------------


def _name(app: "AppTest", text: str) -> "AppTest":
    return app.text_input(key="stm_region_name").set_value(text).run()


@requires_v1
def test_the_name_field_is_offered_for_a_valid_region() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)

    assert not app.exception, [e.value for e in app.exception]
    assert app.text_input(key="stm_region_name").value == ""


@requires_v1
def test_an_unnamed_locked_region_shows_the_numbered_default() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)

    app = _lock(app)

    assert "Investigation 1" in _banner(app)


@requires_v1
def test_a_custom_name_appears_in_the_locked_banner() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _name(app, "Possible frequency shift")

    assert not app.exception, [e.value for e in app.exception]
    banner = _banner(app)
    assert "Possible frequency shift" in banner
    assert "Investigation 1" not in banner


@requires_v1
def test_a_whitespace_only_name_still_shows_the_default() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _name(app, "   ")

    assert "Investigation 1" in _banner(app)


@requires_v1
def test_the_banner_keeps_its_timestamps_and_sample_range_when_named() -> None:
    """A name is added to the banner, it does not replace the evidence on it."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _name(app, "Burst 3")

    banner = _banner(app)
    assert "Burst 3" in banner
    assert "00:00.030" in banner and "00:00.085" in banner
    assert "6,000" in banner and "17,000" in banner
    assert "Duration 0.055 s" in banner


@requires_v1
def test_the_name_survives_reruns() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _name(app, "Burst 3")

    for _ in range(3):
        app = app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_name"] == "Burst 3"
    assert "Burst 3" in _banner(app)


@requires_v1
def test_naming_never_moves_the_region() -> None:
    """The one thing a label must never do."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    before = (app.session_state["stm_region_start"],
              app.session_state["stm_region_end"])

    app = _name(app, "Burst 3")
    app = _name(app, "Renamed again")

    assert not app.exception, [e.value for e in app.exception]
    assert (app.session_state["stm_region_start"],
            app.session_state["stm_region_end"]) == before
    assert app.session_state["stm_locked"] is True


@requires_v1
def test_naming_never_runs_an_analysis() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _name(app, "Burst 3")

    assert app.session_state["stm_investigations"] == []


# --- the name reaches the history ------------------------------------------------------------


@requires_v1
def test_a_custom_name_is_stored_with_the_analysis_and_shown_in_the_history() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _name(app, "Interesting QPSK region")

    app = _analyze(app)

    assert not app.exception, [e.value for e in app.exception]
    stored = app.session_state["stm_investigations"][-1]
    assert stored["name"] == "Interesting QPSK region"
    assert stored["start_sample"] == 6_000 and stored["end_sample"] == 17_000, (
        "the name must not disturb the analysed bounds")
    options = " ".join(str(o) for o in app.selectbox[0].options)
    captions = " ".join(str(c.value) for c in app.caption)
    assert "Interesting QPSK region" in (options + captions)


@requires_v1
def test_an_unnamed_analysis_is_stored_with_the_numbered_default() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)

    app = _analyze(app)

    assert app.session_state["stm_investigations"][-1]["name"] == "Investigation 1"


@requires_v1
def test_two_named_investigations_stay_distinguishable_in_the_history() -> None:
    """The reason the field exists: telling two entries apart at a glance."""
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))

    app = _drag(app, 0.010, 0.050)
    app = _name(app, "Burst 3")
    app = _analyze(app)

    app = _drag(app, 0.090, 0.140)
    app = _name(app, "Possible frequency shift")
    app = _analyze(app)

    assert not app.exception, [e.value for e in app.exception]
    names = [item["name"] for item in app.session_state["stm_investigations"]]
    assert names == ["Burst 3", "Possible frequency shift"]
    options = " ".join(str(o) for o in app.selectbox[0].options)
    assert "Burst 3" in options and "Possible frequency shift" in options
    bounds = [(i["start_sample"], i["end_sample"])
              for i in app.session_state["stm_investigations"]]
    assert bounds == [(2_000, 10_000), (18_000, 28_000)], (
        "each entry keeps its own real sample bounds")


@requires_v1
def test_the_history_row_still_carries_timestamps_and_sample_count() -> None:
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)
    app = _name(app, "Burst 3")

    app = _analyze(app)

    row = str(app.selectbox[0].options[0])
    assert "Burst 3" in row
    assert "0.0300-0.0850 s" in row
    assert "11,000 samples" in row


@requires_v1
def test_clearing_the_box_after_an_analysis_keeps_the_stored_name() -> None:
    """Emptying the field must not silently renumber a completed investigation."""
    from radiofry.pipeline import analyze_capture

    signal = _capture(".iq")
    app = _page(signal, analyze_capture(signal))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _name(app, "Burst 3")
    app = _analyze(app)

    app = _name(app, "")

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_investigations"][0]["name"] == "Burst 3"
    assert "Burst 3" in _banner(app)


@requires_v1
def test_a_name_containing_markup_is_escaped_not_rendered() -> None:
    """The banner is drawn with unsafe_allow_html, so the label must be escaped."""
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)

    app = _name(app, "<b>burst</b>")

    banner = _banner(app)
    assert "&lt;b&gt;burst&lt;/b&gt;" in banner
    assert "<b>burst</b>" not in banner


@requires_v1
def test_loading_a_different_capture_clears_the_name() -> None:
    app = _page(_capture(".iq"))
    app = _drag(app, 0.030, 0.085)
    app = _lock(app)
    app = _name(app, "Burst 3")

    app.session_state["signal"] = _synthetic(8_192, 48_000.0)
    app = app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["stm_region_name"] == ""
