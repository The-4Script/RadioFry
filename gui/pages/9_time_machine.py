"""Signal Time Machine: move through the real recording and inspect it.

Every view on this page is derived from one `WindowSelection` produced by
`radiofry.explore.time_machine`, so the waveform, spectrum, waterfall and IQ scatter
always describe the same region of the same capture. The page is a thin rendering shell:
the window arithmetic and view data live in the tested module, not here.
"""

import html
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gui.theme import (
    render_empty_state, render_method_note, render_page_shell, render_stage_header)
from radiofry.explore.time_machine import (
    DEFAULT_WINDOW_SECONDS,
    MIN_ANALYSIS_SAMPLES,
    advance_cursor,
    analysis_for_region,
    calculate_marker_delta,
    capture_overview,
    compare_reports,
    compare_two_regions,
    detect_candidate_bursts,
    downsample_for_display,
    extract_window,
    format_frequency_axis_unit,
    format_smart_frequency,
    format_smart_time,
    format_time_axis_unit,
    format_timestamp,
    investigation_name,
    region_container,
    region_from_drag,
    select_region,
    select_window,
    symbol_rate_decimated,
    total_duration_seconds,
    window_measurements,
    window_spectrogram,
    window_spectrum,
)

UNAVAILABLE = "Unavailable for selected window"
WINDOW_OPTIONS_MS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0]


@st.cache_data(show_spinner=False)
def _cached_overview(iq: np.ndarray, sample_rate: float):
    """Whole-capture spectrogram used as the region-selection surface.

    Cached on the samples themselves, so scrubbing the timeline, editing the region
    bounds or running an analysis never recomputes it.
    """
    return capture_overview(iq, sample_rate)


render_page_shell(9)
signal = st.session_state.get("signal")
report = st.session_state.get("report")
render_stage_header(
    "09",
    "Signal Time Machine",
    "Move through the recording itself. Every view below shows the same selected window "
    "of the same capture.",
    "Ready" if signal is not None else "Waiting",
    "ready" if signal is not None else "muted",
)

if signal is None:
    render_empty_state(
        "Analyze a WAV or IQ capture on the home page first.",
        "The Time Machine explores the capture you already analyzed - it does not load "
        "a separate file.")
    render_method_note(
        "What this stage is",
        "An exploration view over the real samples. It does not re-run the analysis "
        "pipeline and does not produce new modulation decisions.")
    st.stop()

samples_total = int(signal.iq.size)
sample_rate = signal.sample_rate
duration = total_duration_seconds(samples_total, sample_rate)

# ---------------------------------------------------------------- timeline controller
st.markdown("<div class='evidence-label'>Timeline</div>", unsafe_allow_html=True)

if st.session_state.get("stm_capture_key") != (samples_total, sample_rate):
    # A different capture was analyzed. Reset before the widgets are created, so the
    # stored cursor can never point past the new recording's end.
    st.session_state["stm_capture_key"] = (samples_total, sample_rate)
    st.session_state["stm_cursor"] = 0.0
    st.session_state["stm_playing"] = False
    st.session_state["stm_frozen"] = False
    st.session_state.pop("stm_drag_applied", None)
    st.session_state.pop("stm_drag_reason", None)
    # A lock names a region of the recording that is no longer loaded, so it cannot
    # survive a capture change without pointing at the wrong samples.
    st.session_state["stm_locked"] = False
    st.session_state["stm_region_name"] = ""
st.session_state.setdefault("stm_cursor", 0.0)
st.session_state.setdefault("stm_locked", False)
st.session_state.setdefault("stm_region_name", "")
st.session_state.setdefault("stm_region_notes", "")
st.session_state.setdefault("stm_playing", False)
st.session_state.setdefault("stm_frozen", False)
st.session_state.setdefault("stm_speed", 1.0)
st.session_state.setdefault("stm_window_ms", DEFAULT_WINDOW_SECONDS * 1_000)
st.session_state.setdefault("stm_investigations", [])
st.session_state.setdefault("stm_marker_a_time", None)
st.session_state.setdefault("stm_marker_a_freq", None)
st.session_state.setdefault("stm_marker_a_power", None)
st.session_state.setdefault("stm_marker_b_time", None)
st.session_state.setdefault("stm_marker_b_freq", None)
st.session_state.setdefault("stm_marker_b_power", None)
st.session_state.setdefault("stm_spectral_cursor_freq", 0.0)
if duration is not None:
    st.session_state.setdefault("stm_region_start", 0.0)
    st.session_state.setdefault("stm_region_end", round(min(duration, 0.05), 6))

    # A drag on the overview arrives as that chart's selection event. It is turned into
    # a region HERE, before the region number_inputs are created, because a keyed
    # widget's session_state entry cannot be written once the widget exists - the same
    # rule behind the earlier freeze bug. The box is fingerprinted so a selection that
    # Streamlit keeps replaying is not re-applied on every rerun, which would otherwise
    # overwrite whatever the user then typed into the numeric controls.
    drag = (st.session_state.get("stm_overview_chart") or {}).get("selection") or {}
    boxes = drag.get("box") or []
    if not boxes:
        # Selection cleared in the chart: forget the fingerprint so an identical drag
        # can be made again later.
        st.session_state.pop("stm_drag_applied", None)
    if boxes:
        span = boxes[0].get("x") or []
        if len(span) >= 2:
            fingerprint = (round(float(min(span)), 9), round(float(max(span)), 9))
            if st.session_state.get("stm_drag_applied") != fingerprint:
                # Mark the gesture seen either way. A drag refused because of the lock
                # must not spring back and overwrite the region the moment it unlocks.
                st.session_state["stm_drag_applied"] = fingerprint
                if st.session_state["stm_locked"]:
                    # A locked region is the analyst's investigation target; it is not
                    # replaced by a stray gesture.
                    st.session_state["stm_drag_reason"] = (
                        "the signal is locked. Unlock it to select a different region.")
                else:
                    dragged = region_from_drag(samples_total, sample_rate,
                                               span[0], span[-1])
                    st.session_state["stm_drag_reason"] = (
                        "" if dragged.ok else dragged.reason)
                    if dragged.ok:
                        picked = dragged.selection
                        st.session_state["stm_pending_region"] = (
                            picked.start_seconds, picked.end_seconds)

    # Single choke point for every programmatic region move, so "a locked region does
    # not change" is one condition rather than a rule each caller has to remember.
    pending_region = st.session_state.pop("stm_pending_region", None)
    if pending_region is not None and not st.session_state["stm_locked"]:
        st.session_state["stm_region_start"] = float(
            min(max(0.0, pending_region[0]), duration))
        st.session_state["stm_region_end"] = float(
            min(max(0.0, pending_region[1]), duration))

# A keyed widget's session_state entry can only be written BEFORE that widget is
# created, so every programmatic cursor move happens here: the playback tick and any
# pending "jump to timestamp". Doing it after the slider raises
# StreamlitWidgetAlreadyInstantiatedError.
if duration is not None:
    # Focus resizes the detail window, which is another keyed widget, so it uses the
    # same pending handoff rather than writing the select_slider's key later. The
    # smallest offered window that still covers the region wins; a region longer than
    # any window gets the widest one, and is simply not shown whole.
    pending_window = st.session_state.pop("stm_pending_window_ms", None)
    if pending_window is not None:
        covering = [ms for ms in WINDOW_OPTIONS_MS if ms >= pending_window]
        st.session_state["stm_window_ms"] = (min(covering) if covering
                                             else max(WINDOW_OPTIONS_MS))

    pending = st.session_state.pop("stm_pending_cursor", None)
    if pending is not None:
        st.session_state["stm_cursor"] = float(min(max(0.0, pending), duration))
    elif st.session_state["stm_playing"] and not st.session_state["stm_frozen"]:
        step = max(st.session_state["stm_window_ms"] / 1_000.0, duration / 200.0)
        advanced = advance_cursor(st.session_state["stm_cursor"], duration,
                                  step_seconds=step,
                                  speed=st.session_state["stm_speed"], loop=False)
        st.session_state["stm_cursor"] = advanced
        if advanced >= duration:
            # Playback runs to the end and stops. Looping forever would leave the
            # browser in a permanent rerun cycle with nothing marking the end.
            st.session_state["stm_playing"] = False

if duration is None:
    st.warning(
        "This capture has no sample rate, so it has no time axis. The whole capture is "
        "shown as a single window. Re-analyze a raw IQ file with its sample rate to "
        "enable the timeline.")
    cursor = 0.0
    window_seconds = 0.0
    speed = 1.0
else:
    # Every timeline widget owns its own session_state key. Passing `value=` while also
    # mirroring the result into a separate key makes the widget and the mirror fight,
    # which is how an earlier revision produced a freeze that could not be released.
    control_row = st.columns([1, 1, 1, 2, 2])
    with control_row[0]:
        if st.button("Play", use_container_width=True,
                     disabled=st.session_state["stm_frozen"]):
            st.session_state["stm_playing"] = True
    with control_row[1]:
        if st.button("Pause", use_container_width=True):
            st.session_state["stm_playing"] = False
    with control_row[2]:
        frozen = st.toggle("Freeze", key="stm_frozen",
                           help="Hold the current window still for inspection.")
        if frozen:
            st.session_state["stm_playing"] = False
    with control_row[3]:
        speed = st.select_slider("Playback speed", options=[0.25, 0.5, 1.0, 2.0, 4.0],
                                 key="stm_speed")
    with control_row[4]:
        window_ms = st.select_slider(
            "Analysis window (ms)",
            options=WINDOW_OPTIONS_MS,
            key="stm_window_ms",
            help="Zoom: a shorter window shows finer detail, a longer one more context.")
        window_seconds = window_ms / 1_000.0

    st.slider(
        "Position (s)", min_value=0.0, max_value=float(duration), key="stm_cursor",
        step=max(float(duration) / 1_000.0, 1e-6),
        disabled=st.session_state["stm_frozen"],
        help="Scrub through the recording. Every view below follows this position.")
    cursor = float(st.session_state["stm_cursor"])

    jump_left, jump_right = st.columns([3, 1])
    with jump_left:
        target = st.number_input("Jump to timestamp (s)", min_value=0.0,
                                 max_value=float(duration), value=cursor,
                                 step=0.001, format="%.4f",
                                 disabled=st.session_state["stm_frozen"])
    with jump_right:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("Jump", use_container_width=True,
                     disabled=st.session_state["stm_frozen"]):
            st.session_state["stm_pending_cursor"] = float(target)
            st.rerun()

# ---------------------------------------------------------------- the single selection
selection = select_window(samples_total, sample_rate, cursor, window_seconds
                          if duration is not None else 0.0)
window = extract_window(signal.iq, selection)

state_bits = []
if st.session_state.get("stm_frozen"):
    state_bits.append("**FROZEN** - the window below is held still")
elif st.session_state.get("stm_playing"):
    state_bits.append(f"**Playing** at {speed}x")
if state_bits:
    st.info(" · ".join(state_bits))

position_row = st.columns(4)
position_row[0].metric(
    "Cursor", f"{selection.start_seconds:.4f} s"
    if selection.start_seconds is not None else "n/a")
position_row[1].metric(
    "Window", f"{selection.duration_seconds * 1000:.2f} ms"
    if selection.duration_seconds is not None else "whole capture")
position_row[2].metric("Window samples", f"{selection.length_samples:,}")
position_row[3].metric(
    "Total duration", f"{duration:.4f} s" if duration is not None else "Unknown")
st.caption(
    f"Showing samples {selection.start_sample:,} to {selection.end_sample:,} "
    f"of {samples_total:,} - every view below is this exact region.")

if window.size == 0:
    st.error("The selected window contains no samples.")
    st.stop()

# Resolved again here purely so the waveform can shade the investigation region; the
# authoritative validation happens in the region section further down.
_preview_region = (select_region(samples_total, sample_rate,
                                 float(st.session_state.get("stm_region_start", 0.0)),
                                 float(st.session_state.get("stm_region_end", 0.0)))
                   if duration is not None else None)
region_start_sample = _preview_region.selection.start_sample \
    if _preview_region and _preview_region.ok else -1
region_end_sample = _preview_region.selection.end_sample \
    if _preview_region and _preview_region.ok else -1

# A lock can only ever hold a region that validates, so a capture edited out from under
# one releases it rather than leaving a target pointing at nothing.
if st.session_state["stm_locked"] and region_start_sample < 0:
    st.session_state["stm_locked"] = False
locked = bool(st.session_state["stm_locked"])

if locked:
    # Shown here, above every view, so the investigation target stays on screen while
    # the analyst explores the rest of the recording. The colour is the same yellow the
    # region band uses on the charts below, so the banner and the highlight read as one
    # thing.
    held = _preview_region.selection
    # Only an analysis that ran on exactly these samples may be reported here. Until one
    # has, the target has no modulation, no frequency and no identity - and says so,
    # rather than borrowing the whole-capture answer.
    held_analysis = analysis_for_region(
        st.session_state["stm_investigations"], held.start_sample, held.end_sample)
    # A name is a note to the analyst. It is resolved for display only - matching is
    # still done on sample bounds, so renaming cannot change which analysis is found.
    # What the analyst typed wins; an already-analyzed region otherwise keeps the name
    # its history entry holds, so clearing the box does not renumber it.
    held_name = (st.session_state["stm_region_name"].strip()
                 or (held_analysis or {}).get("name")
                 or investigation_name(None,
                                       len(st.session_state["stm_investigations"]) + 1))
    if held_analysis is None:
        held_status = ("<span style='color:var(--muted)'>Not analyzed yet</span> "
                       "&middot; press Analyze Selection to measure this region")
    else:
        held_fusion = (held_analysis["report"].get("stages", {}) or {}).get(
            "fusion", {}) or {}
        held_label = held_fusion.get("label") or "Unclassified"
        held_trust = held_fusion.get("trust_score")
        held_status = f"Analyzed &middot; <b>{held_label}</b>"
        if isinstance(held_trust, (int, float)):
            held_status += f" (trust {held_trust:.3f})"
    st.markdown(
        "<div class='evidence-panel' style='border-color:#f2c14e;"
        "border-left-width:4px;background:rgba(242,193,78,0.07);margin-bottom:0.75rem'>"
        "<div class='evidence-label' style='color:#f2c14e'>&#128274; Signal locked "
        "&middot; active investigation target</div>"
        f"<div style='font-size:1.35rem;font-weight:700;margin:0.35rem 0 0.1rem'>"
        f"{html.escape(held_name)}</div>"
        f"<div style='font-size:1.05rem;font-weight:600;margin-bottom:0.2rem'>"
        f"{format_timestamp(held.start_seconds)} &rarr; "
        f"{format_timestamp(held.end_seconds)}</div>"
        f"<div style='color:var(--muted)'>Duration {held.duration_seconds:.3f} s "
        f"&middot; samples {held.start_sample:,}&ndash;{held.end_sample:,} "
        f"of {samples_total:,}</div>"
        f"<div style='margin-top:0.45rem'>{held_status}</div></div>",
        unsafe_allow_html=True)
    st.caption(
        "This region stays selected while you scrub, play and zoom. The playback "
        "window above moves independently - unlock to choose a different region.")

# ---------------------------------------------------------------- selection surface
st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "Select a region</div>", unsafe_allow_html=True)

if duration is None:
    st.info("Region selection needs a sample rate to place a time axis.")
else:
    st.caption(
        "The signal is locked, so dragging will not change the region. Unlock it below "
        "to select a different one - scrubbing and playback still work as normal."
        if locked else
        "Drag horizontally across the overview to choose the investigation region, then "
        "press Analyze Selection further down. Dragging only moves the selection - it "
        "never runs the pipeline. Double-click the chart to clear the drag.")
    overview_times, overview_frequencies, overview_power = _cached_overview(
        signal.iq, float(sample_rate))
    if overview_power.size == 0:
        st.info("This capture is too short to build an overview. Use the numeric region "
                "controls below instead.")
    else:
        overview_figure = go.Figure(go.Heatmap(
            x=overview_times, y=overview_frequencies / 1_000.0, z=overview_power.T,
            colorscale="Viridis", showscale=False,
            hovertemplate="t=%{x:.4f} s<br>%{y:.1f} kHz<br>%{z:.1f} dB<extra></extra>"))
        if region_start_sample >= 0:
            # Drawn from the validated region bounds, so the band always marks the exact
            # samples that Analyze Selection would run on.
            overview_figure.add_vrect(
                x0=region_start_sample / sample_rate,
                x1=region_end_sample / sample_rate,
                fillcolor="#f2c14e", opacity=0.38 if locked else 0.30,
                line_width=3 if locked else 2, line_color="#f2c14e", layer="above",
                annotation_text="\U0001F512 LOCKED" if locked else None,
                annotation_position="top left",
                annotation_font={"color": "#f2c14e", "size": 12})
        if selection.start_seconds is not None:
            # The playback window is a different idea from the investigation region, so
            # it gets a different mark rather than a second band.
            overview_figure.add_vline(x=selection.start_seconds, line_width=1.5,
                                      line_dash="dot", line_color="#7fe7c4")
        overview_figure.update_layout(
            height=260, margin={"l": 0, "r": 0, "t": 10, "b": 0},
            dragmode="select", selectdirection="h",
            xaxis_title="Time (s)", yaxis_title="Offset (kHz)")
        st.plotly_chart(overview_figure, use_container_width=True,
                        key="stm_overview_chart", on_select="rerun",
                        selection_mode="box")
        legend = ("Yellow band = \U0001F512 locked investigation region." if locked
                  else "Yellow band = investigation region.")
        if selection.start_seconds is not None:
            legend += "  Dotted line = current playback window."
        st.caption(
            legend + "  This overview is a spectrogram of the whole recording; the "
            "detail views below still follow the playback window.")
    if st.session_state.get("stm_drag_reason"):
        st.warning(f"That drag could not be used: "
                   f"{st.session_state['stm_drag_reason']}")

# ---------------------------------------------------------------- synchronized views
waveform_column, spectrum_column = st.columns(2)

with waveform_column:
    st.markdown("<div class='evidence-panel'><h3>Waveform</h3>"
                "<p style='color:var(--muted)'>Real I/Q samples from the selected "
                "window.</p>", unsafe_allow_html=True)
    indices, displayed = downsample_for_display(window)
    # Plotted against absolute capture time where there is one, so the region band below
    # lands in the same place it does on the overview.
    if sample_rate:
        times_axis = (selection.start_seconds or 0.0) + indices / sample_rate
    else:
        times_axis = indices
    waveform_figure = go.Figure()
    waveform_figure.add_scatter(x=times_axis, y=displayed.real, mode="lines", name="I",
                                line={"width": 1.2, "color": "#38bdf8"},
                                hovertemplate="Time: %{x:.5f} s<br>I: %{y:.4f}<extra></extra>")
    waveform_figure.add_scatter(x=times_axis, y=displayed.imag, mode="lines", name="Q",
                                line={"width": 1.2, "color": "#f2c14e"},
                                hovertemplate="Time: %{x:.5f} s<br>Q: %{y:.4f}<extra></extra>")
    if region_start_sample >= 0 and sample_rate:
        band_start = max(region_start_sample / sample_rate, float(times_axis[0]))
        band_end = min(region_end_sample / sample_rate, float(times_axis[-1]))
        if band_end > band_start:
            waveform_figure.add_vrect(x0=band_start, x1=band_end, fillcolor="#f2c14e",
                                      opacity=0.20, line_width=0, layer="below")
    # Marker lines if active and within visible window
    m_a_t = st.session_state.get("stm_marker_a_time")
    m_b_t = st.session_state.get("stm_marker_b_time")
    if m_a_t is not None and len(times_axis) and times_axis[0] <= m_a_t <= times_axis[-1]:
        waveform_figure.add_vline(x=m_a_t, line_width=1.5, line_dash="dash", line_color="#06b6d4",
                                  annotation_text="A", annotation_position="top left",
                                  annotation_font={"color": "#06b6d4", "size": 11})
    if m_b_t is not None and len(times_axis) and times_axis[0] <= m_b_t <= times_axis[-1]:
        waveform_figure.add_vline(x=m_b_t, line_width=1.5, line_dash="dash", line_color="#ec4899",
                                  annotation_text="B", annotation_position="top right",
                                  annotation_font={"color": "#ec4899", "size": 11})

    waveform_figure.update_layout(
        height=260, margin={"l": 35, "r": 10, "t": 24, "b": 35},
        xaxis_title="Time (s)" if sample_rate else "Sample index",
        yaxis_title="Amplitude",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        xaxis={"gridcolor": "rgba(51, 65, 85, 0.4)", "zerolinecolor": "rgba(100, 116, 139, 0.5)"},
        yaxis={"gridcolor": "rgba(51, 65, 85, 0.4)", "zerolinecolor": "rgba(100, 116, 139, 0.5)"},
        legend={"orientation": "h", "y": 1.18, "font": {"size": 10}})
    st.plotly_chart(waveform_figure, use_container_width=True, key="stm_waveform_chart")
    region_overlap = None
    if duration is not None:
        overlap_start = max(selection.start_sample, region_start_sample)
        overlap_end = min(selection.end_sample, region_end_sample)
        if overlap_end > overlap_start:
            region_overlap = (overlap_start, overlap_end)
    if region_overlap:
        st.caption(
            f"Shaded: investigation region, samples {region_overlap[0]:,}-"
            f"{region_overlap[1]:,} of this window.")
    elif duration is not None:
        st.caption("The investigation region lies outside this window, so nothing is "
                   "shaded here.")
    if displayed.size < window.size:
        st.caption(f"Decimated to {displayed.size:,} of {window.size:,} samples for "
                   "display; values are untouched samples, not interpolated.")
    st.markdown("</div>", unsafe_allow_html=True)

with spectrum_column:
    st.markdown("<div class='evidence-panel'><h3>Spectrum</h3>"
                "<p style='color:var(--muted)'>FFT of this window, normalised to its "
                "own peak.</p>", unsafe_allow_html=True)
    frequencies, magnitude_db = window_spectrum(window, sample_rate)
    if frequencies.size == 0:
        st.info(UNAVAILABLE + " (needs a sample rate and at least 4 samples).")
    else:
        freq_mult, freq_unit = format_frequency_axis_unit(frequencies)
        freq_scaled = frequencies * freq_mult
        spectrum_figure = go.Figure()
        spectrum_figure.add_scatter(
            x=freq_scaled, y=magnitude_db, mode="lines",
            line={"width": 1.3, "color": "#38bdf8"},
            hovertemplate=f"Offset: %{{x:.2f}} {freq_unit}<br>Power: %{{y:.1f}} dB<extra></extra>",
            name="Spectrum"
        )
        peak_idx = int(np.argmax(magnitude_db))
        peak_freq_scaled = float(freq_scaled[peak_idx])
        peak_pwr = float(magnitude_db[peak_idx])
        spectrum_figure.add_annotation(
            x=peak_freq_scaled, y=peak_pwr,
            text=f"Peak: {peak_freq_scaled:.2f} {freq_unit} ({peak_pwr:.1f} dB)",
            showarrow=True, arrowhead=2, arrowcolor="#f2c14e",
            font={"color": "#f2c14e", "size": 10},
            bgcolor="rgba(15,23,42,0.85)", bordercolor="#f2c14e", borderwidth=1,
            yshift=10
        )
        # Spectral probe cursor line
        probe_f = st.session_state.get("stm_spectral_cursor_freq")
        if probe_f is not None and frequencies[0] <= probe_f <= frequencies[-1]:
            spectrum_figure.add_vline(
                x=probe_f * freq_mult, line_width=1.5, line_dash="dot", line_color="#a855f7",
                annotation_text="Probe", annotation_position="top left",
                annotation_font={"color": "#a855f7", "size": 10}
            )
        # Markers on spectrum
        m_a_f = st.session_state.get("stm_marker_a_freq")
        m_b_f = st.session_state.get("stm_marker_b_freq")
        if m_a_f is not None and frequencies[0] <= m_a_f <= frequencies[-1]:
            spectrum_figure.add_vline(
                x=m_a_f * freq_mult, line_width=1.5, line_dash="dash", line_color="#06b6d4",
                annotation_text="A", annotation_position="bottom left",
                annotation_font={"color": "#06b6d4", "size": 10}
            )
        if m_b_f is not None and frequencies[0] <= m_b_f <= frequencies[-1]:
            spectrum_figure.add_vline(
                x=m_b_f * freq_mult, line_width=1.5, line_dash="dash", line_color="#ec4899",
                annotation_text="B", annotation_position="bottom right",
                annotation_font={"color": "#ec4899", "size": 10}
            )
        spectrum_figure.update_layout(
            height=260, margin={"l": 40, "r": 10, "t": 24, "b": 35},
            xaxis_title=f"Offset from centre ({freq_unit})",
            yaxis_title="Relative power (dB)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis={"gridcolor": "rgba(51, 65, 85, 0.4)", "zerolinecolor": "rgba(100, 116, 139, 0.5)"},
            yaxis={"gridcolor": "rgba(51, 65, 85, 0.4)", "zerolinecolor": "rgba(100, 116, 139, 0.5)", "range": [-85, 5]},
            showlegend=False
        )
        st.plotly_chart(spectrum_figure, use_container_width=True, key="stm_spectrum_chart")
    st.markdown("</div>", unsafe_allow_html=True)

waterfall_column, scatter_column = st.columns(2)

with waterfall_column:
    st.markdown("<div class='evidence-panel'><h3>Waterfall</h3>"
                "<p style='color:var(--muted)'>Energy across time and frequency inside "
                "this window.</p>", unsafe_allow_html=True)
    times, spectrogram_frequencies, power_db = window_spectrogram(window, sample_rate)
    if power_db.size == 0:
        st.info(UNAVAILABLE + " (window too short for a spectrogram block).")
    else:
        offset = selection.start_seconds or 0.0
        time_axis = offset + times
        freq_mult_wf, freq_unit_wf = format_frequency_axis_unit(spectrogram_frequencies)
        freq_scaled_wf = spectrogram_frequencies * freq_mult_wf
        clamped_power = np.clip(power_db.T, -80.0, 0.0)
        waterfall_figure = go.Figure(go.Heatmap(
            x=time_axis,
            y=freq_scaled_wf,
            z=clamped_power,
            colorscale="Viridis",
            zmin=-80.0,
            zmax=0.0,
            colorbar={"title": "dB", "thickness": 10, "len": 0.85, "tickfont": {"size": 9, "color": "#94a3b8"}},
            hovertemplate=f"Time: %{{x:.5f}} s<br>Offset: %{{y:.2f}} {freq_unit_wf}<br>Power: %{{z:.1f}} dB<extra></extra>"
        ))
        waterfall_figure.update_layout(
            height=260, margin={"l": 40, "r": 10, "t": 24, "b": 35},
            xaxis_title="Time (s, absolute in capture)",
            yaxis_title=f"Offset ({freq_unit_wf})",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            xaxis={"gridcolor": "rgba(51, 65, 85, 0.4)"},
            yaxis={"gridcolor": "rgba(51, 65, 85, 0.4)"}
        )
        st.plotly_chart(waterfall_figure, use_container_width=True, key="stm_waterfall_chart")
    st.markdown("</div>", unsafe_allow_html=True)

parameters = (report or {}).get("stages", {}).get("parameters", {}) or {}
symbol_rate = parameters.get("symbol_rate_hz")

with scatter_column:
    st.markdown("<div class='evidence-panel'><h3>IQ Scatter</h3>"
                "<p style='color:var(--muted)'>Raw I against Q for this window.</p>",
                unsafe_allow_html=True)
    mode = st.radio(
        "Points", ["All samples", "Symbol-rate decimated"], horizontal=True,
        label_visibility="collapsed",
        help="Symbol-rate decimation uses the whole-capture symbol rate; timing phase "
             "is NOT re-estimated for this window.")
    points = window
    if mode == "Symbol-rate decimated":
        points = symbol_rate_decimated(window, sample_rate, symbol_rate)
        if points.size == 0:
            st.info(UNAVAILABLE + " - no usable symbol rate was estimated for this "
                    "capture, so the samples cannot be decimated per symbol.")
    if points.size:
        _, shown = downsample_for_display(points, max_points=4_000)
        scatter_figure = go.Figure()
        scatter_figure.add_scattergl(
            x=shown.real, y=shown.imag, mode="markers",
            marker={"size": 3.5, "color": "#38bdf8", "opacity": 0.55},
            hovertemplate="I: %{x:.4f}<br>Q: %{y:.4f}<extra></extra>",
            name="IQ"
        )
        scatter_figure.update_layout(
            height=260, margin={"l": 35, "r": 10, "t": 24, "b": 35},
            xaxis_title="I",
            yaxis_title="Q",
            yaxis={"scaleanchor": "x", "scaleratio": 1, "gridcolor": "rgba(51, 65, 85, 0.4)", "zerolinecolor": "rgba(100, 116, 139, 0.5)"},
            xaxis={"gridcolor": "rgba(51, 65, 85, 0.4)", "zerolinecolor": "rgba(100, 116, 139, 0.5)"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            showlegend=False
        )
        st.plotly_chart(scatter_figure, use_container_width=True, key="stm_scatter_chart")
        st.caption(
            "Raw I/Q scatter, not a decoded constellation: symbol timing has not been "
            "re-established for this window."
            if mode == "All samples" else
            f"Decimated at the reported {symbol_rate:,.1f} Hz symbol rate. Timing phase "
            "is not re-estimated here, so the cloud may be smeared even when the rate "
            "is correct.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- RF measurement tools
if duration is not None and frequencies.size > 0:
    st.markdown("<div class='evidence-label' style='margin-top:1.5rem;'>"
                "RF Measurement Tools</div>", unsafe_allow_html=True)

    tab_markers, tab_spectral, tab_bursts, tab_compare = st.tabs([
        "Markers & Deltas",
        "Spectral Probe Cursor",
        "Candidate Burst Detector",
        "A/B Region Comparison"
    ])

    with tab_markers:
        st.caption("Place Marker A and Marker B to calculate precise Δ Time, Δ Frequency, "
                   "Δ Power, and the corresponding Pulse Repetition / Symbol rate.")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("<span style='color:#06b6d4;font-weight:600'>Marker A</span>",
                        unsafe_allow_html=True)
            btn_a_col1, btn_a_col2 = st.columns(2)
            with btn_a_col1:
                if st.button("Set A to Window Peak", use_container_width=True):
                    st.session_state["stm_marker_a_time"] = float(cursor)
                    st.session_state["stm_marker_a_freq"] = float(frequencies[peak_idx])
                    st.session_state["stm_marker_a_power"] = float(magnitude_db[peak_idx])
                    st.rerun()
            with btn_a_col2:
                if st.button("Clear Marker A", use_container_width=True):
                    st.session_state["stm_marker_a_time"] = None
                    st.session_state["stm_marker_a_freq"] = None
                    st.session_state["stm_marker_a_power"] = None
                    st.rerun()
            ma_t = st.session_state.get("stm_marker_a_time")
            ma_f = st.session_state.get("stm_marker_a_freq")
            ma_p = st.session_state.get("stm_marker_a_power")
            st.markdown(f"**Time:** {format_smart_time(ma_t)} &middot; "
                        f"**Freq:** {format_smart_frequency(ma_f)} &middot; "
                        f"**Power:** {f'{ma_p:.1f} dB' if ma_p is not None else 'Unset'}")

        with m_col2:
            st.markdown("<span style='color:#ec4899;font-weight:600'>Marker B</span>",
                        unsafe_allow_html=True)
            btn_b_col1, btn_b_col2 = st.columns(2)
            with btn_b_col1:
                if st.button("Set B to Window Peak", use_container_width=True):
                    st.session_state["stm_marker_b_time"] = float(cursor)
                    st.session_state["stm_marker_b_freq"] = float(frequencies[peak_idx])
                    st.session_state["stm_marker_b_power"] = float(magnitude_db[peak_idx])
                    st.rerun()
            with btn_b_col2:
                if st.button("Clear Marker B", use_container_width=True):
                    st.session_state["stm_marker_b_time"] = None
                    st.session_state["stm_marker_b_freq"] = None
                    st.session_state["stm_marker_b_power"] = None
                    st.rerun()
            mb_t = st.session_state.get("stm_marker_b_time")
            mb_f = st.session_state.get("stm_marker_b_freq")
            mb_p = st.session_state.get("stm_marker_b_power")
            st.markdown(f"**Time:** {format_smart_time(mb_t)} &middot; "
                        f"**Freq:** {format_smart_frequency(mb_f)} &middot; "
                        f"**Power:** {f'{mb_p:.1f} dB' if mb_p is not None else 'Unset'}")

        # Compute and render marker deltas
        deltas = calculate_marker_delta(
            time_a=ma_t, time_b=mb_t, freq_a=ma_f, freq_b=mb_f, power_a=ma_p, power_b=mb_p
        )
        delta_cols = st.columns(4)
        delta_cols[0].metric(
            "\u0394 Time",
            format_smart_time(deltas.get("delta_time_s")) if deltas.get("delta_time_s") is not None else "Unset"
        )
        rate_val = deltas.get("time_frequency_hz") or deltas.get("pri_freq_hz")
        delta_cols[1].metric(
            "1 / \u0394 Time (Rate)",
            format_smart_frequency(rate_val) if rate_val is not None else "Unset"
        )
        delta_cols[2].metric(
            "\u0394 Frequency",
            format_smart_frequency(deltas.get("delta_freq_hz")) if deltas.get("delta_freq_hz") is not None else "Unset"
        )
        dp = deltas.get("delta_power_db")
        delta_cols[3].metric(
            "\u0394 Power",
            f"{dp:+.1f} dB" if dp is not None else "Unset"
        )

    with tab_spectral:
        st.caption("Inspect power at any frequency bin inside the current window spectrum.")
        f_min = float(frequencies[0])
        f_max = float(frequencies[-1])
        curr_probe = float(st.session_state.get("stm_spectral_cursor_freq", 0.0))
        curr_probe = float(np.clip(curr_probe, f_min, f_max))
        probe_input_col, probe_btn_col = st.columns([3, 2])
        with probe_input_col:
            probe_freq = st.slider(
                "Probe Frequency Offset (Hz)",
                min_value=round(f_min, 1),
                max_value=round(f_max, 1),
                value=round(curr_probe, 1),
                step=max(1.0, round((f_max - f_min) / 500.0, 1)),
                key="stm_probe_slider"
            )
            st.session_state["stm_spectral_cursor_freq"] = probe_freq
        closest_idx = int(np.argmin(np.abs(frequencies - probe_freq)))
        probed_f_val = float(frequencies[closest_idx])
        probed_pwr = float(magnitude_db[closest_idx])
        rel_to_peak = probed_pwr - peak_pwr

        with probe_btn_col:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            p_b1, p_b2 = st.columns(2)
            with p_b1:
                if st.button("Probe \u2192 Marker A", use_container_width=True):
                    st.session_state["stm_marker_a_time"] = float(cursor)
                    st.session_state["stm_marker_a_freq"] = probed_f_val
                    st.session_state["stm_marker_a_power"] = probed_pwr
                    st.rerun()
            with p_b2:
                if st.button("Probe \u2192 Marker B", use_container_width=True):
                    st.session_state["stm_marker_b_time"] = float(cursor)
                    st.session_state["stm_marker_b_freq"] = probed_f_val
                    st.session_state["stm_marker_b_power"] = probed_pwr
                    st.rerun()

        p_row = st.columns(4)
        p_row[0].metric("Probed Frequency", format_smart_frequency(probed_f_val))
        p_row[1].metric("Relative Power", f"{probed_pwr:.1f} dB")
        p_row[2].metric("vs Spectral Peak", f"{rel_to_peak:+.1f} dB")
        window_mid = (selection.start_seconds or 0.0) + (selection.duration_seconds or 0.0) / 2.0
        p_row[3].metric("Window Center Time", format_smart_time(window_mid))

    with tab_bursts:
        st.caption("Conservative energy-based detector to locate candidate transmission bursts "
                   "above local noise floor.")
        b_c1, b_c2 = st.columns([3, 1])
        with b_c1:
            threshold_db = st.slider(
                "Energy Threshold Above Noise (dB)",
                min_value=3.0, max_value=24.0, value=6.0, step=0.5,
                help="Higher threshold finds only prominent bursts; lower finds weaker activity."
            )
        with b_c2:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            run_burst_scan = st.button("Scan Recording", use_container_width=True)

        if run_burst_scan or "stm_detected_bursts" in st.session_state:
            if run_burst_scan:
                with st.spinner("Scanning for candidate active regions..."):
                    detected = detect_candidate_bursts(
                        signal.iq, sample_rate, threshold_db_above_noise=threshold_db
                    )
                    st.session_state["stm_detected_bursts"] = detected
            bursts = st.session_state.get("stm_detected_bursts", [])
            if not bursts:
                st.info("No candidate active regions found above the selected threshold.")
            else:
                st.markdown(f"**Found {len(bursts)} candidate active region{'s' if len(bursts) > 1 else ''}:**")
                for i, b in enumerate(bursts[:8]):
                    b_row = st.columns([3, 1, 1])
                    with b_row[0]:
                        st.markdown(
                            f"**Candidate Burst #{i + 1}**: {format_smart_time(b['start_s'])} &rarr; "
                            f"{format_smart_time(b['end_s'])} "
                            f"(Duration: {format_smart_time(b['duration_s'])}, Peak: {b['peak_snr_db']:.1f} dB SNR)"
                        )
                    with b_row[1]:
                        if st.button(f"Inspect #{i + 1}", key=f"btn_inspect_burst_{i}", use_container_width=True):
                            st.session_state["stm_pending_region"] = (b["start_s"], b["end_s"])
                            st.session_state["stm_region_name"] = f"Candidate Burst #{i + 1}"
                            st.session_state["stm_pending_cursor"] = b["start_s"]
                            st.rerun()
                    with b_row[2]:
                        if st.button(f"Focus #{i + 1}", key=f"btn_focus_burst_{i}", use_container_width=True):
                            st.session_state["stm_pending_cursor"] = b["start_s"]
                            st.session_state["stm_pending_window_ms"] = b["duration_s"] * 1_000.0
                            st.rerun()
                st.caption("Active regions detected via time-domain energy envelope. "
                           "Click 'Inspect' to select as the investigation target.")

    with tab_compare:
        st.caption("Compare RF measurements side-by-side between two analyzed regions.")
        investigations_list = st.session_state.get("stm_investigations", [])
        if len(investigations_list) < 2:
            st.info("At least two regions must be analyzed via 'Analyze Selection' to compare them.")
        else:
            inv_labels = [
                f"#{idx + 1} {investigation_name(item.get('name'), idx + 1)} "
                f"({format_smart_time(item['start_s'])}-{format_smart_time(item['end_s'])})"
                for idx, item in enumerate(investigations_list)
            ]
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                idx_a = st.selectbox("Region A", list(range(len(investigations_list))),
                                     index=0, format_func=lambda i: inv_labels[i], key="stm_cmp_a")
            with cmp_col2:
                idx_b = st.selectbox("Region B", list(range(len(investigations_list))),
                                     index=min(1, len(investigations_list) - 1),
                                     format_func=lambda i: inv_labels[i], key="stm_cmp_b")
            cmp_rows = compare_two_regions(investigations_list[idx_a], investigations_list[idx_b])
            if cmp_rows:
                st.table({
                    "Metric": [r[0] for r in cmp_rows],
                    "Region A": [r[1] for r in cmp_rows],
                    "Region B": [r[2] for r in cmp_rows],
                    "Difference": [r[3] for r in cmp_rows],
                })


# ---------------------------------------------------------------- region investigation
st.markdown("<div class='evidence-label' style='margin-top:1.5rem;'>"
            "Region investigation</div>", unsafe_allow_html=True)

if duration is None:
    st.info(
        "Region analysis needs a sample rate to locate a time range. Re-analyze this "
        "capture with its sample rate to enable it.")
else:
    st.markdown(
        "<div class='evidence-panel'><p style='color:var(--muted);margin:0'>Choose a "
        "span of the recording - by dragging on the overview above or by typing bounds "
        "here, whichever is easier - and run the <b>existing</b> RadioFry pipeline on "
        "exactly those samples. The two controls are the same selection: dragging "
        "updates these numbers and editing these numbers moves the highlight. Neither "
        "triggers analysis - only the button below does.</p></div>",
        unsafe_allow_html=True)

    bound_left, bound_right, bound_action = st.columns([2, 2, 1])
    # Locking makes the region immutable rather than copying it, so there is still one
    # region and every control that could change it is disabled from one flag.
    with bound_left:
        st.number_input("Region start (s)", min_value=0.0, max_value=float(duration),
                        step=0.001, format="%.4f", key="stm_region_start",
                        disabled=locked)
    with bound_right:
        st.number_input("Region end (s)", min_value=0.0, max_value=float(duration),
                        step=0.001, format="%.4f", key="stm_region_end",
                        disabled=locked)
    with bound_action:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("Use current window", use_container_width=True, disabled=locked,
                     help="Set the region to the window shown above."):
            st.session_state["stm_pending_region"] = (
                selection.start_seconds or 0.0, selection.end_seconds or 0.0)
            st.rerun()

    region = select_region(samples_total, sample_rate,
                           float(st.session_state["stm_region_start"]),
                           float(st.session_state["stm_region_end"]))

    if not region.ok:
        st.warning(region.reason)
        region_start_sample = region_end_sample = -1
    else:
        chosen = region.selection
        region_start_sample = chosen.start_sample
        region_end_sample = chosen.end_sample
        summary = st.columns(4)
        summary[0].metric("Region start", f"{chosen.start_seconds:.4f} s")
        summary[1].metric("Region end", f"{chosen.end_seconds:.4f} s")
        summary[2].metric("Region duration", f"{chosen.duration_seconds * 1000:.1f} ms")
        summary[3].metric("Region samples", f"{chosen.length_samples:,}")
        st.caption(
            f"Samples {chosen.start_sample:,} to {chosen.end_sample:,} of "
            f"{samples_total:,} - this exact span is what will be analyzed.")

        # The name is a label the analyst attaches to this investigation. It is never
        # passed to the pipeline and never touches the region bounds - it stays editable
        # while locked precisely because the flow is lock, then name, then analyze.
        resolved_name = investigation_name(
            st.session_state["stm_region_name"],
            len(st.session_state["stm_investigations"]) + 1)
        st.text_input(
            "Investigation name (optional)", key="stm_region_name", max_chars=80,
            placeholder=resolved_name,
            help="A note to yourself, shown on the locked banner and in the history "
                 "below. It does not affect the analysis.")
        st.text_input(
            "Investigation notes / bookmark (optional)", key="stm_region_notes", max_chars=140,
            placeholder="e.g. Tone burst, carrier frequency deviation, candidate preamble",
            help="Optional investigator note saved with this region analysis.")

        # Lock / unlock / focus. None of these touch the pipeline: locking flips a flag,
        # focus moves the playback window. Only Analyze Selection below runs anything.
        lock_row = st.columns([2, 1, 1])
        with lock_row[0]:
            if locked:
                if st.button("Unlock Signal", use_container_width=True,
                             help="Release the investigation target so a new region "
                                  "can be selected."):
                    st.session_state["stm_locked"] = False
                    st.session_state["stm_drag_reason"] = ""
                    st.rerun()
            elif st.button("Lock Signal", use_container_width=True,
                           help="Hold this region as the investigation target while "
                                "you explore the rest of the recording."):
                st.session_state["stm_locked"] = True
                st.rerun()
        with lock_row[1]:
            if st.button("Focus", use_container_width=True,
                         help="Move the playback window onto this region. The region "
                              "itself does not change."):
                st.session_state["stm_pending_cursor"] = chosen.start_seconds
                st.session_state["stm_pending_window_ms"] = \
                    chosen.duration_seconds * 1_000.0
                st.rerun()
        if locked:
            st.caption(
                "Locked. The region bounds above are held and the drag surface will "
                "not replace them. Playback, freeze and scrubbing all still work.")

        if st.button("Analyze Selection", type="primary", use_container_width=True):
            with st.spinner("Running the RadioFry pipeline on the selected region..."):
                try:
                    from radiofry.pipeline import analyze_capture

                    selection_report = analyze_capture(region_container(signal, chosen))
                except (OSError, ValueError, RuntimeError, ImportError) as error:
                    st.error(f"Region analysis failed: {error}")
                else:
                    st.session_state["stm_investigations"].append({
                        "name": investigation_name(
                            st.session_state["stm_region_name"],
                            len(st.session_state["stm_investigations"]) + 1),
                        "start_s": chosen.start_seconds,
                        "end_s": chosen.end_seconds,
                        "samples": chosen.length_samples,
                        "start_sample": chosen.start_sample,
                        "end_sample": chosen.end_sample,
                        "notes": st.session_state.get("stm_region_notes", "").strip(),
                        "report": selection_report,
                    })
                    st.rerun()

investigations = st.session_state.get("stm_investigations", [])
if investigations:
    st.markdown("<div class='evidence-label' style='margin-top:1.5rem;'>"
                "Selection analysis</div>", unsafe_allow_html=True)
    # Name first, bounds after: the name is how the analyst tells two entries apart, but
    # the timestamps and sample count stay on the row so nothing is identified by a
    # label alone.
    labels = [
        f"#{index + 1}  {investigation_name(item.get('name'), index + 1)}  "
        f"- {item['start_s']:.4f}-{item['end_s']:.4f} s "
        f"({item['samples']:,} samples)"
        for index, item in enumerate(investigations)]
    picked = st.selectbox("Investigation", list(range(len(investigations))),
                          index=len(investigations) - 1,
                          format_func=lambda i: labels[i])
    investigation = investigations[picked]
    selection_report = investigation["report"]

    st.caption(
        f"\"{investigation_name(investigation.get('name'), picked + 1)}\" - analysis of "
        f"the selected region only: samples "
        f"{investigation['start_sample']:,} to {investigation['end_sample']:,}, "
        f"{investigation['start_s']:.4f} s to {investigation['end_s']:.4f} s. "
        "The name is a label only; the bounds above are what was analyzed. "
        "The whole-capture results below are unchanged.")
    if investigation.get("notes"):
        st.info(f"**Investigator notes:** {html.escape(investigation['notes'])}")

    rows = compare_reports(report, selection_report)
    if rows:
        st.markdown("**Whole capture vs selected region**")
        st.table({
            "Field": [row[0] for row in rows],
            "Whole capture": [row[1] for row in rows],
            "Selected region": [row[2] for row in rows],
        })
        st.caption("Only fields the pipeline returned for both are compared; anything "
                   "missing on either side is left out rather than filled in.")
    else:
        st.info("No field was returned by the pipeline for both the whole capture and "
                "this region, so there is nothing to compare.")

    demodulation = (selection_report.get("stages", {}) or {}).get("demodulation") or {}
    result = demodulation.get("result") if isinstance(demodulation, dict) else None
    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("<div class='evidence-panel'><h3>Region demodulation</h3>",
                    unsafe_allow_html=True)
        if isinstance(demodulation, dict) and demodulation.get("available") and result:
            st.markdown(f"**Scheme** &nbsp; {result.get('modulation', UNAVAILABLE)}")
            bits = result.get("bits")
            st.markdown(f"**Recovered bits** &nbsp; "
                        f"{len(bits):,}" if bits is not None else
                        f"**Recovered bits** &nbsp; {UNAVAILABLE}")
        else:
            message = demodulation.get("message") if isinstance(demodulation, dict) else ""
            st.info(message or "Demodulation was not available for this region.")
        st.markdown("</div>", unsafe_allow_html=True)
    with detail_right:
        st.markdown("<div class='evidence-panel'><h3>Region bitstream stages</h3>",
                    unsafe_allow_html=True)
        bitstream = (selection_report.get("stages", {}) or {}).get(
            "bitstream_analysis") or {}
        if bitstream.get("available"):
            st.markdown(f"**Interleaver** &nbsp; "
                        f"{bitstream.get('selected_interleaver', UNAVAILABLE)}")
            st.markdown(f"**FEC** &nbsp; {bitstream.get('selected_fec', UNAVAILABLE)}")
            st.markdown(f"**Bits** &nbsp; {bitstream.get('bits', UNAVAILABLE):,}"
                        if isinstance(bitstream.get("bits"), int)
                        else f"**Bits** &nbsp; {UNAVAILABLE}")
        else:
            st.info(bitstream.get("message")
                    or "No bitstream stages ran for this region.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Full selection report (JSON)", expanded=False):
        st.json(selection_report)

    if st.button("Clear investigations"):
        st.session_state["stm_investigations"] = []
        st.rerun()

# ---------------------------------------------------------------- parameters
st.markdown("<div class='evidence-label' style='margin-top:1.5rem;'>Parameters</div>",
            unsafe_allow_html=True)
global_column, window_column = st.columns(2)

with global_column:
    st.markdown("<div class='evidence-panel'><h3>Whole-capture analysis</h3>"
                "<p style='color:var(--muted)'>From the existing RadioFry pipeline. "
                "These describe the <b>entire recording</b> and were <b>not</b> "
                "recomputed for the selected window.</p>", unsafe_allow_html=True)
    fusion = (report or {}).get("stages", {}).get("fusion", {}) or {}
    label = fusion.get("label") if isinstance(fusion, dict) else getattr(fusion, "label", None)
    trust = fusion.get("trust_score") if isinstance(fusion, dict) else getattr(fusion, "trust_score", None)
    rows = [
        ("Modulation", label or UNAVAILABLE),
        ("Modulation trust", f"{trust:.3f}" if isinstance(trust, (int, float)) else UNAVAILABLE),
        ("Carrier frequency", f"{parameters['carrier_frequency_hz']:,.1f} Hz"
         if parameters.get("carrier_frequency_hz") is not None else UNAVAILABLE),
        ("Occupied bandwidth", f"{parameters['occupied_bandwidth_hz']:,.1f} Hz"
         if parameters.get("occupied_bandwidth_hz") is not None else UNAVAILABLE),
        ("Estimated SNR", f"{parameters['snr_db']:.1f} dB"
         if parameters.get("snr_db") is not None else UNAVAILABLE),
        ("Symbol rate", f"{symbol_rate:,.1f} Hz" if symbol_rate is not None else UNAVAILABLE),
    ]
    for name, value in rows:
        st.markdown(f"**{name}** &nbsp; {value}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with window_column:
    st.markdown("<div class='evidence-panel'><h3>Selected window</h3>"
                "<p style='color:var(--muted)'>Measured from <b>these samples only</b>. "
                "Amplitude statistics and the spectral peak are all that can honestly "
                "be derived from a window this short.</p>", unsafe_allow_html=True)
    measurements = window_measurements(
        window, sample_rate,
        spectrum_data=(frequencies, magnitude_db) if 'frequencies' in locals() and frequencies.size else None)
    if not measurements:
        st.info(UNAVAILABLE)
    else:
        st.markdown(f"**Samples** &nbsp; {int(measurements['samples']):,}")
        st.markdown(f"**RMS amplitude** &nbsp; {measurements['rms_amplitude']:.4f}")
        st.markdown(f"**Peak amplitude** &nbsp; {measurements['peak_amplitude']:.4f}")
        st.markdown(
            f"**Amplitude CV** &nbsp; {measurements['amplitude_cv']:.4f}"
            if "amplitude_cv" in measurements else f"**Amplitude CV** &nbsp; {UNAVAILABLE}")
        st.markdown(
            f"**Spectral peak offset** &nbsp; {measurements['peak_offset_hz']:,.1f} Hz"
            if "peak_offset_hz" in measurements
            else f"**Spectral peak offset** &nbsp; {UNAVAILABLE}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- playback pacing
if st.session_state.get("stm_playing") and not st.session_state.get("stm_frozen") \
        and duration is not None:
    # The cursor was already advanced at the top of this run. Pace the animation so the
    # rerun cycle does not spin as fast as the machine allows.
    time.sleep(0.15)
    st.rerun()

render_method_note(
    "Method and limits",
    "Every view on this page is computed from the real samples in the selected window; "
    "nothing is synthesised. Waveform and scatter points are decimated for display but "
    "are untouched sample values. The whole-capture parameters shown on the left come "
    "from the existing analysis and are NOT recomputed as the cursor moves - only the "
    "'Selected window' panel is measured per position. The IQ scatter is a raw I/Q "
    "plot, not a decoded constellation, because symbol timing is not re-established for "
    "each window. Playback advances the analysis cursor through the recording; it does "
    "not produce audio. Region analysis runs the existing RadioFry pipeline - the same "
    "code path as the whole-capture analysis - on exactly the selected samples, and "
    "only when the button is pressed. Its results describe that region alone. The "
    "overview used for dragging is a spectrogram of the real capture, and the "
    "highlighted band on both the overview and the waveform is drawn from the same "
    "validated region bounds as the numeric controls, so it always marks the samples "
    "that would actually be analyzed. Dragging is a selection gesture only: it runs no "
    "classifier, no demodulator and no pipeline stage. Locking a signal holds that same "
    "region rather than copying it - there is still one investigation region, and while "
    "it is locked nothing but an explicit unlock can change its bounds. Lock, unlock and "
    "focus are all free: they move the view or set a flag, never the pipeline. A locked "
    "region reports a modulation only after Analyze Selection has run on exactly its "
    "samples; until then it says so, and it never borrows the whole-capture answer or "
    "the result of a different region.")
