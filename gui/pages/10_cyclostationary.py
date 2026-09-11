"""Cyclostationary spectral correlation: the symbol clock the PSD cannot show.

Advanced analysis, deliberately explicit. Nothing on this page runs until the analyst
presses the button: the estimator costs seconds, not milliseconds, and the ordinary
upload-and-analyze workflow must not pay for it. The page is a rendering shell - the
estimator and every axis it produces live in `radiofry.dsp.spectral_correlation`.
"""

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
from radiofry.dsp.spectral_correlation import (
    SCDConfig,
    compute_scd,
    cycle_profile,
    decimate_surface,
    dominant_cycle_frequencies,
    estimate_cost,
)

render_page_shell(10)
signal = st.session_state.get("signal")
report = st.session_state.get("report")
render_stage_header(
    "10",
    "Cyclostationary SCD",
    "Spectral correlation across frequency and cyclic frequency. A digitally modulated "
    "signal correlates its own sidebands at the symbol rate; stationary noise does not.",
    "Ready" if signal is not None else "Waiting",
    "ready" if signal is not None else "muted",
)

if signal is None:
    render_empty_state(
        "Analyze a WAV or IQ capture on the home page first.",
        "This stage re-examines the capture you already analyzed. It does not load a "
        "separate file and does not change the existing analysis.")
    render_method_note(
        "What this stage is",
        "An advanced, explicitly triggered cyclostationary analysis. It produces no "
        "modulation decision and does not feed the report.")
    st.stop()

sample_rate = signal.sample_rate
samples_total = int(signal.iq.size)
center_frequency = signal.metadata.get(
    "center_frequency_hz", signal.metadata.get("carrier_frequency_hz"))

if not sample_rate:
    st.warning(
        "This capture has no sample rate, so there is no frequency or cyclic-frequency "
        "axis to place. Re-analyze a raw IQ file with its sample rate to enable SCD.")
    render_method_note(
        "Why a sample rate is required",
        "Both axes of an SCD are physical frequencies in Hz. Without a sample rate the "
        "estimator could only return bin indices, which would invite them to be read as "
        "frequencies they are not.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_scd(iq: np.ndarray, rate: float, fft_size: int, alpha_max: float,
                max_frames: int, normalization: str, conjugate: bool,
                center: float | None):
    """Cached on the samples and the exact configuration.

    Re-running an identical configuration is free, so switching between the 3D and 2D
    views or changing the peak count never recomputes the surface.
    """
    return compute_scd(
        iq, rate,
        SCDConfig(fft_size=fft_size, alpha_max_hz=alpha_max, max_frames=max_frames,
                  normalization=normalization, conjugate=conjugate),
        center_frequency_hz=center)


# ---------------------------------------------------------------- configuration
st.markdown("<div class='evidence-label'>Analysis configuration</div>",
            unsafe_allow_html=True)

mode = st.radio(
    "Analysis mode",
    ["Non-conjugate SCD", "Conjugate SCD"], horizontal=True,
    help="Non-conjugate measures the symbol clock. Conjugate measures impropriety: it "
         "responds to real-valued constellations such as BPSK and PAM and stays silent "
         "for circular ones such as QPSK and QAM.")
conjugate = mode.startswith("Conjugate")

controls = st.columns(4)
with controls[0]:
    alpha_max = st.select_slider(
        "Cyclic range (kHz)",
        options=[float(sample_rate) / d / 1_000.0 for d in (32, 16, 8, 4, 2)],
        value=float(sample_rate) / 8 / 1_000.0,
        format_func=lambda v: f"{v:,.1f}",
        help="Highest cyclic frequency searched. This sets the frame hop: a wider range "
             "forces a shorter hop, which shortens the span the frames cover and "
             "coarsens cyclic resolution.")
with controls[1]:
    fft_size = st.select_slider(
        "FFT size", options=[64, 128, 256, 512], value=128,
        help="Frequency resolution is fs/FFT. A longer FFT resolves frequency better "
             "but leaves fewer independent looks for the cyclic average.")
with controls[2]:
    max_frames = st.select_slider(
        "Frame budget", options=[256, 512, 768, 1536, 3072], value=768,
        help="More frames means a longer span, finer cyclic resolution and "
             "proportionally more computation.")
with controls[3]:
    normalization = st.radio(
        "Magnitude", ["coherence", "density"], horizontal=True,
        help="Coherence is normalised to [0, 1] and comparable across the band. Density "
             "is the raw |S(alpha, f)| and is dominated by wherever the power is.")

config = SCDConfig(fft_size=int(fft_size), alpha_max_hz=float(alpha_max) * 1_000.0,
                   max_frames=int(max_frames), normalization=normalization,
                   conjugate=conjugate)
cost = estimate_cost(samples_total, float(sample_rate), config)

preview = st.columns(4)
preview[0].metric("Frames", f"{cost['frames']:,}")
preview[1].metric("Frame hop", f"{cost['hop']:,} samples")
preview[2].metric("Surface", f"{cost['alpha_bins']:,} x {fft_size}")
preview[3].metric("Span analyzed",
                  f"{cost['duration_seconds'] * 1e3:,.1f} ms"
                  if cost["duration_seconds"] else "n/a")
st.caption(
    f"Reads {cost['samples_used']:,} of {samples_total:,} samples. The surface holds "
    f"{cost['surface_cells']:,} cells ({cost['surface_megabytes']:.2f} MB). Nothing is "
    "computed until you press the button below.")

run = st.button(f"Run {mode}", type="primary", use_container_width=True)
state_key = (samples_total, float(sample_rate), int(fft_size), float(alpha_max),
             int(max_frames), normalization, conjugate)
if run:
    st.session_state["scd_request"] = state_key

requested = st.session_state.get("scd_request")
if requested is None:
    st.info(f"{mode} has not been computed for this capture yet.")
    render_method_note(
        "Method",
        "The estimator is a time-smoothed cyclic periodogram: the capture is shifted by "
        "+/-alpha/2 in the time domain, framed, transformed, and the two shifted spectra "
        "are correlated and averaged across frames. Averaging that per-frame phase is "
        "what makes the vertical axis a cyclic frequency rather than a second time axis, "
        "so this is spectral correlation and not a relabelled spectrogram. Nothing here "
        "modifies the capture, the report, or any pipeline stage.")
    st.stop()

if requested != state_key:
    st.warning(
        "The configuration has changed since the last run. The surface below is the "
        "previous configuration - press Run to recompute.")

with st.spinner("Computing spectral correlation density..."):
    started = time.perf_counter()
    result = _cached_scd(signal.iq, float(sample_rate), requested[2],
                         requested[3] * 1_000.0, requested[4], requested[5],
                         requested[6],
                         float(center_frequency) if center_frequency else None)
    wall_seconds = time.perf_counter() - started

if not result.ok:
    st.error("SCD could not be computed for this capture.")
    for warning in result.warnings:
        st.warning(warning)
    render_method_note(
        "Why this can happen",
        "The estimator refuses rather than guesses. A capture with too few samples to "
        "form frames, no measurable power, or no usable sample rate has no spectral "
        "correlation to report, and returning an empty surface is the honest outcome.")
    st.stop()

# ---------------------------------------------------------------- what was computed
st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "What was computed</div>", unsafe_allow_html=True)
summary = st.columns(5)
summary[0].metric("Samples analyzed", f"{result.samples_used:,}",
                  delta=f"of {result.samples_available:,}", delta_color="off")
summary[1].metric("Cyclic resolution", f"{result.alpha_resolution_hz:,.1f} Hz")
summary[2].metric("Frequency resolution", f"{result.frequency_resolution_hz:,.1f} Hz")
summary[3].metric("Independent looks", f"{result.effective_averages:,.0f}")
summary[4].metric("Compute time",
                  f"{result.elapsed_seconds:,.2f} s" if result.elapsed_seconds
                  else f"{wall_seconds:,.2f} s")
st.caption(
    f"Mode: **{result.mode_label}** · {result.method.replace('_', ' ')} · "
    f"{result.normalization} normalization · "
    f"frame hop {result.hop:,} samples · unambiguous to "
    f"{result.sample_rate / (2 * result.hop):,.0f} Hz cyclic frequency · "
    f"{result.frames:,} frames over "
    f"{(result.duration_seconds or 0) * 1e3:,.1f} ms of the recording.")

if result.normalization == "coherence" and result.effective_averages > 0:
    # Without this an analyst has no way to tell which values on the surface mean
    # anything: coherence between wholly uncorrelated bands still averages to about
    # 1/sqrt(N), so a surface can look textured while carrying no cyclic structure.
    noise_floor = 1.0 / np.sqrt(result.effective_averages)
    st.caption(
        f"Estimator noise floor: coherence of about **{noise_floor:.2f}** arises between "
        f"completely uncorrelated bands at {result.effective_averages:,.0f} independent "
        f"looks. Features below {1.5 * noise_floor:.2f} are not listed at all, but a "
        "listed feature only a little above that is weak evidence - compare each one's "
        "coherence against this floor rather than reading the list as a set of equals.")

for warning in result.warnings:
    st.warning(warning)

# ---------------------------------------------------------------- detected features
features = dominant_cycle_frequencies(result, count=5)
st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "Cyclic features</div>", unsafe_allow_html=True)

if result.conjugate:
    # alpha = 0 is the measurement in this mode, not a ridge to look past.
    propriety = float(cycle_profile(result)[0]) if result.ok else 0.0
    improper = propriety > (1.5 * np.sqrt(np.log(max(2, result.frequencies_hz.size)))
                            / np.sqrt(max(result.effective_averages, 1e-9)))
    st.markdown(
        f"**Impropriety at alpha = 0: {propriety:.3f}** &nbsp; "
        + ("&mdash; consistent with a real-valued constellation (BPSK, PAM) or a "
           "real signal." if improper else
           "&mdash; consistent with a circularly symmetric constellation (QPSK, 8PSK, "
           "QAM) or proper noise."),
        unsafe_allow_html=True)
    st.caption(
        "In conjugate mode alpha = 0 is the measurement, not the power spectrum: it is "
        "the signal's own impropriety, which vanishes for any circularly symmetric "
        "constellation and is large for a real-valued one. This is evidence about the "
        "constellation's symmetry, not a modulation decision - a real-valued signal with "
        "no symbol clock at all reads as improper too, and a frequency offset moves every "
        "conjugate feature to 2*f_offset + k*Rs.")
if features:
    st.table({
        "Cyclic frequency (Hz)": [f"{f.alpha_hz:,.1f}" for f in features],
        result.magnitude_label: [f"{f.magnitude:.4f}" for f in features],
        "Peak / background": [f"{f.peak_to_background:.2f}" for f in features],
        "Strongest at frequency (Hz)": [f"{f.frequency_hz:,.0f}" for f in features],
    })
    reported = (report or {}).get("stages", {}).get("parameters", {}) or {}
    estimated_rs = reported.get("symbol_rate_hz")
    if result.conjugate:
        st.caption(
            "Conjugate features sit at 2*f_offset + k*Rs, so they are not directly "
            "comparable with the reported symbol rate unless the capture is centred. "
            "They are shown as independent evidence about constellation symmetry.")
    elif isinstance(estimated_rs, (int, float)) and estimated_rs:
        closest = min(features, key=lambda f: abs(f.alpha_hz - estimated_rs))
        st.caption(
            f"The existing pipeline reported a symbol rate of {estimated_rs:,.1f} Hz; "
            f"the nearest cyclic feature is at {closest.alpha_hz:,.1f} Hz "
            f"({abs(closest.alpha_hz - estimated_rs):,.1f} Hz away). This is an "
            "independent observation shown for comparison - it does not change the "
            "reported symbol rate, and harmonics at multiples of the symbol rate are "
            "genuine features rather than disagreements.")
    else:
        st.caption(
            "A linearly modulated signal shows features at its symbol rate and at "
            "multiples of it. These are candidates read off the surface; this page does "
            "not claim a symbol rate.")
else:
    st.info(
        "No cyclic feature stood above the estimator's own noise floor. Stationary "
        "signals - noise, a bare carrier - correctly produce nothing here, and so does a "
        "modulated signal whose symbol rate lies outside the cyclic range searched.")

# ---------------------------------------------------------------- 3D surface
st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "Spectral correlation surface</div>", unsafe_allow_html=True)

frequencies, alphas, magnitude = decimate_surface(result)
plot_frequencies = (frequencies + float(center_frequency)) if center_frequency \
    else frequencies

surface = go.Figure(go.Surface(
    x=plot_frequencies / 1_000.0, y=alphas / 1_000.0, z=magnitude,
    colorscale="Viridis", showscale=True,
    colorbar={"title": {"text": "|SCD|" if result.normalization == "density"
                        else "Coherence"}},
    hovertemplate=("f = %{x:.2f} kHz<br>alpha = %{y:.2f} kHz<br>"
                   "magnitude = %{z:.4f}<extra></extra>")))
surface.update_layout(
    height=560, margin={"l": 0, "r": 0, "t": 10, "b": 0},
    scene={
        "xaxis": {"title": {"text": ("RF frequency (kHz)" if center_frequency
                                     else "Baseband frequency (kHz)")}},
        "yaxis": {"title": {"text": "Cyclic frequency alpha (kHz)"}},
        "zaxis": {"title": {"text": result.magnitude_label}},
        "camera": {"eye": {"x": 1.6, "y": -1.5, "z": 0.9}},
    })
st.plotly_chart(surface, use_container_width=True, key="scd_surface")
st.caption(
    f"Drag to rotate, scroll to zoom. Drawn from {magnitude.shape[0]} x "
    f"{magnitude.shape[1]} of the {result.magnitude.shape[0]} x "
    f"{result.magnitude.shape[1]} computed cells - decimated by striding, so every "
    "plotted point is a computed value rather than an interpolated one. The heatmap "
    "below uses the full surface.")

st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "Cyclic profile and the frequency x cyclic frequency projection</div>",
            unsafe_allow_html=True)
profile_column, heatmap_column = st.columns(2)
with profile_column:
    profile = cycle_profile(result)
    line = go.Figure(go.Scatter(x=result.alphas_hz / 1_000.0, y=profile,
                                mode="lines", line={"width": 1.4}))
    for feature in features:
        line.add_vline(x=feature.alpha_hz / 1_000.0, line_width=1,
                       line_dash="dot", line_color="#f2c14e")
    line.update_layout(
        height=250, margin={"l": 0, "r": 0, "t": 24, "b": 0},
        xaxis_title="Cyclic frequency alpha (kHz)",
        yaxis_title="Peak over frequency")
    st.plotly_chart(line, use_container_width=True, key="scd_profile")
    st.caption(
        "Cyclic profile: the strongest correlation found at each cyclic frequency. "
        "alpha = 0 is the ordinary power spectrum and is always maximal, so it is "
        "excluded when features are picked.")

with heatmap_column:
    heat = go.Figure(go.Heatmap(
        x=(result.absolute_frequencies_hz() / 1_000.0), y=result.alphas_hz / 1_000.0,
        z=result.magnitude, colorscale="Viridis",
        colorbar={"title": {"text": "|SCD|" if result.normalization == "density"
                            else "Coherence"}},
        hovertemplate=("f = %{x:.2f} kHz<br>alpha = %{y:.2f} kHz<br>"
                       "magnitude = %{z:.4f}<extra></extra>")))
    heat.update_layout(
        height=250, margin={"l": 0, "r": 0, "t": 24, "b": 0},
        xaxis_title=("RF frequency (kHz)" if center_frequency
                     else "Baseband frequency offset (kHz)"),
        yaxis_title="Cyclic frequency alpha (kHz)")
    st.plotly_chart(heat, use_container_width=True, key="scd_heatmap")
    st.caption(
        "Full computed surface, no decimation. A horizontal band away from alpha = 0 is "
        "a cyclic feature; the band at alpha = 0 is the power spectrum and is present "
        "for every signal, including noise.")

render_method_note(
    "Method, assumptions and limits",
    "The estimate is a time-smoothed cyclic periodogram. The capture is shifted by "
    "+/-alpha/2 in the time domain - exact for any alpha, so both axes stay uniform - "
    "framed with a Hann window, transformed, and the two shifted spectra are correlated "
    "and averaged over frames. Averaging the surviving per-frame phase is what makes "
    "alpha a cyclic frequency; without it this would be an ordinary cross-spectrum. "
    "Only alpha >= 0 is computed, because |S| is even in alpha. The frame hop is derived "
    "from the cyclic range, not chosen: the per-frame phase repeats every fs/hop, so a "
    "longer hop would place spurious unit-coherence ridges at multiples of that "
    "frequency - a bare carrier alone is enough to produce them. Coherence is masked to "
    "zero where either sideband carries no measurable power, because a ratio of leakage "
    "to leakage saturates towards 1 exactly where there is no signal. Cyclic resolution "
    "is 1/T for the span actually analyzed and frequency resolution is fs/FFT; a peak is "
    "never sharper than those. The frequency axis is a baseband offset unless the capture "
    "carried a hardware centre frequency. This page runs no classifier, changes no "
    "pipeline stage, and does not alter the reported symbol rate.")
