"""Where RadioFry has been measured, and where it degrades.

This page plots the recorded end-to-end benchmark, nothing else. It does not run the
pipeline, does not depend on the loaded capture, and never fills a gap: a condition that
was not measured stays visibly absent, and a condition that was measured but produced no
bits is shown as its own category rather than as missing data.
"""

import sys
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
from radiofry.evaluation.capability_surface import (
    DEFAULT_BER_THRESHOLD,
    DEFAULT_LOG_FLOOR,
    METRICS,
    NO_OUTPUT,
    available_modulations,
    build_surface,
    compare_modulations,
    coverage_table,
    load_benchmark,
    log_values,
)

render_page_shell(11)
render_stage_header(
    "11",
    "Capability envelope",
    "The SNR and samples-per-symbol conditions RadioFry has actually been measured at - "
    "and where its performance falls away.",
    "Evidence", "ready")


@st.cache_data(show_spinner=False)
def _load():
    """Benchmark artifacts change only when a benchmark is re-run."""
    return load_benchmark()


source = _load()

if not source.ok:
    render_empty_state(
        "No benchmark evidence is available in this checkout.",
        "This page plots recorded measurements only. With no benchmark artifact there is "
        "nothing measured to show, and inventing a surface would defeat the purpose of "
        "the page.")
    for warning in source.warnings:
        st.warning(warning)
    render_method_note(
        "Where the evidence comes from",
        "reports/benchmark_v039/results.pkl, produced by the Entry 039 end-to-end "
        "benchmark. That directory is gitignored, so it exists only where the benchmark "
        "has been run locally.")
    st.stop()

for warning in source.warnings:
    st.warning(warning)

modulations = available_modulations(source.records)

# ---------------------------------------------------------------- controls
controls = st.columns([2, 2, 2, 2])
with controls[0]:
    modulation = st.selectbox("Modulation", modulations,
                              index=modulations.index("QPSK")
                              if "QPSK" in modulations else 0)
with controls[1]:
    metric = st.selectbox(
        "Metric", list(METRICS), index=0,
        format_func=lambda key: METRICS[key]["label"])
with controls[2]:
    threshold = st.select_slider(
        "BER threshold", options=[1e-4, 1e-3, 1e-2, 1e-1],
        value=DEFAULT_BER_THRESHOLD, format_func=lambda v: f"{v:.0e}",
        help="A reading aid only. RadioFry defines no pass/fail BER; this colours cells "
             "so the boundary of usable performance is visible.")
with controls[3]:
    log_scale = st.toggle("Log BER axis", value=True,
                          help="BER spans four orders of magnitude, so a linear axis "
                               "flattens everything below about 0.05 into the floor.")

surface = build_surface(source.records, modulation, metric, float(threshold))
is_ber = surface.kind == "ber"
use_log = bool(log_scale) and is_ber

st.caption(f"Evidence: {source.description} · {METRICS[metric]['description']}")

if not surface.ok:
    for warning in surface.warnings:
        st.warning(warning)
    st.stop()

# ---------------------------------------------------------------- coverage
st.markdown("<div class='evidence-label'>Coverage</div>", unsafe_allow_html=True)
coverage = st.columns(5)
coverage[0].metric("Captures", f"{surface.total_captures:,}")
coverage[1].metric("Measurements", f"{surface.total_measurements:,}")
coverage[2].metric("Cells measured",
                   f"{surface.measured_cells} / {surface.total_cells}")
coverage[3].metric("SNR points tested", f"{surface.snr_axis.size}")
coverage[4].metric("SPS points tested", f"{surface.sps_axis.size}")
st.caption(
    surface.coverage_note() +
    f" Tested at SNR {', '.join(f'{v:g}' for v in surface.snr_axis)} dB and "
    f"samples-per-symbol {', '.join(str(int(v)) for v in surface.sps_axis)}. "
    "These are discrete measured points, not a continuous validated region.")
for warning in surface.warnings:
    st.warning(warning)

# ---------------------------------------------------------------- surface
plotted = log_values(surface.values, DEFAULT_LOG_FLOOR) if use_log else surface.values
if use_log:
    z_title, z_ticks = "log10(median BER)", None
elif is_ber:
    z_title, z_ticks = "Median BER", None
else:
    z_title, z_ticks = surface.metric_label, None

detail = np.empty(surface.values.shape, dtype=object)
for row in range(surface.values.shape[0]):
    for column in range(surface.values.shape[1]):
        value = surface.values[row, column]
        detail[row, column] = (
            f"{surface.captures[row, column]} captures, "
            f"{surface.measurements[row, column]} measured, "
            f"{surface.classified_correct[row, column]} classified correctly<br>"
            f"status: {surface.status[row, column]}<br>"
            + (f"value: {value:.5f}" if np.isfinite(value) else "value: not measured"))

left, right = st.columns([3, 2])
with left:
    figure = go.Figure(go.Surface(
        x=surface.snr_axis, y=surface.sps_axis, z=plotted,
        customdata=detail, colorscale="RdYlGn_r" if is_ber else "RdYlGn",
        connectgaps=False, showscale=True,
        colorbar={"title": {"text": z_title}},
        hovertemplate=("SNR %{x:.0f} dB<br>sps %{y}<br>" + z_title +
                       " %{z:.3f}<br>%{customdata}<extra></extra>")))
    figure.update_layout(
        height=520, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "xaxis": {"title": {"text": "SNR (dB)"},
                      "tickvals": surface.snr_axis.tolist()},
            "yaxis": {"title": {"text": "Samples per symbol"},
                      "tickvals": surface.sps_axis.tolist()},
            "zaxis": {"title": {"text": z_title}},
            "camera": {"eye": {"x": 1.7, "y": -1.5, "z": 0.8}},
        })
    st.plotly_chart(figure, use_container_width=True, key="capability_surface")
    st.caption(
        f"{modulation}: {surface.metric_label.lower()}. Gaps in the mesh are conditions "
        "with no measurement - the surface is deliberately not closed across them. "
        + (f"BER = 0 is drawn at the {DEFAULT_LOG_FLOOR:.0e} floor, since a log axis has "
           "no position for zero; those cells mean no errors were seen in the bits "
           "compared, not that the link is error-free. " if use_log else "")
        + "Each point is the median over the seeds run at that condition, so the surface "
        "interpolates between measured points for display only.")

with right:
    status_colours = {"validated": "#3fb950", "degraded": "#f2c14e",
                      NO_OUTPUT: "#d1495b", "not_benchmarked": "#30363d"}
    codes = np.vectorize(
        {"validated": 3, "degraded": 2, NO_OUTPUT: 1, "not_benchmarked": 0}.get
    )(surface.status).astype(float)
    status_map = go.Figure(go.Heatmap(
        x=surface.snr_axis, y=surface.sps_axis, z=codes,
        customdata=detail, zmin=0, zmax=3,
        colorscale=[[0.0, status_colours["not_benchmarked"]],
                    [0.33, status_colours[NO_OUTPUT]],
                    [0.66, status_colours["degraded"]],
                    [1.0, status_colours["validated"]]],
        showscale=False, xgap=2, ygap=2,
        hovertemplate="SNR %{x:.0f} dB<br>sps %{y}<br>%{customdata}<extra></extra>"))
    status_map.update_layout(
        height=300, margin={"l": 0, "r": 0, "t": 24, "b": 0},
        xaxis={"title": "SNR (dB)", "tickvals": surface.snr_axis.tolist()},
        yaxis={"title": "Samples per symbol", "tickvals": surface.sps_axis.tolist()})
    st.plotly_chart(status_map, use_container_width=True, key="capability_status")
    if is_ber:
        st.markdown(
            f"<div style='font-size:0.82rem;line-height:1.7'>"
            f"<span style='color:#3fb950'>&#9632;</span> measured at or below "
            f"{threshold:.0e} BER &nbsp; "
            f"<span style='color:#f2c14e'>&#9632;</span> measured above it &nbsp;<br>"
            f"<span style='color:#d1495b'>&#9632;</span> run, but produced no bits to "
            f"score &nbsp; "
            f"<span style='color:#30363d'>&#9632;</span> never benchmarked</div>",
            unsafe_allow_html=True)
        st.caption(
            "The red category is a limitation of the pipeline at that condition, not a "
            "gap in the experiment: captures were analyzed and no bitstream came out.")

# ---------------------------------------------------------------- per-cell provenance
with st.expander("Per-cell measurements", expanded=False):
    rows = coverage_table(surface)
    st.table({
        "SPS": [row["sps"] for row in rows],
        "SNR (dB)": [f"{row['snr_db']:g}" for row in rows],
        "Captures": [row["captures"] for row in rows],
        "Measured": [row["measurements"] for row in rows],
        "Classified correctly": [row["classified_correct"] for row in rows],
        surface.metric_label: [
            "not measured" if row["value"] is None else f"{row['value']:.5f}"
            for row in rows],
        "Status": [row["status"] for row in rows],
    })

# ---------------------------------------------------------------- comparison
st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "Across modulations</div>", unsafe_allow_html=True)
summary = compare_modulations(source.records, metric, float(threshold))
st.table({
    "Modulation": [row["modulation"] for row in summary],
    "Captures": [f"{row['captures']:,}" for row in summary],
    "Cells measured": [f"{row['measured_cells']}/{row['total_cells']}"
                       for row in summary],
    "Cells at or below threshold": [row["validated_cells"] for row in summary],
    "Cells with no output": [row["no_output_cells"] for row in summary],
    f"{surface.metric_label} (across cells)": [
        "no measurement" if row["median"] is None else f"{row['median']:.5f}"
        for row in summary],
    "Best cell": ["-" if row["best"] is None else f"{row['best']:.5f}"
                  for row in summary],
})
st.caption(
    "Median and best are taken over the cells that carry a measurement, so a modulation "
    "with few measured cells is summarising fewer conditions than one with many. Read "
    "the 'cells measured' column alongside the medians.")

render_method_note(
    "What this surface is, and what it is not",
    "Every plotted value is the median of measurements recorded by the Entry 039 "
    "end-to-end benchmark: 480 digital captures over 8 production classes x "
    "samples-per-symbol 4/8/16/32 x SNR 20/15/10/5/0 dB x 3 seeds, at 200 kHz with 8192 "
    "samples per capture, scored against generator ground truth after the pipeline ran. "
    "Nothing is interpolated into a cell that holds no measurement, and no value is "
    "extrapolated beyond the tested grid. The production BER population is every capture "
    "in a cell for which the pipeline emitted bits, including those it had misclassified "
    "- demodulating with the wrong scheme really does yield garbage bits, and excluding "
    "those would flatter the surface - so the count classified correctly is shown "
    "alongside. The oracle metric replaces the estimated symbol rate with the true one; "
    "the gap between the two surfaces is the symbol-rate estimator's contribution, which "
    "Entry 039 identified as the dominant end-to-end failure mode. The BER threshold "
    "colours cells as a reading aid and is configurable; RadioFry defines no pass/fail "
    "BER, and no region here should be read as a guarantee. This is an experimental "
    "envelope over synthetic captures at one sample rate and one capture length: real "
    "signals with different impairments may behave differently, and conditions outside "
    "the tested grid carry no evidence either way.")
