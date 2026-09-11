"""Why RadioFry accepted, rejected or rerouted this signal.

Every region drawn here is produced by calling the real `fuse_modulation`, so the picture
cannot drift from the code it explains. The page is a rendering shell: the evaluation and
the capture extraction live in `radiofry.fusion.decision_landscape`.
"""

import html
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
from radiofry.fusion.decision_landscape import (
    DEFAULT_THRESHOLD,
    OUTCOMES,
    LandscapeScenario,
    describe_capture,
    evaluate_landscape,
    scenario_from_report,
    scenario_variants,
    threshold_lines,
)

# One colour per implemented branch. Green accepts, amber wants a human, red refuses.
OUTCOME_COLOURS = {
    "accepted": "#3fb950",
    "accepted_review": "#9ccc5a",
    "analog_subtype": "#4aa8d8",
    "analog_recovery": "#7fe7c4",
    "guard_reroute": "#f2c14e",
    "guard_block": "#d1495b",
    "rejected": "#6e7681",
}

render_page_shell(12)
signal = st.session_state.get("signal")
report = st.session_state.get("report")
render_stage_header(
    "12",
    "Fusion decision landscape",
    "The implemented decision boundaries of the CNN + classical fusion rules, and where "
    "this capture falls inside them.",
    "Ready" if report else "Waiting",
    "ready" if report else "muted",
)

if not report:
    render_empty_state(
        "Analyze a WAV or IQ capture on the home page first.",
        "This stage explains the fusion decision that was made for a capture, so it "
        "needs an analysis to explain.")
    render_method_note(
        "What this stage is",
        "A map of the rules in radiofry.fusion.confidence_fusion, drawn by running them. "
        "It is not a learned probability surface and it changes no pipeline behaviour.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=12)
def _landscape(ml_label: str, classical_family: str,
               ranked: tuple[tuple[str, float], ...],
               evidence_items: tuple[tuple[str, float], ...] | None,
               threshold: float, resolution: int):
    """Cached on the exact scenario; the grid itself is pure and cheap to recompute."""
    scenario = LandscapeScenario(
        ml_label=ml_label, classical_family=classical_family,
        ranked_alternatives=ranked,
        classical_evidence=dict(evidence_items) if evidence_items else None,
        threshold=threshold)
    return evaluate_landscape(scenario, resolution=resolution)


position = describe_capture(report, DEFAULT_THRESHOLD)

# ---------------------------------------------------------------- this capture
st.markdown("<div class='evidence-label'>This capture</div>", unsafe_allow_html=True)

summary = st.columns(5)
summary[0].metric("CNN prediction",
                  position.cnn_label or "Unavailable",
                  delta=(f"{position.cnn_confidence:.3f} confidence"
                         if position.cnn_confidence is not None else None),
                  delta_color="off")
summary[1].metric("Classical family",
                  position.classical_family or "Unavailable",
                  delta=(f"{position.classical_confidence:.3f} confidence"
                         if position.classical_confidence is not None else None),
                  delta_color="off")
summary[2].metric("Estimated SNR",
                  f"{position.snr_db:.1f} dB" if position.snr_db is not None else "n/a")
summary[3].metric("Fusion decision", position.fused_label or "Unavailable",
                  delta=(f"trust {position.trust_score:.3f}"
                         if position.trust_score is not None else None),
                  delta_color="off")
summary[4].metric("Review", "Recommended" if position.review_recommended else "Not flagged")

if position.outcome:
    outcome = OUTCOMES[position.outcome]
    colour = OUTCOME_COLOURS.get(position.outcome, "#8b949e")
    st.markdown(
        f"<div class='evidence-panel' style='border-left:4px solid {colour}'>"
        f"<div class='evidence-label' style='color:{colour}'>"
        f"{html.escape(outcome['label'])}</div>"
        f"<p style='margin:0.35rem 0 0'>{html.escape(outcome['detail'])}</p></div>",
        unsafe_allow_html=True)

if position.guards:
    st.warning("Guards that fired: " + "; ".join(position.guards))
else:
    st.caption("No guard or reroute fired for this capture: the decision came from the "
               "acceptance threshold and the family-agreement check alone.")
for note in position.notes:
    st.warning(note)
if position.reproduced is False:
    st.error(
        "Re-running the recorded inputs through fusion did not reproduce the recorded "
        "decision, so the point below may not belong to this landscape.")

if not position.plottable:
    st.info("This capture has no position on the confidence plane, so only the decision "
            "regions are shown below.")

# ---------------------------------------------------------------- scenario
base = scenario_from_report(report, DEFAULT_THRESHOLD) or LandscapeScenario()
variants = scenario_variants(base)

st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "Decision regions</div>", unsafe_allow_html=True)
controls = st.columns([3, 2, 2])
with controls[0]:
    variant_name = st.selectbox(
        "Context", list(variants),
        help="Fusion branches on categorical context - which label the CNN emitted and "
             "which family the classical detector returned. Switch context to see the "
             "guards this capture did not trigger.")
with controls[1]:
    threshold = st.select_slider(
        "Acceptance threshold", options=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        value=DEFAULT_THRESHOLD,
        help="The production default is 0.4. Changing it here only redraws the map; it "
             "does not alter any analysis.")
with controls[2]:
    resolution = st.select_slider("Grid", options=[41, 61, 81, 121], value=81,
                                  help="Cells per axis. Each cell is one evaluation of "
                                       "the real fusion function.")

scenario = variants[variant_name]
if threshold != scenario.threshold:
    scenario = LandscapeScenario(
        ml_label=scenario.ml_label, classical_family=scenario.classical_family,
        ranked_alternatives=scenario.ranked_alternatives,
        classical_evidence=scenario.classical_evidence, threshold=float(threshold))

grid = _landscape(
    scenario.ml_label, scenario.classical_family, scenario.ranked_alternatives,
    tuple(sorted(scenario.classical_evidence.items()))
    if scenario.classical_evidence else None,
    float(scenario.threshold), int(resolution))

st.caption(f"Context: {scenario.describe()} · {resolution}x{resolution} evaluations of "
           "the production fusion function.")
for warning in grid.warnings:
    st.info(warning)

present = grid.present_outcomes
codes = {key: index for index, key in enumerate(present)}
numeric = np.vectorize(codes.get)(grid.outcomes).astype(float)
hover = np.empty(grid.outcomes.shape, dtype=object)
for row in range(grid.outcomes.shape[0]):
    for column in range(grid.outcomes.shape[1]):
        hover[row, column] = (
            f"{OUTCOMES[grid.outcomes[row, column]]['label']}<br>"
            f"decision: {grid.labels[row, column]}<br>"
            f"trust {grid.trust[row, column]:.3f} · "
            f"review {'yes' if grid.review[row, column] else 'no'}")

if len(present) == 1:
    colorscale = [[0.0, OUTCOME_COLOURS.get(present[0], "#8b949e")],
                  [1.0, OUTCOME_COLOURS.get(present[0], "#8b949e")]]
else:
    colorscale = [[index / (len(present) - 1),
                   OUTCOME_COLOURS.get(key, "#8b949e")]
                  for index, key in enumerate(present)]

map_column, surface_column = st.columns(2)

with map_column:
    figure = go.Figure(go.Heatmap(
        x=grid.cnn_axis, y=grid.classical_axis, z=numeric, customdata=hover,
        colorscale=colorscale, showscale=False, zmin=0, zmax=max(1, len(present) - 1),
        hovertemplate=("CNN confidence %{x:.3f}<br>classical confidence %{y:.3f}"
                       "<br>%{customdata}<extra></extra>")))
    for line in threshold_lines(scenario):
        if line["axis"] == "cnn":
            figure.add_vline(x=line["value"], line_width=2, line_dash="dash",
                             line_color="#e6edf3",
                             annotation_text=f"{line['name']} = {line['value']:g}",
                             annotation_position="top")
        else:
            figure.add_hline(y=line["value"], line_width=2, line_dash="dash",
                             line_color="#e6edf3",
                             annotation_text=f"{line['name']} = {line['value']:g}",
                             annotation_position="right")
    if position.plottable and variant_name == "This capture":
        figure.add_scatter(
            x=[position.cnn_confidence], y=[position.classical_confidence],
            mode="markers+text", text=["this capture"], textposition="top center",
            marker={"size": 14, "color": "#ffffff", "symbol": "x-thin",
                    "line": {"width": 3, "color": "#ffffff"}},
            hovertemplate=(f"this capture<br>CNN {position.cnn_confidence:.3f}"
                           f"<br>classical {position.classical_confidence:.3f}"
                           "<extra></extra>"),
            showlegend=False)
    figure.update_layout(
        height=430, margin={"l": 0, "r": 0, "t": 30, "b": 0},
        xaxis_title="CNN confidence", yaxis_title="Classical confidence")
    st.plotly_chart(figure, use_container_width=True, key="fusion_map")
    st.caption(
        "Each cell is one evaluation of the production fusion function, not a model of "
        "it. Boundaries are hard: fusion is a chain of threshold rules, so the regions "
        "meet at discontinuities rather than fading into one another.")

with surface_column:
    surface = go.Figure(go.Surface(
        x=grid.cnn_axis, y=grid.classical_axis, z=grid.trust, customdata=hover,
        surfacecolor=numeric, colorscale=colorscale, showscale=False,
        cmin=0, cmax=max(1, len(present) - 1),
        hovertemplate=("CNN confidence %{x:.3f}<br>classical confidence %{y:.3f}"
                       "<br>trust %{z:.3f}<br>%{customdata}<extra></extra>")))
    surface.update_layout(
        height=430, margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={"xaxis": {"title": {"text": "CNN confidence"}},
               "yaxis": {"title": {"text": "Classical confidence"}},
               "zaxis": {"title": {"text": "Trust score"}},
               "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 0.9}}})
    st.plotly_chart(surface, use_container_width=True, key="fusion_surface")
    st.caption(
        "Height is the trust score fusion assigns, coloured by the branch taken. Trust "
        "is a heuristic agreement score - confidence x 1.1 when the classical family "
        "agrees, x 0.5 when it does not - and is not a calibrated probability.")

# ---------------------------------------------------------------- legend
st.markdown("<div class='evidence-label' style='margin-top:1.25rem;'>"
            "What each region means</div>", unsafe_allow_html=True)
rows = "".join(
    f"<div style='margin-bottom:0.45rem'>"
    f"<span style='color:{OUTCOME_COLOURS.get(key, '#8b949e')};font-size:1.1rem'>"
    f"&#9632;</span> <b>{html.escape(OUTCOMES[key]['label'])}</b> &mdash; "
    f"{html.escape(OUTCOMES[key]['detail'])}</div>"
    for key in present)
st.markdown(f"<div class='evidence-panel'>{rows}</div>", unsafe_allow_html=True)

render_method_note(
    "What this landscape is, and what it is not",
    "Every region is produced by calling radiofry.fusion.confidence_fusion."
    "fuse_modulation with the axes as its confidence arguments, so the map cannot "
    "disagree with the implementation - it is the implementation, sampled. The axes are "
    "the two continuous quantities fusion actually reads: CNN confidence, where the "
    "acceptance threshold lives, and classical confidence, where the digital-family "
    "sufficiency floor lives. SNR is deliberately NOT an axis: it is not an argument to "
    "fuse_modulation and no threshold in the fusion code refers to it. SNR shapes the "
    "inputs upstream, which is why it is reported for the capture above, but plotting it "
    "as a decision axis would invent a dimension the logic does not have. Everything "
    "else fusion reads is categorical - the CNN's label, the classical family, the "
    "ranked alternatives, the envelope evidence - and selects which regime is drawn; use "
    "the context selector to see the guards this capture did not trigger. The ranked "
    "alternatives are held at the values this capture produced while the top-1 "
    "confidence sweeps, which is an idealisation: in a real softmax they would co-vary. "
    "Nothing here is learned and nothing is a probability: the trust score is an explicit "
    "agreement heuristic, and the regions are threshold rules. This page runs no "
    "classifier and changes no pipeline behaviour.")
