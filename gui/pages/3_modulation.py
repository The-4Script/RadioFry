import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gui.theme import apply_radiofry_theme, render_empty_state, render_method_note, render_stage_header

apply_radiofry_theme()
report = st.session_state.get("report")
modulation = report["stages"].get("fusion", {}) if report else {}
status = "Review" if modulation.get("review_recommended") else "Ready"
status_kind = "review" if modulation.get("review_recommended") else "ready"
render_stage_header("03", "Modulation", "Compare independent signal evidence before choosing a modulation hypothesis.", status, status_kind)

if report is None:
    render_empty_state("Analyze a capture on the home page first.")
else:
    classical = report["stages"].get("classical_modulation", {})
    prediction = report["stages"].get("cnn_modulation", {})
    fusion = report["stages"].get("fusion", {})
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='evidence-panel'><h3>Independent checks</h3>", unsafe_allow_html=True)
        st.metric("Classical family", classical.get("family", "Unknown"))
        st.metric("Family confidence", f"{classical.get('confidence', 0):.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='evidence-panel'><h3>Fused hypothesis</h3>", unsafe_allow_html=True)
        st.metric("Decision", fusion.get("label", "Unclassified"))
        st.metric("Trust score", f"{fusion.get('trust_score', 0):.1%}")
        st.caption(f"CNN top-1: {prediction.get('confidence', 0):.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    if prediction.get("available"):
        st.subheader("Model alternatives")
        top_k = prediction.get("top_k", [])
        st.bar_chart({label: probability for label, probability in top_k})
    else:
        st.warning(prediction.get("message", "CNN inference is unavailable."))
    if fusion.get("review_recommended"):
        st.warning("Review recommended: confidence is low or independent evidence disagrees.")

    demodulation = report["stages"].get("demodulation", {})
    symbols = demodulation.get("result", {}).get("symbols", []) if demodulation.get("available") else []
    if symbols:
        st.subheader("Recovered constellation")
        points = np.asarray([complex(item.get("real", 0.0), item.get("imag", 0.0)) if isinstance(item, dict) else complex(float(item), 0.0) for item in symbols], dtype=np.complex64)
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.scatter(points.real, points.imag, s=8, alpha=0.6, color="#147d7b")
        axis.set_xlabel("In-phase")
        axis.set_ylabel("Quadrature")
        axis.grid(alpha=0.25)
        st.pyplot(figure, clear_figure=True)
        with st.expander("Inspect symbol trajectory", expanded=False):
            trajectory = go.Figure(go.Scatter3d(x=points.real, y=points.imag, z=np.arange(points.size), mode="markers", marker={"size": 3, "color": np.arange(points.size), "colorscale": "Turbo"}))
            trajectory.update_layout(height=500, margin={"l": 0, "r": 0, "t": 30, "b": 0})
            st.plotly_chart(trajectory, width="stretch")

    render_method_note("Method and limits", "The fused result combines a classical family estimate with the CNN prediction. The CNN was trained on a finite modulation distribution; out-of-distribution signals require human review and are not guaranteed to be classified.")
