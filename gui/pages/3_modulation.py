import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from gui.theme import apply_radiofry_theme, render_section_explainer

apply_radiofry_theme()

st.header("Modulation")
render_section_explainer(
    "Stage 3: decide what kind of signal it is",
    "This is the moment the system tries to answer the big question: what modulation type is this signal using? In everyday terms, it is asking whether the signal looks like BPSK, QPSK, QAM, or something else.",
    "The app combines a classical signal-family estimate with a deep-learning CNN. The fused decision uses independent evidence, reducing the risk of over-trusting a single model. The output includes confidence scores and a constellation-style view of the recovered symbols.",
    "Review the decision, the confidence values, and the constellation. If confidence is low or evidence disagrees, the system will suggest human review instead of pretending certainty.",
)

report = st.session_state.get("report")
if report is None:
    st.info("Analyze a capture on the home page first.")
else:
    classical = report["stages"].get("classical_modulation", {})
    prediction = report["stages"].get("cnn_modulation", {})
    fusion = report["stages"].get("fusion", {})
    left, right = st.columns(2)
    with left:
        st.subheader("Classical cross-check")
        st.metric("Family", classical.get("family", "unknown"))
        st.metric("Classical family confidence", f"{classical.get('confidence', 0):.1%}")
    with right:
        st.subheader("Fused decision")
        st.metric("Decision", fusion.get("label", "Unclassified"))
        st.metric("Trust score", f"{fusion.get('trust_score', 0):.1%}")
        st.caption("Trust score is a fused value, not the CNN probability.")
        st.metric("CNN top-1 confidence", f"{prediction.get('confidence', 0):.1%}")
    if prediction.get("available"):
        st.subheader("CNN top predictions")
        st.caption("CNN confidence is the model softmax probability for each hypothesis.")
        top_k = prediction.get("top_k", [])
        st.bar_chart({label: probability for label, probability in top_k})
        st.table([{"modulation": label, "confidence": f"{probability:.1%}"} for label, probability in top_k])
    else:
        st.warning(prediction.get("message", "CNN inference is unavailable."))
    if fusion.get("review_recommended"):
        st.warning("Review recommended: the independent estimates are low-confidence or disagree.")
    st.caption(f"Classical family: {fusion.get('classical_family', 'unknown')} | alternatives: {', '.join(fusion.get('alternatives', [])) or 'none'}")
    demodulation = report["stages"].get("demodulation", {})
    symbols = demodulation.get("result", {}).get("symbols", []) if demodulation.get("available") else []
    if symbols:
        st.caption("The 2D view shows symbol locations; the 3D view adds symbol order to reveal phase rotation, drift, and transient clusters.")
        points = np.asarray([
            complex(item.get("real", 0.0), item.get("imag", 0.0)) if isinstance(item, dict) else complex(float(item), 0.0)
            for item in symbols
        ], dtype=np.complex64)
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.scatter(points.real, points.imag, s=8, alpha=0.6)
        axis.set_xlabel("In-phase")
        axis.set_ylabel("Quadrature")
        axis.set_title("Recovered constellation")
        axis.grid(alpha=0.25)
        st.pyplot(figure, clear_figure=True)
        trajectory = go.Figure(go.Scatter3d(x=points.real, y=points.imag, z=np.arange(points.size), mode="markers", marker={"size": 3, "color": np.arange(points.size), "colorscale": "Turbo", "opacity": 0.75}))
        trajectory.update_layout(height=500, margin={"l": 0, "r": 0, "t": 30, "b": 0}, scene={"xaxis_title": "In-phase", "yaxis_title": "Quadrature", "zaxis_title": "Symbol index"})
        st.plotly_chart(trajectory, width="stretch")
    for bucket in ("low", "mid", "high"):
        plot_path = Path(f"reports/confusion_{bucket}.png")
        if plot_path.exists():
            st.image(str(plot_path), caption=f"{bucket.title()} SNR confusion matrix")
