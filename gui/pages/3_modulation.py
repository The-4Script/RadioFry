import streamlit as st
from pathlib import Path

st.header("Modulation")
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
        st.metric("Confidence", f"{classical.get('confidence', 0):.1%}")
    with right:
        st.subheader("Fused decision")
        st.metric("Decision", fusion.get("label", "Unclassified"))
        st.metric("Trust score", f"{fusion.get('trust_score', 0):.1%}")
    if prediction.get("available"):
        st.subheader("CNN top predictions")
        top_k = prediction.get("top_k", [])
        st.bar_chart({label: probability for label, probability in top_k})
    else:
        st.warning(prediction.get("message", "CNN inference is unavailable."))
    if fusion.get("review_recommended"):
        st.warning("Review recommended: the independent estimates are low-confidence or disagree.")
    for bucket in ("low", "mid", "high"):
        plot_path = Path(f"reports/confusion_{bucket}.png")
        if plot_path.exists():
            st.image(str(plot_path), caption=f"{bucket.title()} SNR confusion matrix")
