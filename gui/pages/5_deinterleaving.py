import streamlit as st
from pathlib import Path

from gui.theme import render_bit_preview, render_empty_state, render_method_note, render_page_shell, render_stage_header
from gui.model_metrics import load_model_metrics

render_page_shell(5)
report = st.session_state.get("report")
analysis = report.get("stages", {}).get("bitstream_analysis", {}) if report else {}
prediction = analysis.get("interleaver") or {}
ready = bool(prediction)
render_stage_header("05", "Deinterleaving", "Test whether recovered bits were rearranged before transmission and attempt to restore their order.", "Available" if ready else "Waiting", "ready" if ready else "muted")

if report is None:
    render_empty_state("Analyze a capture on the home page first.")
elif not ready:
    render_empty_state(analysis.get("message", "Provide demodulated bits to run interleaver classification."))
else:
    result = analysis.get("deinterleaving", {})
    left, right = st.columns(2)
    left.metric("Predicted type", prediction.get("label", "Unknown"))
    right.metric("Confidence", f"{prediction.get('confidence', 0):.1%}")
    metrics = load_model_metrics(Path(__file__).resolve().parents[2] / "models_saved" / "interleaver_classifier.pkl")
    if metrics:
        st.caption(f"Measured synthetic-data test accuracy: {metrics.get('test_accuracy', 0):.1%}; 5-fold CV: {metrics.get('cv_accuracy_mean', 0):.1%} +/- {metrics.get('cv_accuracy_std', 0):.1%}. This does not estimate real-signal accuracy.")
    st.caption(f"Selected method: {analysis.get('selected_interleaver', 'unknown')}")
    if not prediction.get("available", True):
        st.warning(prediction.get("message", "Interleaver classifier unavailable."))
    if result:
        columns = st.columns(2)
        columns[0].metric("Search score", f"{result.get('score', 0):.3f}")
        columns[1].metric("Parameters", str(result.get("parameters", {})) or "None")
        render_bit_preview(result.get("bits", []), "Deinterleaved preview")
        if result.get("limitation"):
            st.info(result["limitation"])
    render_method_note("Method and limits", "Block, convolutional, and diagonal searches are deterministic transforms. Pseudo-random deinterleaving requires a seed; a classifier label alone does not prove the original ordering.")
