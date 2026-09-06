import streamlit as st
from pathlib import Path

from gui.theme import render_bit_preview, render_empty_state, render_method_note, render_page_shell, render_stage_header
from gui.model_metrics import load_model_metrics

render_page_shell(6)
report = st.session_state.get("report")
analysis = report.get("stages", {}).get("bitstream_analysis", {}) if report else {}
prediction = analysis.get("fec") or {}
result = analysis.get("fec_decoding") or {}
ready = bool(prediction)
success = bool(result.get("success"))
render_stage_header("06", "FEC decoding", "Separate scheme classification from the harder question: whether damaged bits were actually repaired.", "Recovered" if success else ("Attempted" if ready else "Waiting"), "ready" if success else ("review" if ready else "muted"))

if report is None:
    render_empty_state("Analyze a capture on the home page first.")
elif not ready:
    render_empty_state(analysis.get("message", "Provide demodulated bits to run FEC classification."))
else:
    left, mid, right = st.columns(3)
    left.metric("Predicted scheme", prediction.get("label", "Unknown"))
    mid.metric("Confidence", f"{prediction.get('confidence', 0):.1%}")
    right.metric("Decoded bits", f"{len(result.get('bits', [])):,}")
    metrics = load_model_metrics(Path(__file__).resolve().parents[2] / "models_saved" / "fec_classifier.pkl")
    if metrics:
        st.caption(f"Measured synthetic-data test accuracy: {metrics.get('test_accuracy', 0):.1%}; 5-fold CV: {metrics.get('cv_accuracy_mean', 0):.1%} +/- {metrics.get('cv_accuracy_std', 0):.1%}. This does not estimate real-signal accuracy.")
    st.caption(f"Selected scheme: {analysis.get('selected_fec', 'unknown')}")
    if not prediction.get("available", True):
        st.warning(prediction.get("message", "FEC classifier unavailable."))
    if success:
        st.success("The selected decoder returned a successful result.")
    else:
        st.warning(result.get("message", "The decoder did not establish a successful recovery."))
    render_bit_preview(result.get("bits", []), "Decoded bitstream")
    render_method_note("Method and limits", "A predicted FEC scheme is not the same as successful decoding. Convolutional, Reed-Solomon, and concatenated paths require compatible parameters; LDPC is classification-only until code metadata is supplied.")
