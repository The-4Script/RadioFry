from pathlib import Path

import streamlit as st

from gui.theme import render_empty_state, render_method_note, render_page_shell, render_stage_header

render_page_shell(2)
report = st.session_state.get("report")
status = "Ready" if report is not None else "Waiting"
status_kind = "ready" if report is not None else "muted"
render_stage_header("02", "Signal parameters", "Measure the recording before interpreting it: occupied bandwidth, noise level, and timing cues.", status, status_kind)

if report is None:
    render_empty_state("Analyze a capture on the home page first.", "Parameter estimates are derived from the processed signal.")
else:
    parameters = report["stages"].get("parameters", {})
    source = report.get("source", {})
    st.caption(f"{source.get('format', 'unknown').upper()} · {source.get('samples', 0):,} samples · sample rate: {source.get('sample_rate') or 'unknown'}")
    columns = st.columns(3)
    columns[0].metric("Occupied bandwidth", f"{parameters.get('occupied_bandwidth_hz') or 0:,.1f} Hz" if parameters.get("occupied_bandwidth_hz") is not None else "Unknown")
    columns[1].metric("Estimated SNR", f"{parameters.get('snr_db'):.1f} dB" if parameters.get("snr_db") is not None else "Unknown")
    columns[2].metric("Symbol rate", f"{parameters.get('symbol_rate_hz'):.1f} Hz" if parameters.get("symbol_rate_hz") is not None else "Unknown")

    st.markdown("<div class='evidence-panel'><div class='evidence-label'>Interpretation</div><p>These values describe the recording conditions. They support later decisions but do not identify a protocol by themselves.</p></div>", unsafe_allow_html=True)
    with st.expander("All measured fields", expanded=False):
        st.json(parameters)
    render_method_note("Method and limits", f"Estimator method: {parameters.get('method', 'unknown')}. Absolute sample-rate and symbol-rate claims depend on reliable sensor metadata or a manual override.")
