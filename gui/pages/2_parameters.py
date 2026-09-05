import streamlit as st
from pathlib import Path

st.header("Signal Parameters")
report = st.session_state.get("report")
if report is None:
    st.info("Analyze a capture on the home page first.")
else:
    parameters = report["stages"].get("parameters", {})
    columns = st.columns(3)
    columns[0].metric("Occupied bandwidth", f"{parameters.get('occupied_bandwidth_hz', 0):,.1f} Hz")
    columns[1].metric("Estimated SNR", f"{parameters.get('snr_db', 0):,.1f} dB")
    columns[2].metric("Symbol rate", f"{parameters.get('symbol_rate_hz', 0):,.1f} Hz")
    st.json(parameters)
    plot_path = Path("reports/accuracy_vs_snr.png")
    if plot_path.exists():
        st.image(str(plot_path), caption="Modulation accuracy by SNR")
