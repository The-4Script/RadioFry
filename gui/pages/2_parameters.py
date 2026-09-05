import streamlit as st
from pathlib import Path

from gui.theme import apply_radiofry_theme, render_section_explainer

apply_radiofry_theme()


def _format_value(value: object, unit: str) -> str:
    return f"{value:,.1f} {unit}" if isinstance(value, (int, float)) else "Unknown"

st.header("Signal Parameters")
render_section_explainer(
    "Stage 2: estimate what the signal is doing",
    "This page answers a simple question: how fast is the signal, how much noise is around it, and how wide is the occupied bandwidth?",
    "The parameter-estimation stage extracts key channel statistics: occupied bandwidth, SNR, symbol timing, and method metadata. These values provide the foundation for modulation classification and downstream bitstream inspection.",
    "Check the measured values carefully. If the sample rate is uncertain, the app may use manual overrides or indicate that the estimate needs context before a strong interpretation is made.",
)

report = st.session_state.get("report")
if report is None:
    st.info("Analyze a capture on the home page first.")
else:
    parameters = report["stages"].get("parameters", {})
    source = report.get("source", {})
    st.caption(f"Source: {source.get('format', 'unknown')} | samples: {source.get('samples', 0):,} | measured sample rate: {source.get('sample_rate') or 'unknown'}")
    columns = st.columns(3)
    columns[0].metric("Occupied bandwidth", _format_value(parameters.get("occupied_bandwidth_hz"), "Hz"))
    columns[1].metric("Estimated SNR", _format_value(parameters.get("snr_db"), "dB"))
    columns[2].metric("Symbol rate", _format_value(parameters.get("symbol_rate_hz"), "Hz"))
    st.json(parameters)
    st.info(f"Estimation method: {parameters.get('method', 'unknown')}. Unknown absolute sample rates require a sensor header or manual input.")
    plot_path = Path("reports/accuracy_vs_snr.png")
    if plot_path.exists():
        st.image(str(plot_path), caption="Modulation accuracy by SNR")
