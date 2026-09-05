import streamlit as st

from gui.theme import apply_radiofry_theme, render_bit_preview, render_empty_state, render_method_note, render_stage_header

apply_radiofry_theme()
report = st.session_state.get("report")
demodulation = report.get("stages", {}).get("demodulation", {}) if report else {}
status = "Available" if demodulation.get("available") else "Needs review"
status_kind = "ready" if demodulation.get("available") else "review"
render_stage_header("04", "Demodulation", "Translate the waveform into symbols and bits using the current modulation hypothesis.", status, status_kind)

if report is None:
    render_empty_state("Analyze a capture on the home page first.")
elif not demodulation.get("available"):
    render_empty_state(demodulation.get("message", "Demodulation was not available."), "The pipeline avoids forcing a bitstream when the modulation decision is not trustworthy enough.")
else:
    result = demodulation.get("result", {})
    left, right = st.columns(2)
    left.metric("Modulation used", result.get("modulation", "Unknown"))
    right.metric("Recovered bits", f"{len(result.get('bits', [])):,}")
    render_bit_preview(result.get("bits", []), "Recovered bitstream")
    render_method_note("Method and limits", "Demodulation uses the selected modulation family and estimated signal parameters. Symbol timing, carrier recovery, and protocol-specific framing may still need expert calibration.")
