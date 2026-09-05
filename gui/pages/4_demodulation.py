import streamlit as st

from gui.theme import apply_radiofry_theme, render_section_explainer

apply_radiofry_theme()

st.header("Demodulation")
render_section_explainer(
    "Stage 4: convert signal symbols into usable data",
    "Once the system knows the likely modulation, it tries to turn the symbol cloud into actual bit data. This is the point where the raw signal becomes a sequence of ones and zeros that can be checked for structure.",
    "The demodulation stage produces recovered symbol and bit data when the classification is strong enough. It is not an arbitrary guess; it is a structured attempt to recover the original digital message from the waveform.",
    "If the signal is too noisy or out of distribution, the system will stop early and explain why instead of forcing a false result.",
)

report = st.session_state.get("report")
demodulation = report.get("stages", {}).get("demodulation", {}) if report else {}
if demodulation.get("available"):
	result = demodulation.get("result", {})
	st.metric("Modulation", result.get("modulation", "unknown"))
	st.metric("Recovered bits", len(result.get("bits", [])))
	st.caption(demodulation.get("message", ""))
else:
	st.info(demodulation.get("message", "Analyze a capture on the home page first."))

if demodulation.get("available"):
	result = demodulation.get("result", {})
	bits = result.get("bits", [])
	st.subheader("Recovered bitstream")
	st.code("".join(str(int(bit)) for bit in bits[:512]) or "No bits recovered")
