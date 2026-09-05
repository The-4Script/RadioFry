import streamlit as st

st.header("Demodulation")
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
