import streamlit as st

st.header("FEC Decoding")
report = st.session_state.get("report")
analysis = report.get("stages", {}).get("bitstream_analysis", {}) if report else {}
prediction = analysis.get("fec")
if prediction:
	st.metric("Predicted scheme", prediction.get("label", "unknown"))
	st.metric("Confidence", f"{prediction.get('confidence', 0):.1%}")
	st.caption(f"Selected scheme: {analysis.get('selected_fec', prediction.get('label', 'unknown'))}")
	if not prediction.get("available", True):
		st.warning(prediction.get("message", "FEC classifier unavailable."))
	result = analysis.get("fec_decoding", {})
	if result:
		if result.get("success"):
			st.success("FEC decoding completed.")
		else:
			st.warning(result.get("message", "FEC decoding was degraded."))
		st.metric("Decoded bits", len(result.get("bits", [])))
		st.caption("Decoded payload preview")
		st.code("".join(str(int(bit)) for bit in result.get("bits", [])[:512]))
else:
	st.info(analysis.get("message", "Provide demodulated bits to run FEC classification."))
