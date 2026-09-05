import streamlit as st

st.header("FEC Decoding")
report = st.session_state.get("report")
analysis = report.get("stages", {}).get("bitstream_analysis", {}) if report else {}
prediction = analysis.get("fec")
if prediction:
	st.metric("Predicted scheme", prediction.get("label", "unknown"))
	st.metric("Confidence", f"{prediction.get('confidence', 0):.1%}")
	result = analysis.get("fec_decoding", {})
	if result:
		if result.get("success"):
			st.success("FEC decoding completed.")
		else:
			st.warning(result.get("message", "FEC decoding was degraded."))
		st.metric("Decoded bits", len(result.get("bits", [])))
else:
	st.info(analysis.get("message", "Provide demodulated bits to run FEC classification."))
