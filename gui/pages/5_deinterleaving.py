import streamlit as st

st.header("De-interleaving")
report = st.session_state.get("report")
analysis = report.get("stages", {}).get("bitstream_analysis", {}) if report else {}
prediction = analysis.get("interleaver")
if prediction:
	st.metric("Predicted type", prediction.get("label", "unknown"))
	st.metric("Confidence", f"{prediction.get('confidence', 0):.1%}")
	result = analysis.get("deinterleaving", {})
	if result:
		st.json({"parameters": result.get("parameters", {}), "score": result.get("score", 0), "limitation": result.get("limitation")})
else:
	st.info(analysis.get("message", "Provide demodulated bits to run interleaver classification."))
