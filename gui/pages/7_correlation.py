import streamlit as st

st.header("Bitstream Correlation")
report = st.session_state.get("report")
correlation = report.get("stages", {}).get("bitstream_analysis", {}).get("correlation") if report else None
if correlation:
	st.metric("Sync pattern", correlation.get("sync_pattern") or "Not detected")
	st.metric("Detected positions", len(correlation.get("sync_positions", [])))
	st.metric("Estimated period", correlation.get("period") or "Unknown")
else:
	st.info("Correlation becomes available after demodulation produces bits.")
