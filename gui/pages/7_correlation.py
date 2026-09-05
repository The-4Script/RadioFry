import streamlit as st

st.header("Bitstream Correlation")
report = st.session_state.get("report")
correlation = report.get("stages", {}).get("bitstream_analysis", {}).get("correlation") if report else None
if correlation:
	st.caption("Correlation evidence is a hypothesis about synchronization and framing. It does not identify payload semantics without protocol knowledge.")
	st.metric("Sync pattern", correlation.get("sync_pattern") or "Not detected")
	st.metric("Detected positions", len(correlation.get("sync_positions", [])))
	st.metric("Estimated period", correlation.get("period") or "Unknown")
	st.metric("Autocorrelation peak", f"{correlation.get('autocorrelation_peak', 0):.3f}")
	st.metric("Sync match score", f"{correlation.get('sync_match_score', 0):.1%}")
	left, right = st.columns(2)
	with left:
		st.caption("Header bits")
		st.code("".join(str(int(bit)) for bit in correlation.get("header_bits", [])[:256]) or "Not detected")
	with right:
		st.caption("Payload bits")
		st.code("".join(str(int(bit)) for bit in correlation.get("payload_bits", [])[:256]) or "Not detected")
else:
	st.info("Correlation becomes available after demodulation produces bits.")
