import streamlit as st

from gui.theme import apply_radiofry_theme, render_section_explainer

apply_radiofry_theme()

st.header("Bitstream Correlation")
render_section_explainer(
    "Stage 7: look for the signal’s hidden timing rhythm",
    "A digital transmission often has repeating markers, like a clock or a start-of-message pattern. Correlation helps find those repeating structures and estimate where a frame or header might begin.",
    "This stage looks for periodicity and known sync patterns. It is a clue-finding step, not proof of a network protocol. It helps explain how the data is framed and where the meaningful information likely starts.",
    "When it finds a pattern, the app highlights likely header and payload regions to support downstream interpretation and reporting.",
)

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
