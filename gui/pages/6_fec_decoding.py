import streamlit as st

from gui.theme import apply_radiofry_theme, render_section_explainer

apply_radiofry_theme()

st.header("FEC Decoding")
render_section_explainer(
    "Stage 6: recover meaning from damaged transmissions",
    "Forward error correction is like adding a safety net to digital data. If the signal gets noisy or bits get corrupted, this layer tries to repair the damage before the payload is used.",
    "The app classifies likely FEC schemes, such as convolutional or Reed–Solomon, then attempts error correction and reports whether the payload was successfully repaired or whether the signal still needs review.",
    "This is one of the most important checks when a transmission is partly degraded, because it explains whether a message is recoverable or simply too damaged to trust.",
)

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
