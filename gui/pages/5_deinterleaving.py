import streamlit as st

from gui.theme import apply_radiofry_theme, render_section_explainer

apply_radiofry_theme()

st.header("De-interleaving")
render_section_explainer(
    "Stage 5: untangle the bit stream",
    "Some transmissions deliberately rearrange their bits before sending them. This can make the message harder to read without the right de-interleaving method.",
    "The classifier estimates whether the data was scrambled using a block, convolutional, diagonal, or pseudo-random pattern. Once selected, it attempts to reverse that rearrangement so the payload can be inspected more meaningfully.",
    "If the arrangement is too irregular or intentionally random, the system records that limitation and avoids pretending it solved something it did not.",
)

report = st.session_state.get("report")
analysis = report.get("stages", {}).get("bitstream_analysis", {}) if report else {}
prediction = analysis.get("interleaver")
if prediction:
	st.metric("Predicted type", prediction.get("label", "unknown"))
	st.metric("Confidence", f"{prediction.get('confidence', 0):.1%}")
	st.caption(f"Selected type: {analysis.get('selected_interleaver', prediction.get('label', 'unknown'))}")
	if not prediction.get("available", True):
		st.warning(prediction.get("message", "Interleaver classifier unavailable."))
	result = analysis.get("deinterleaving", {})
	if result:
		st.json({"parameters": result.get("parameters", {}), "score": result.get("score", 0), "limitation": result.get("limitation")})
		st.caption("De-interleaved preview")
		st.code("".join(str(int(bit)) for bit in result.get("bits", [])[:512]))
else:
	st.info(analysis.get("message", "Provide demodulated bits to run interleaver classification."))
