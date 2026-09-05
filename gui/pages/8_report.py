import streamlit as st

from gui.theme import apply_radiofry_theme, render_section_explainer
from radiofry.reporting.report_builder import report_json

apply_radiofry_theme()

st.header("Report")
render_section_explainer(
    "Stage 8: package the evidence for review",
    "This final page turns the technical journey into a clean, shareable record. It preserves the numerical results, confidence measurements, and stage-by-stage evidence in a structured report.",
    "The report is useful for both decision makers and domain experts. It is not just a printout; it is a traceable evidence package that explains the signal analysis pipeline and its findings in a reusable format.",
    "Download the JSON or PDF report when you want to share the result outside the app or keep it as an audit trail for a review.",
)

report = st.session_state.get("report")
if report is None:
    st.info("Analyze a capture on the home page first.")
else:
    st.json(report)
    st.download_button("Download JSON report", report_json(report), "radiofry-report.json", "application/json")