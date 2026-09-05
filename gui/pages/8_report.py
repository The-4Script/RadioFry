import streamlit as st
from radiofry.reporting.report_builder import report_json

st.header("Report")
report = st.session_state.get("report")
if report is None:
    st.info("Analyze a capture on the home page first.")
else:
    st.json(report)
    st.download_button("Download JSON report", report_json(report), "radiofry-report.json", "application/json")