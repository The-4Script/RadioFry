import streamlit as st

st.header("Report")
report = st.session_state.get("report")
if report is None:
    st.info("Analyze a capture on the home page first.")
else:
    st.json(report)