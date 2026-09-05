import streamlit as st

st.header("Upload and Preview")
signal = st.session_state.get("signal")
if signal is None:
    st.info("Upload a WAV or IQ capture on the home page.")
else:
    st.write({"format": signal.source_format, "samples": signal.iq.size, "duration_sec": signal.duration_sec})
    st.line_chart(signal.iq.real[: min(signal.iq.size, 5000)])
