"""Streamlit entry point for the RadioFry analysis pipeline."""

from pathlib import Path

import streamlit as st

from radiofry.ingestion.iq_parser import IQFormat
from radiofry.pipeline import analyze_capture, load_capture
from radiofry.reporting.report_builder import build_report, report_json


st.set_page_config(page_title="RadioFry", page_icon="RF", layout="wide")
st.title("RadioFry")
st.caption("Hardware-agnostic RF signal analysis")

uploaded = st.file_uploader("Upload a WAV or raw IQ capture", type=["wav", "iq"])
if uploaded is not None:
    temporary_path = Path(st.session_state.get("upload_path", ""))
    if not temporary_path.exists() or st.session_state.get("upload_name") != uploaded.name:
        temporary_path = Path.cwd() / ".radiofry_upload"
        temporary_path.write_bytes(uploaded.getvalue())
        st.session_state["upload_path"] = str(temporary_path)
        st.session_state["upload_name"] = uploaded.name
    try:
        if uploaded.name.lower().endswith(".wav"):
            signal = load_capture(temporary_path)
        else:
            sample_rate = st.number_input("IQ sample rate (Hz)", min_value=0.0, value=0.0)
            signal = load_capture(
                temporary_path, sample_rate=sample_rate or None,
                iq_format=IQFormat(st.selectbox("IQ dtype", ["int16", "float32"])),
            )
        report = analyze_capture(signal)
        st.session_state["report"] = report
        st.session_state["signal"] = signal
        st.metric("Samples", f"{signal.iq.size:,}")
        st.metric("Sample rate", signal.sample_rate or "Unknown")
        st.download_button("Download JSON report", report_json(report), "radiofry-report.json", "application/json")
    except (OSError, ValueError) as error:
        st.error(str(error))
