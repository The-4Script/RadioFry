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
        suffix = Path(uploaded.name).suffix.lower()
        temporary_path = Path.cwd() / f".radiofry_upload{suffix}"
        temporary_path.write_bytes(uploaded.getvalue())
        st.session_state["upload_path"] = str(temporary_path)
        st.session_state["upload_name"] = uploaded.name
    with st.expander("Expert analysis controls", expanded=True):
        symbol_rate_input = st.number_input("Symbol rate override (Hz, optional)", min_value=0.0, value=0.0)
        interleaver_choice = st.selectbox("Interleaver handling", ["Auto", "None", "Block", "Convolutional", "Diagonal", "Pseudo-random"])
        fec_choice = st.selectbox("FEC handling", ["Auto", "None", "Convolutional", "Reed-Solomon", "Concatenated", "LDPC"])
    interleaver_override = interleaver_choice.lower().replace("-", "_") if interleaver_choice != "Auto" else None
    fec_override = fec_choice.lower().replace("-", "_") if fec_choice != "Auto" else None

    if uploaded.name.lower().endswith(".wav"):
        sample_rate = None
        iq_format = None
        analyze = st.button("Analyze capture", type="primary")
    else:
        with st.form("iq_options"):
            sample_rate = st.number_input("IQ sample rate (Hz, optional)", min_value=0.0, value=0.0)
            dtype = st.selectbox("IQ dtype", ["int16", "float32"])
            byte_order = st.selectbox("IQ byte order", ["little", "big"])
            analyze = st.form_submit_button("Analyze capture", type="primary")
        iq_format = IQFormat(dtype, byte_order)

    if analyze:
        try:
            signal = load_capture(temporary_path, sample_rate=sample_rate or None, iq_format=iq_format)
            report = analyze_capture(
                signal,
                symbol_rate_override=symbol_rate_input or None,
                interleaver_override=interleaver_override,
                fec_override=fec_override,
            )
            st.session_state["report"] = report
            st.session_state["signal"] = signal
            st.session_state["upload_analysis_key"] = uploaded.name
        except (OSError, ValueError) as error:
            st.error(str(error))

    signal = st.session_state.get("signal")
    report = st.session_state.get("report")
    if signal is not None and report is not None:
        columns = st.columns(2)
        columns[0].metric("Samples", f"{signal.iq.size:,}")
        columns[1].metric("Sample rate", signal.sample_rate or "Unknown")
        st.download_button("Download JSON report", report_json(report), "radiofry-report.json", "application/json")
