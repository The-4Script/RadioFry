"""Streamlit entry point for the RadioFry analysis pipeline."""

from pathlib import Path
import tempfile

import streamlit as st

from gui.theme import apply_radiofry_theme, render_section_explainer
from radiofry.ingestion.iq_parser import IQFormat
from radiofry.pipeline import analyze_capture, load_capture
from radiofry.reporting.report_builder import build_pdf_report, report_json


st.set_page_config(page_title="RadioFry", page_icon="RF", layout="wide")
apply_radiofry_theme()

st.sidebar.markdown(
    """
    <div class='hero-panel'>
        <div class='highlight'>RF Intelligence Demo</div>
        <h3 style='margin-top: 0.9rem;'>RadioFry</h3>
        <p style='color: #dfeeff; margin-bottom: 0.3rem;'>Signal understanding for non-experts and experts alike.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.caption("Demo walkthrough")
for index, item in enumerate([
    "Upload a waveform or IQ capture",
    "Estimate signal parameters",
    "Classify modulation and confidence",
    "Inspect interleaver and FEC structure",
    "Generate a report for review",
], start=1):
    st.sidebar.markdown(f"<div class='info-pill'><b>{index}.</b> {item}</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='hero-panel'>
        <div class='highlight'>Public demo • expert review ready</div>
        <h1 style='margin: 0.7rem 0 0.4rem; font-size: 3rem;'>RadioFry</h1>
        <p style='font-size: 1.15rem; line-height: 1.6; color: #eaf2ff; margin-bottom: 0.75rem;'>
            RadioFry turns raw radio signals into understandable insights. It reads a capture, estimates how the signal is behaving,
            identifies the likely modulation, and checks whether the data has been scrambled or protected by error-correction.
        </p>
        <p style='margin-bottom: 0.2rem; color: #bdd7fb;'>
            In plain English: if a signal is noisy, distorted, or hidden inside a transmission format, RadioFry helps explain what it may be.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("What this system does")
left, center, right = st.columns(3)
with left:
    st.markdown("<div class='glass-card'><h4>For the public</h4><p>It helps us understand what a radio transmission might be saying, even when the raw data looks like noise.</p></div>", unsafe_allow_html=True)
with center:
    st.markdown("<div class='glass-card'><h4>For engineers</h4><p>It estimates bandwidth, symbol rate, modulation family, and bitstream structure using a multi-stage signal analysis pipeline.</p></div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='glass-card'><h4>For decision makers</h4><p>It turns technical evidence into a structured report so the signal can be reviewed, explained, and acted on.</p></div>", unsafe_allow_html=True)

render_section_explainer(
    "Judge walkthrough: how to explore this site",
    "This demo is designed like a guided story. You start by uploading a signal, then the app explains what it sees, what it thinks the signal is, and how confident it is. The goal is to make a complex radio-analysis pipeline understandable without losing technical depth.",
    "The pipeline performs four stages: ingest → parameter estimation → modulation classification/fusion → bitstream analysis (interleaver/FEC/correlation). Each stage produces evidence that supports or challenges the final interpretation and is captured in a JSON/PDF report.",
    "Start on the home page, upload a file, then follow the pages in order: Upload & Preview → Parameters → Modulation → Demodulation → Deinterleaving → FEC → Correlation → Report. Each page explains the current step and what should happen next.",
)

st.subheader("Demo flow")
step1, step2, step3, step4, step5 = st.columns(5)
for column, content in zip(
    [step1, step2, step3, step4, step5],
    [
        ("1", "Upload", "Choose a WAV or IQ capture and inspect the raw waveform."),
        ("2", "Measure", "Estimate SNR, bandwidth, and signal timing."),
        ("3", "Classify", "Use classical signal checks and a CNN to infer modulation."),
        ("4", "Decode", "Check the bitstream for interleaving and forward-error correction."),
        ("5", "Report", "Review a structured evidence package with expert summary."),
    ],
):
    with column:
        st.markdown(
            f"<div class='glass-card'><div class='step-badge'>{content[0]}</div><h4>{content[1]}</h4><p>{content[2]}</p></div>",
            unsafe_allow_html=True,
        )

st.subheader("How to use this demo")
use_columns = st.columns(2)
with use_columns[0]:
    st.markdown(
        """
        <div class='glass-card'>
            <h4>Layman explanation</h4>
            <p>Think of this as a digital detective for radio signals. It reads the raw data, looks for patterns, and tries to explain what kind of transmission it might be and whether the information was protected or scrambled.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with use_columns[1]:
    st.markdown(
        """
        <div class='glass-card'>
            <h4>Expert explanation</h4>
            <p>The app performs format-independent signal analysis from IQ samples, estimates key modulation and channel parameters, and combines independent evidence to produce a structured confidence-aware interpretation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info("For meaningful RF identification, use baseband I/Q from a documented SDR or a synthetic RF capture. Music and sonified space audio can test visualization, but cannot validate modulation, FEC, or interleaver claims.")

st.subheader("Upload and analyze")
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
        pdf_path = Path(tempfile.gettempdir()) / "radiofry-report.pdf"
        build_pdf_report(report, str(pdf_path), signal)
        st.download_button("Download formal PDF report", pdf_path.read_bytes(), "radiofry-report.pdf", "application/pdf")
        st.success("Analysis ready. Continue through the pages to review the evidence step by step.")
else:
    st.markdown(
        """
        <div class='glass-card'>
            <p>Upload a file to unlock the full demo. Once a signal is loaded, the system will walk through signal understanding, interpretation, and reporting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
