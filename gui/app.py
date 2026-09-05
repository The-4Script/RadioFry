"""Streamlit entry point for the RadioFry analysis pipeline."""

from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from gui.theme import apply_radiofry_theme, render_method_note, render_sidebar, render_tutorial
from radiofry.ingestion.iq_parser import IQFormat
from radiofry.pipeline import DEFAULT_MAX_CAPTURE_BYTES, analyze_capture, load_capture
from radiofry.reporting.report_builder import build_pdf_report, report_json


st.set_page_config(page_title="RadioFry", page_icon="RF", layout="wide")
st.session_state.setdefault("show_tutorial", True)
apply_radiofry_theme()
render_sidebar(active_stage=0, has_analysis="report" in st.session_state)

st.markdown(
    """
    <div class="hero-panel">
        <div class="brand-mark">SIH26147 / Signal analysis workbench</div>
        <h1 style="margin: 0.7rem 0 0.45rem;">RadioFry</h1>
        <p style="font-size: 1.08rem; line-height: 1.55; margin-bottom: 0.35rem;">
            A transparent workflow for turning WAV and raw IQ captures into measurable signal evidence.
        </p>
        <p style="color: #c9d8d2; margin: 0;">Upload a capture, inspect the evidence, and decide where expert review is still needed.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_tutorial()

st.subheader("Start an analysis")
st.caption("Use a short baseband recording from an SDR or a documented synthetic capture. Music and sonified audio are useful only for testing the visual layer.")

uploaded = st.file_uploader("Choose a WAV or raw IQ capture", type=["wav", "iq"], label_visibility="visible")
if uploaded is not None:
    if uploaded.size > DEFAULT_MAX_CAPTURE_BYTES:
        st.error(f"Capture exceeds the {DEFAULT_MAX_CAPTURE_BYTES // (1024 * 1024)} MB upload limit.")
        st.stop()

    upload_key = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_key") != upload_key:
        previous_path = Path(st.session_state.get("upload_path", ""))
        if previous_path.is_file():
            previous_path.unlink(missing_ok=True)
        st.session_state.pop("report", None)
        st.session_state.pop("signal", None)
        st.session_state["upload_key"] = upload_key

    temporary_path = Path(st.session_state.get("upload_path", ""))
    if not temporary_path.is_file() or st.session_state.get("upload_name") != uploaded.name:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(prefix="radiofry-", suffix=suffix, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        temporary_path.write_bytes(uploaded.getvalue())
        st.session_state["upload_path"] = str(temporary_path)
        st.session_state["upload_name"] = uploaded.name

    with st.expander("Analysis assumptions", expanded=True):
        control_left, control_right = st.columns(2)
        with control_left:
            symbol_rate_input = st.number_input("Symbol rate override (Hz)", min_value=0.0, value=0.0, help="Leave at zero to use the estimator.")
            interleaver_choice = st.selectbox("Interleaver", ["Auto", "None", "Block", "Convolutional", "Diagonal", "Pseudo-random"])
        with control_right:
            fec_choice = st.selectbox("FEC", ["Auto", "None", "Convolutional", "Reed-Solomon", "Concatenated", "LDPC"])
            st.caption("Overrides are useful when sensor metadata or protocol documentation is available.")

    interleaver_override = interleaver_choice.lower().replace("-", "_") if interleaver_choice != "Auto" else None
    fec_override = fec_choice.lower().replace("-", "_") if fec_choice != "Auto" else None

    if uploaded.name.lower().endswith(".wav"):
        sample_rate = None
        iq_format = None
        analyze = st.button("Analyze capture", type="primary", use_container_width=True)
    else:
        with st.form("iq_options"):
            st.markdown("**Raw IQ interpretation**")
            iq_left, iq_right = st.columns(2)
            with iq_left:
                sample_rate = st.number_input("Sample rate (Hz)", min_value=0.0, value=0.0)
                dtype = st.selectbox("Value type", ["int16", "float32"])
            with iq_right:
                byte_order = st.selectbox("Byte order", ["little", "big"])
                st.caption("Raw IQ files have no header, so these values must come from the sensor or recording notes.")
            analyze = st.form_submit_button("Analyze capture", type="primary", use_container_width=True)
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
            st.rerun()
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            st.error(str(error))

    signal = st.session_state.get("signal")
    report = st.session_state.get("report")
    if signal is not None and report is not None:
        st.success("Analysis complete. Use the stages in the sidebar to inspect the evidence.")
        summary_left, summary_mid, summary_right = st.columns(3)
        summary_left.metric("Samples", f"{signal.iq.size:,}")
        summary_mid.metric("Sample rate", signal.sample_rate or "Unknown")
        summary_right.metric("Source", signal.source_format.upper())
        download_left, download_right = st.columns(2)
        with download_left:
            st.download_button("Download JSON", report_json(report), "radiofry-report.json", "application/json", use_container_width=True)
        with download_right:
            pdf_path = Path(tempfile.gettempdir()) / "radiofry-report.pdf"
            build_pdf_report(report, str(pdf_path), signal)
            st.download_button("Download PDF", pdf_path.read_bytes(), "radiofry-report.pdf", "application/pdf", use_container_width=True)
else:
    st.markdown("<div class='evidence-panel'><div class='evidence-label'>No capture selected</div><p>Choose a file above to unlock the evidence stages.</p></div>", unsafe_allow_html=True)

render_method_note(
    "What RadioFry can and cannot claim",
    "RadioFry estimates signal parameters, compares modulation evidence, and attempts bitstream structure recovery. Confidence values indicate model or method agreement; they are not proof of protocol identity. LDPC, pseudo-random interleaving, unknown symbol timing, and undocumented FEC parameters may require expert input.",
)
