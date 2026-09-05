import tempfile
from pathlib import Path

import streamlit as st

from gui.theme import apply_radiofry_theme, render_empty_state, render_method_note, render_stage_header
from radiofry.reporting.report_builder import build_pdf_report, report_json

apply_radiofry_theme()
report = st.session_state.get("report")
render_stage_header("08", "Report", "Package the findings, confidence, and limitations into an auditable handoff.", "Ready" if report else "Waiting", "ready" if report else "muted")

if report is None:
    render_empty_state("Analyze a capture on the home page first.")
else:
    source = report.get("source", {})
    stages = report.get("stages", {})
    fusion = stages.get("fusion", {})
    bitstream = stages.get("bitstream_analysis", {})
    st.markdown("<div class='evidence-panel'><div class='evidence-label'>Executive readout</div><h3>{}</h3><p>Trust score: {:.1%}. {}.</p></div>".format(fusion.get("label", "Unclassified"), fusion.get("trust_score", 0), "Human review is recommended" if fusion.get("review_recommended") else "The fused decision is available for inspection"), unsafe_allow_html=True)
    left, mid, right = st.columns(3)
    left.metric("Source", str(source.get("format", "unknown")).upper())
    mid.metric("Samples", f"{source.get('samples', 0):,}")
    right.metric("Sample rate", source.get("sample_rate") or "Unknown")

    st.subheader("Stage status")
    status_rows = []
    for label, value in [
        ("Runtime artifacts", stages.get("runtime", {}).get("ready", False)),
        ("Modulation inference", stages.get("cnn_modulation", {}).get("available", False)),
        ("Demodulation", stages.get("demodulation", {}).get("available", False)),
        ("Bitstream analysis", bitstream.get("available", False)),
        ("FEC recovery", bitstream.get("fec_decoding", {}).get("success", False)),
        ("Correlation", "correlation" in bitstream),
    ]:
        status_rows.append({"stage": label, "status": "Available" if value else "Needs review"})
    st.table(status_rows)

    limitations = [
        stages.get("cnn_modulation", {}).get("message"),
        stages.get("demodulation", {}).get("message"),
        bitstream.get("deinterleaving", {}).get("limitation"),
        bitstream.get("fec_decoding", {}).get("message"),
    ]
    limitations = [item for item in limitations if item]
    if limitations:
        st.subheader("Review notes")
        for item in limitations:
            st.warning(item)

    downloads = st.columns(2)
    with downloads[0]:
        st.download_button("Download JSON", report_json(report), "radiofry-report.json", "application/json", use_container_width=True)
    with downloads[1]:
        signal = st.session_state.get("signal")
        pdf_path = Path(tempfile.gettempdir()) / "radiofry-report.pdf"
        build_pdf_report(report, str(pdf_path), signal)
        st.download_button("Download PDF", pdf_path.read_bytes(), "radiofry-report.pdf", "application/pdf", use_container_width=True)

    with st.expander("Inspect raw report JSON", expanded=False):
        st.json(report)
    render_method_note("What this report means", "The report preserves the pipeline's evidence and uncertainty. A label, confidence value, or recovered bit sequence should be interpreted alongside its assumptions and review notes.")
