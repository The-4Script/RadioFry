import streamlit as st

from gui.theme import apply_radiofry_theme, render_bit_preview, render_empty_state, render_method_note, render_stage_header

apply_radiofry_theme()
report = st.session_state.get("report")
correlation = report.get("stages", {}).get("bitstream_analysis", {}).get("correlation") if report else None
render_stage_header("07", "Bitstream correlation", "Look for repeated markers and likely frame boundaries in the recovered bitstream.", "Available" if correlation else "Waiting", "ready" if correlation else "muted")

if report is None:
    render_empty_state("Analyze a capture on the home page first.")
elif not correlation:
    render_empty_state("Correlation becomes available after demodulation produces bits.")
else:
    st.caption("Correlation is framing evidence. It does not identify payload semantics or a network protocol by itself.")
    left, mid, right = st.columns(3)
    left.metric("Sync pattern", correlation.get("sync_pattern") or "Not detected")
    mid.metric("Detected positions", len(correlation.get("sync_positions", [])))
    right.metric("Estimated period", correlation.get("period") or "Unknown")
    metrics = st.columns(2)
    metrics[0].metric("Autocorrelation peak", f"{correlation.get('autocorrelation_peak', 0):.3f}")
    metrics[1].metric("Sync match score", f"{correlation.get('sync_match_score', 0):.1%}")
    headers, payload = st.columns(2)
    with headers:
        render_bit_preview(correlation.get("header_bits", []), "Header candidate", 256)
    with payload:
        render_bit_preview(correlation.get("payload_bits", []), "Payload candidate", 256)
    render_method_note("Method and limits", "The correlation stage searches for periodicity and known synchronization patterns. Header and payload labels are hypotheses that need protocol context for confirmation.")
