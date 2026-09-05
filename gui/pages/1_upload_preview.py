import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import stft

from gui.theme import apply_radiofry_theme, render_empty_state, render_method_note, render_stage_header

apply_radiofry_theme()
render_stage_header("01", "Upload & preview", "Confirm that the capture is present, interpretable, and worth taking into the deeper analysis stages.")

signal = st.session_state.get("signal")
if signal is None:
    render_empty_state("Analyze a WAV or IQ capture on the home page first.", "This page becomes the visual sanity check before parameter estimation.")
else:
    duration = f"{signal.duration_sec:.2f} s" if signal.duration_sec is not None else "Unknown"
    left, mid, right = st.columns(3)
    left.metric("Format", signal.source_format.upper())
    mid.metric("Samples", f"{signal.iq.size:,}")
    right.metric("Duration", duration)

    preview = signal.iq[: min(signal.iq.size, 5000)]
    waveform, waterfall = st.columns(2)
    with waveform:
        st.markdown("<div class='evidence-panel'><h3>Waveform</h3><p style='color:var(--muted)'>I and Q amplitude over sample index.</p>", unsafe_allow_html=True)
        st.line_chart({"I": preview.real, "Q": preview.imag})
        st.markdown("</div>", unsafe_allow_html=True)
    with waterfall:
        st.markdown("<div class='evidence-panel'><h3>Waterfall</h3><p style='color:var(--muted)'>Where energy concentrates over time and frequency.</p>", unsafe_allow_html=True)
        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.specgram(preview, NFFT=min(256, max(32, preview.size // 4)), Fs=signal.sample_rate or 1.0, noverlap=32, cmap="viridis")
        axis.set_xlabel("Time")
        axis.set_ylabel("Frequency")
        st.pyplot(figure, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Inspect the 3D spectral surface", expanded=False):
        surface_samples = signal.iq[: min(signal.iq.size, 12000)]
        segment_length = min(256, max(32, surface_samples.size // 8))
        frequencies, times, spectrum = stft(surface_samples, fs=signal.sample_rate or 1.0, nperseg=segment_length, noverlap=min(32, segment_length - 1), return_onesided=False)
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        power_db = 10 * np.log10(np.maximum(np.abs(spectrum[order]) ** 2, 1e-12))
        frequency_step = max(1, power_db.shape[0] // 80)
        time_step = max(1, power_db.shape[1] // 100)
        figure = go.Figure(go.Surface(x=times[::time_step], y=frequencies[::frequency_step], z=power_db[::frequency_step, ::time_step], colorscale="Viridis"))
        figure.update_layout(height=520, margin={"l": 0, "r": 0, "t": 30, "b": 0}, scene={"xaxis_title": "Time", "yaxis_title": "Frequency", "zaxis_title": "Power (dB)"})
        st.plotly_chart(figure, width="stretch")

render_method_note("Method and limits", "The preview uses the first few thousand samples for responsiveness. It is a visual inspection aid, not a modulation decision. Mono WAV is converted to analytic IQ; stereo WAV is interpreted as I/Q.")
