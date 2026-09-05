import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.signal import stft

st.header("Upload and Preview")
signal = st.session_state.get("signal")
if signal is None:
    st.info("Upload a WAV or IQ capture on the home page.")
else:
    st.write({"format": signal.source_format, "samples": signal.iq.size, "duration_sec": signal.duration_sec})
    preview = signal.iq[: min(signal.iq.size, 5000)]
    left, right = st.columns(2)
    with left:
        st.subheader("I/Q waveform")
        st.caption("Amplitude versus sample index. DC removal and power normalization are applied before downstream analysis.")
        st.line_chart({"I": preview.real, "Q": preview.imag})
    with right:
        st.subheader("Waterfall")
        st.caption("Time-frequency energy view. Bright bands indicate concentrated spectral energy, not a modulation verdict by themselves.")
        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.specgram(preview, NFFT=min(256, max(32, preview.size // 4)), Fs=signal.sample_rate or 1.0, noverlap=32, cmap="viridis")
        axis.set_xlabel("Time")
        axis.set_ylabel("Frequency")
        st.pyplot(figure, clear_figure=True)

    st.subheader("3D spectral surface")
    st.caption("Interactive frequency-power landscape across time. Use rotation and zoom to inspect transient emitters, drift, and occupied bands.")
    surface_samples = signal.iq[: min(signal.iq.size, 12000)]
    segment_length = min(256, max(32, surface_samples.size // 8))
    frequencies, times, spectrum = stft(
        surface_samples,
        fs=signal.sample_rate or 1.0,
        nperseg=segment_length,
        noverlap=min(32, segment_length - 1),
        return_onesided=False,
    )
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    power_db = 10 * np.log10(np.maximum(np.abs(spectrum[order]) ** 2, 1e-12))
    frequency_step = max(1, power_db.shape[0] // 80)
    time_step = max(1, power_db.shape[1] // 100)
    figure = go.Figure(go.Surface(x=times[::time_step], y=frequencies[::frequency_step], z=power_db[::frequency_step, ::time_step], colorscale="Viridis"))
    figure.update_layout(height=520, margin={"l": 0, "r": 0, "t": 30, "b": 0}, scene={"xaxis_title": "Time", "yaxis_title": "Frequency", "zaxis_title": "Power (dB)"})
    st.plotly_chart(figure, use_container_width=True)
