# RadioFry UI/UX Current State Audit

*Date: September 2026*
*Target: Current RadioFry Repository*

This document provides an exact, factual mapping of the current RadioFry GUI implementation. It is an inspection output intended to brief subsequent UI/UX engineering efforts. It does not propose changes, fixes, or refactoring.

## 1. Repository Structure & Entry Point

- **Framework**: Streamlit
- **Entry Point**: `gui/app.py`
  - Initializes `st.set_page_config` (Title: "RadioFry", Icon: "🍟", Layout: "wide", Initial Sidebar: "collapsed").
  - Imports and applies the central theme via `radiofry.gui.theme.apply_radiofry_theme()`.
  - Serves as the ingest/trigger point ("Upload Signal", "Analyze capture" button).
  - Handles initial WAV/IQ ingestion, triggering `analyze_capture(signal)`, and storing the resulting `report` and `signal` in `st.session_state`.
- **Page Directory**: `gui/pages/` containing 13 individual Streamlit page files.

## 2. Design System & Theming

- **Theme Manager**: `gui/theme.py`
  - Uses CSS injection (`st.markdown("<style>...</style>")`) to override default Streamlit aesthetics.
  - Defines root CSS variables: `var(--bg)`, `var(--panel-bg)`, `var(--text)`, `var(--muted)`, `var(--accent)`, `var(--accent-hover)`, `var(--border)`, `var(--success)`, `var(--warning)`, `var(--danger)`.
  - Provides a dark theme by default.
  - Hides native Streamlit UI elements (header, footer, menu, deploy button).
  - Implements custom CSS classes (e.g., `.evidence-panel`, `.evidence-label`, `.stage-kicker`, `.stage-title`, `.hero-panel`, `.metric-value`, `.metric-label`).
- **Shared Layout Components**:
  - `render_page_shell(stage_number)`: Standard wrapper required at the top of every page. It applies the theme, renders the sidebar, and extracts pipeline data from session state.
  - `render_stage_header(title, subtitle, icon, is_active, stage_number)`: Uniform header styling for pages.
  - `render_sidebar()`: Custom HTML/CSS sidebar navigation bypassing Streamlit's default sidebar. Uses `st.page_link` for navigation and falls back to Markdown links.
  - `render_method_note(title, text)`: Shared component for rendering methodological caveats at the bottom of pages.

## 3. Session State & Pipeline Integration

The GUI is entirely state-driven based on the `st.session_state` dictionary:
- `st.session_state["signal"]`: Holds the ingested signal object (e.g., `MemoryMappedWavSignal`).
- `st.session_state["report"]`: A dictionary output from the `analyze_capture` pipeline containing results from all stages.
- `st.session_state["parameters"]`: Extracted from `report["stages"]["parameters"]["result"]` in `app.py`.
- **Signal Time Machine State Variables** (`9_time_machine.py`):
  - `stm_playing` (bool): Playback active state.
  - `stm_frozen` (bool): Playback freeze state.
  - `stm_cursor_s` (float): Current playhead position in seconds.
  - `stm_window_ms` (float): Current window size in milliseconds.
  - `stm_locked` (bool): Region selection lock state.
  - `stm_investigations` (list): Array of past regional analysis reports.

## 4. Page Navigation & Structure

The application is structured linearly, mirroring the underlying analysis pipeline, plus advanced interactive views at the end.

1. **`app.py`** (Home): Ingestion, basic signal visualization, and global analysis trigger.
2. **`1_upload_preview.py`**: Confirmation of ingest and basic parameters.
3. **`2_parameters.py`**: Carrier frequency, bandwidth, SNR, sample rate.
4. **`3_modulation.py`**: CNN-based classification results, trust scores, and constellation/spectrogram plots of the whole capture.
5. **`4_demodulation.py`**: Symbol recovery, constellation geometry, phase/frequency error tracking.
6. **`5_deinterleaving.py`**: Matrix visualisations of permutation structures.
7. **`6_fec_decoding.py`**: Error correction scheme identification and bit recovery metrics.
8. **`7_correlation.py`**: Preamble/sync word detection and autocorrelation peaks.
9. **`8_report.py`**: Final concatenated bitstream and summary of the analysis chain.
10. **`9_time_machine.py`**: Highly interactive, stateful temporal exploration tool ("Signal Time Machine"). Features:
   - Synchronized spectrogram and time-domain views.
   - Draggable region selection.
   - Re-runs `analyze_capture` on sub-regions and compares with global results.
   - Decimation used for waveform rendering to maintain performance.
11. **`10_cyclostationary.py`**: Advanced DSP (Spectral Correlation Density) for concealed signals.
12. **`11_capability.py`**: Radar/EW specific capability inference (PRF, PRI, modulation agility).
13. **`12_fusion_landscape.py`**: Aggregate overview mapping probability across time/frequency.
14. **`13_signal_dna.py`**: Unsupervised anomaly detection / clustering of signal features.

## 5. Visualizations & Plotly Usage

All charting relies on Plotly (`plotly.graph_objects` and `plotly.express`).
- **Styling**: Plots are styled inline to match the dark theme (`paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"`, specific grid colors).
- **Performance**: High-density scatter plots (e.g., constellations) use `go.Scattergl` (WebGL) rather than standard SVGs.
- **Data Reduction**: `9_time_machine.py` explicitly decimates large sample arrays using `.reshape(-1, factor).mean(axis=1)` to avoid crashing the browser during interactive playback.
- **Interactivity**: The Time Machine uses Plotly's `selection` events alongside Streamlit's `st.plotly_chart(..., on_select="rerun")` to drive session state changes based on user clicks/drags.

## 6. Current UX Characteristics & Constraints

- **Execution Flow**: The user must ingest and run global analysis on `app.py` before other pages function correctly. Missing data displays a fallback `UNAVAILABLE` tag rather than crashing.
- **Modularity**: The UI expects `report` dicts where `report["stages"][stage_name]["available"]` explicitly gates visibility.
- **Responsiveness/Lag**: Heavy dependency on Streamlit's `st.rerun()` loop, particularly in `9_time_machine.py` during playback/dragging, introduces inherent round-trip latency.
- **Look and Feel**: The design achieves a bespoke "investigator dashboard" feel, heavily utilizing custom CSS cards, explicit borders, muted text for caveats, and monospace fonts for technical data.
