import streamlit as st


STAGES = [
    ("01", "Upload & preview"),
    ("02", "Signal parameters"),
    ("03", "Modulation"),
    ("04", "Demodulation"),
    ("05", "Deinterleaving"),
    ("06", "FEC decoding"),
    ("07", "Correlation"),
    ("08", "Report"),
]


def apply_radiofry_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f4f1ea;
            --surface: #fffdf8;
            --surface-muted: #ebe8df;
            --ink: #17252a;
            --muted: #657276;
            --line: #d8d5cc;
            --teal: #147d7b;
            --orange: #c7672f;
            --green: #28794b;
            --red: #a8463d;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stSidebar"] { background: #202d30; border-right: 0; }
        [data-testid="stSidebar"] * { color: #eef4ef; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.14); }
        .block-container { max-width: 1280px; padding-top: 2.5rem; padding-bottom: 4rem; }
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: 0; }
        p, li, label { color: var(--ink); }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        .brand-mark { color: #f4c095; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; }
        .stage-kicker { color: var(--teal); font-size: 0.76rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }
        .stage-title { margin: 0.2rem 0 0.35rem; }
        .stage-purpose { color: var(--muted); font-size: 1.02rem; max-width: 760px; }
        .stage-rule { border-bottom: 1px solid var(--line); margin: 0.8rem 0 1.5rem; }
        .status-badge { display: inline-block; border-radius: 999px; padding: 0.28rem 0.62rem; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
        .status-ready { background: #dff0e3; color: var(--green); }
        .status-review { background: #f8e5d8; color: var(--orange); }
        .status-muted { background: var(--surface-muted); color: var(--muted); }
        .status-error { background: #f7e1de; color: var(--red); }
        .evidence-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 1.05rem 1.15rem; margin-bottom: 1rem; }
        .evidence-panel h3, .evidence-panel h4 { margin-top: 0; }
        .evidence-label { color: var(--muted); font-size: 0.72rem; font-weight: 800; letter-spacing: 0.09em; text-transform: uppercase; }
        .hero-panel { background: #203b3e; border-left: 6px solid var(--orange); border-radius: 8px; color: #f5f0e7; padding: 2rem 2.1rem 1.8rem; }
        .hero-panel h1, .hero-panel h2, .hero-panel h3, .hero-panel p { color: #f5f0e7; }
        .hero-panel p { max-width: 780px; }
        .sidebar-stage { border-left: 2px solid rgba(255,255,255,0.2); margin: 0.25rem 0; padding: 0.35rem 0 0.35rem 0.7rem; }
        .sidebar-stage.active { border-left-color: #f4c095; }
        .sidebar-stage-number { color: #f4c095; font-size: 0.7rem; font-weight: 800; }
        .sidebar-stage-name { font-size: 0.86rem; }
        .tutorial-step { border-left: 3px solid var(--teal); padding: 0.1rem 0 0.1rem 0.9rem; margin: 0.7rem 0; }
        .bit-preview { background: #1f2a2c; border-radius: 6px; color: #dff4e8; font-family: "Cascadia Mono", Consolas, monospace; font-size: 0.78rem; line-height: 1.6; overflow-wrap: anywhere; padding: 0.85rem; }
        .stAlert { border-radius: 7px; }
        .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
        .stButton > button[kind="primary"]:hover { background: #0f6967; border-color: #0f6967; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stage_header(number: str, title: str, purpose: str, status: str = "Ready", status_kind: str = "ready") -> None:
    st.markdown(
        f"""
        <div class="stage-kicker">Stage {number}</div>
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:end;flex-wrap:wrap;">
            <div><h1 class="stage-title">{title}</h1><div class="stage-purpose">{purpose}</div></div>
            <span class="status-badge status-{status_kind}">{status}</span>
        </div>
        <div class="stage-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def render_method_note(title: str, body: str) -> None:
    with st.expander(title, expanded=False):
        st.markdown(body)


def render_empty_state(message: str, detail: str = "") -> None:
    detail_markup = f"<p style='color:var(--muted)'>{detail}</p>" if detail else ""
    st.markdown(f"<div class='evidence-panel'><div class='evidence-label'>Waiting for input</div><p>{message}</p>{detail_markup}</div>", unsafe_allow_html=True)


def render_bit_preview(bits: object, label: str = "Bit preview", limit: int = 512) -> None:
    values = list(bits or [])
    preview = "".join(str(int(bit)) for bit in values[:limit]) or "No bits available"
    st.markdown(f"<div class='evidence-label'>{label} · {len(values):,} total</div><div class='bit-preview'>{preview}</div>", unsafe_allow_html=True)


def render_sidebar(active_stage: int, has_analysis: bool) -> None:
    with st.sidebar:
        st.markdown("<div class='brand-mark'>RadioFry / SIH26147</div>", unsafe_allow_html=True)
        st.markdown("### Analysis bench")
        st.caption("A signal evidence workspace, not a black-box verdict.")
        st.divider()
        for index, (number, name) in enumerate(STAGES, start=1):
            active_class = " active" if index == active_stage else ""
            marker = "●" if has_analysis and index <= active_stage else "○"
            st.markdown(f"<div class='sidebar-stage{active_class}'><span class='sidebar-stage-number'>{number} &nbsp; {marker}</span><br><span class='sidebar-stage-name'>{name}</span></div>", unsafe_allow_html=True)
        st.divider()
        st.checkbox("Show jury tour", key="show_tutorial", value=st.session_state.get("show_tutorial", True))
        st.caption("Use the tour once for context, then hide it for a cleaner working view.")


def render_tutorial() -> None:
    if not st.session_state.get("show_tutorial", True):
        return
    with st.expander("Jury tour · how to read this workspace", expanded=True):
        st.markdown("The tour is optional. Hide it from the sidebar once the workflow is familiar.")
        steps = [
            ("01 · Capture", "Start with a short baseband WAV or raw IQ file. The preview checks whether the signal is present and inspectable."),
            ("02 · Measure", "Bandwidth, SNR, and symbol-rate estimates describe the recording. They are measurements with assumptions, not protocol facts."),
            ("03 · Compare", "Classical signal features and the CNN produce independent modulation hypotheses. Agreement raises confidence; disagreement triggers review."),
            ("04 · Recover", "Demodulation, deinterleaving, FEC, and correlation attempt to recover structure from bits. Each result records what was attempted and what remains uncertain."),
            ("05 · Package", "The report separates evidence, confidence, and limitations so a jury can inspect the reasoning rather than accept a single label."),
        ]
        for title, copy in steps:
            st.markdown(f"<div class='tutorial-step'><strong>{title}</strong><br>{copy}</div>", unsafe_allow_html=True)
