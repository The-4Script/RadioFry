import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError


def apply_radiofry_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1117;
            --bg-raised: #111a22;
            --surface: #16212a;
            --surface-2: #1d2b35;
            --ink: #f4f7f4;
            --muted: #9eafb2;
            --line: #2b3b44;
            --teal: #67e0c8;
            --amber: #f3bd73;
            --blue: #8bbcff;
            --violet: #c1b1ff;
            --orange: #ff9675;
            --green: #7be0a0;
            --red: #ff918b;
        }
        .stApp { background: radial-gradient(circle at 88% 0%, #1a2d35 0, transparent 31rem), var(--bg); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(11,17,23,0.9); }
        [data-testid="stSidebar"] { background: #0d151c; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] * { color: #eaf3f0; }
        [data-testid="stSidebar"] hr { border-color: var(--line); }
        [data-testid="stSidebarNav"] { display: none; }
        .stDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
        .block-container { max-width: 1400px; padding-top: 2.2rem; padding-bottom: 5rem; }
        h1, h2, h3, h4 { color: var(--ink); font-family: "Trebuchet MS", "Segoe UI", sans-serif; letter-spacing: 0; }
        h1 { font-weight: 700; }
        h2 { margin-top: 2rem; }
        p, li, label { color: var(--ink); }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMarkdownContainer"] a { color: var(--teal); }
        .brand-mark { color: var(--teal); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; }
        .stage-kicker { color: var(--teal); font-size: 0.74rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
        .stage-title { margin: 0.2rem 0 0.35rem; }
        .stage-purpose { color: var(--muted); font-size: 1.03rem; max-width: 780px; }
        .stage-rule { border-bottom: 1px solid var(--line); margin: 0.85rem 0 1.8rem; }
        .status-badge { display: inline-block; border-radius: 999px; padding: 0.3rem 0.68rem; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
        .status-ready { background: rgba(103,211,145,0.14); color: var(--green); border: 1px solid rgba(103,211,145,0.35); }
        .status-review { background: rgba(241,170,98,0.14); color: var(--amber); border: 1px solid rgba(241,170,98,0.35); }
        .status-muted { background: rgba(154,173,178,0.12); color: var(--muted); border: 1px solid rgba(154,173,178,0.28); }
        .status-error { background: rgba(255,129,123,0.14); color: var(--red); border: 1px solid rgba(255,129,123,0.35); }
        .evidence-panel { background: rgba(22,33,42,0.88); border: 1px solid var(--line); border-radius: 6px; padding: 1.25rem 1.35rem; margin-bottom: 1rem; box-shadow: 0 14px 34px rgba(0,0,0,0.14); }
        .evidence-panel h3, .evidence-panel h4 { margin-top: 0; }
        .evidence-label { color: var(--muted); font-size: 0.7rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }
        .hero-panel { background: linear-gradient(118deg, #18343a 0%, #15252f 62%, #36262b 100%); border: 1px solid #3c6266; border-radius: 8px; padding: 2.45rem 2.7rem 2.25rem; box-shadow: 0 22px 55px rgba(0,0,0,0.24); position: relative; overflow: hidden; }
        .hero-panel::after { content: ""; position: absolute; width: 14rem; height: 14rem; border: 1px solid rgba(103,224,200,0.25); border-radius: 50%; right: -4rem; top: -5rem; box-shadow: 0 0 0 1.4rem rgba(103,224,200,0.04), 0 0 0 3rem rgba(103,224,200,0.025); }
        .hero-panel h1, .hero-panel h2, .hero-panel h3, .hero-panel p { color: var(--ink); }
        .hero-panel h1 { font-size: clamp(2.4rem, 5vw, 4.2rem); letter-spacing: -0.04em; }
        .hero-panel p { max-width: 820px; color: #c9d8d8; }
        .sidebar-stage { border-left: 2px solid #2b3b42; margin: 0.18rem 0; padding: 0.48rem 0 0.48rem 0.75rem; }
        .sidebar-stage.active { border-left-color: var(--teal); background: rgba(103,224,200,0.09); }
        .sidebar-stage-number { color: var(--teal); font-size: 0.68rem; font-weight: 800; }
        .sidebar-stage-name { font-size: 0.86rem; }
        .portal-link { align-items: center; border-radius: 6px; color: #dce9e6 !important; display: flex; font-size: 0.86rem; gap: 0.55rem; padding: 0.35rem 0.45rem; text-decoration: none !important; }
        .portal-link:hover { background: rgba(81,213,194,0.1); color: var(--teal) !important; }
        .bit-preview { background: #091014; border: 1px solid #2b424a; border-radius: 4px; color: #ccefe5; font-family: "Cascadia Mono", Consolas, monospace; font-size: 0.78rem; line-height: 1.6; overflow-wrap: anywhere; padding: 0.85rem; }
        .stAlert { border-radius: 5px; }
        .stButton > button { border-radius: 5px; min-height: 2.55rem; }
        .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); color: #071113; font-weight: 750; }
        .stButton > button[kind="primary"]:hover { background: #83e3d2; border-color: #83e3d2; }
        [data-testid="stFileUploader"] { background: rgba(22,33,42,0.75); border: 1px dashed #46616a; border-radius: 6px; padding: 0.35rem; }
        [data-testid="stFileUploader"] section { border: 0; background: transparent; }
        [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
        [data-testid="stMetric"] { background: rgba(22,33,42,0.7); border-left: 2px solid var(--teal); padding: 0.8rem 1rem; }
        @media (max-width: 700px) {
            .block-container { padding: 1.2rem 1rem 3rem; }
            .hero-panel { padding: 1.7rem 1.35rem; }
            .hero-panel h1 { font-size: 2.7rem; }
        }
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
    del active_stage, has_analysis
    with st.sidebar:
        st.markdown("<div class='brand-mark'>RadioFry / SIH26147</div>", unsafe_allow_html=True)
        st.markdown("### Analysis bench")
        st.caption("Evidence first. Confidence visible. Review encouraged.")
        st.divider()
        st.markdown("<div class='evidence-label'>Navigate</div>", unsafe_allow_html=True)
        navigation = [
            ("app.py", "/", "⌂", "Overview"),
            ("pages/1_upload_preview.py", "/upload_preview", "∿", "Upload & preview"),
            ("pages/2_parameters.py", "/parameters", "≋", "Parameters"),
            ("pages/3_modulation.py", "/modulation", "◈", "Modulation"),
            ("pages/4_demodulation.py", "/demodulation", "↝", "Demodulation"),
            ("pages/5_deinterleaving.py", "/deinterleaving", "⇄", "Deinterleaving"),
            ("pages/6_fec_decoding.py", "/fec_decoding", "◇", "FEC decoding"),
            ("pages/7_correlation.py", "/correlation", "◎", "Correlation"),
            ("pages/8_report.py", "/report", "▤", "Report"),
        ]
        for page, href, icon, label in navigation:
            try:
                st.page_link(page, label=label)
            except StreamlitPageNotFoundError:
                st.markdown(f"<a class='portal-link' href='{href}'><span>{icon}</span><span>{label}</span></a>", unsafe_allow_html=True)


def render_page_shell(active_stage: int) -> None:
    apply_radiofry_theme()
    render_sidebar(active_stage=active_stage, has_analysis="report" in st.session_state)
