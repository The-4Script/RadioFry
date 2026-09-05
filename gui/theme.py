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

TOUR_STEPS = [
    {
        "number": "01",
        "eyebrow": "Start with the evidence",
        "title": "Bring a signal to the bench",
        "body": "Upload a short baseband WAV or raw IQ capture. RadioFry first checks the recording itself: format, sample count, waveform, and where energy sits in frequency.",
        "action": "Choose a file on the home screen, then open Upload & Preview.",
        "color": "teal",
    },
    {
        "number": "02",
        "eyebrow": "Measure before naming",
        "title": "Read the signal conditions",
        "body": "Bandwidth, SNR, and symbol-rate estimates describe the recording. They are useful measurements with assumptions, not a protocol verdict.",
        "action": "Check unknown or suspicious values before trusting downstream stages.",
        "color": "amber",
    },
    {
        "number": "03",
        "eyebrow": "Compare independent clues",
        "title": "Treat modulation as a hypothesis",
        "body": "A classical signal check and a CNN provide separate clues. Agreement strengthens the case; disagreement moves the result into review instead of hiding uncertainty.",
        "action": "Compare the fused decision with the alternatives and constellation.",
        "color": "blue",
    },
    {
        "number": "04",
        "eyebrow": "Follow the recovery chain",
        "title": "Bits are a beginning, not proof",
        "body": "Demodulation, deinterleaving, FEC, and correlation attempt to recover structure. Each stage tells you what it tried, what worked, and what still needs protocol context.",
        "action": "Use the sidebar stages to inspect each handoff in order.",
        "color": "violet",
    },
    {
        "number": "05",
        "eyebrow": "Finish with an audit trail",
        "title": "Package the reasoning",
        "body": "The final report keeps measurements, confidence, recovered bits, and limitations together. A strong result is one a jury can inspect, not just a label that sounds certain.",
        "action": "Download JSON for traceability and PDF for the review room.",
        "color": "orange",
    },
]


def apply_radiofry_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0a1014;
            --bg-raised: #0f181e;
            --surface: #131f27;
            --surface-2: #182832;
            --ink: #f1f6f5;
            --muted: #9aadb2;
            --line: #2a3b43;
            --teal: #51d5c2;
            --amber: #f1aa62;
            --blue: #72b7ff;
            --violet: #b49cff;
            --orange: #ff825c;
            --green: #67d391;
            --red: #ff817b;
        }
        .stApp { background: var(--bg); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(10,16,20,0.82); }
        [data-testid="stSidebar"] { background: #0d171c; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] * { color: #eaf3f0; }
        [data-testid="stSidebar"] hr { border-color: var(--line); }
        .block-container { max-width: 1320px; padding-top: 2.5rem; padding-bottom: 4rem; }
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: 0; }
        p, li, label { color: var(--ink); }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMarkdownContainer"] a { color: var(--teal); }
        .brand-mark { color: var(--teal); font-size: 0.7rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; }
        .stage-kicker { color: var(--teal); font-size: 0.74rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
        .stage-title { margin: 0.2rem 0 0.35rem; }
        .stage-purpose { color: var(--muted); font-size: 1.03rem; max-width: 780px; }
        .stage-rule { border-bottom: 1px solid var(--line); margin: 0.85rem 0 1.6rem; }
        .status-badge { display: inline-block; border-radius: 999px; padding: 0.3rem 0.68rem; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
        .status-ready { background: rgba(103,211,145,0.14); color: var(--green); border: 1px solid rgba(103,211,145,0.35); }
        .status-review { background: rgba(241,170,98,0.14); color: var(--amber); border: 1px solid rgba(241,170,98,0.35); }
        .status-muted { background: rgba(154,173,178,0.12); color: var(--muted); border: 1px solid rgba(154,173,178,0.28); }
        .status-error { background: rgba(255,129,123,0.14); color: var(--red); border: 1px solid rgba(255,129,123,0.35); }
        .evidence-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 1.1rem 1.2rem; margin-bottom: 1rem; box-shadow: 0 12px 30px rgba(0,0,0,0.12); }
        .evidence-panel h3, .evidence-panel h4 { margin-top: 0; }
        .evidence-label { color: var(--muted); font-size: 0.7rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }
        .hero-panel { background: linear-gradient(135deg, #142b32 0%, #17212e 54%, #2b2026 100%); border: 1px solid #31505a; border-radius: 14px; padding: 2.35rem 2.4rem 2.05rem; box-shadow: 0 22px 55px rgba(0,0,0,0.24); }
        .hero-panel h1, .hero-panel h2, .hero-panel h3, .hero-panel p { color: var(--ink); }
        .hero-panel h1 { font-size: 3rem; }
        .hero-panel p { max-width: 820px; color: #c9d8d8; }
        .sidebar-stage { border-left: 2px solid #2b3b42; margin: 0.25rem 0; padding: 0.4rem 0 0.4rem 0.75rem; }
        .sidebar-stage.active { border-left-color: var(--teal); background: rgba(81,213,194,0.08); }
        .sidebar-stage-number { color: var(--teal); font-size: 0.68rem; font-weight: 800; }
        .sidebar-stage-name { font-size: 0.86rem; }
        .tour-shell { background: #111d24; border: 1px solid #33515a; border-radius: 14px; padding: 1.2rem 1.35rem 1.1rem; box-shadow: 0 18px 42px rgba(0,0,0,0.2); }
        .tour-kicker { color: var(--teal); font-size: 0.7rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }
        .tour-title { color: var(--ink); font-size: 1.6rem; font-weight: 750; margin: 0.35rem 0 0.5rem; }
        .tour-body { color: #c5d2d3; font-size: 1rem; line-height: 1.6; max-width: 780px; }
        .tour-action { border-left: 3px solid var(--amber); color: #f0c38d; margin-top: 1rem; padding: 0.45rem 0 0.45rem 0.8rem; }
        .tour-dot { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; margin-right: 0.35rem; }
        .tour-teal { background: var(--teal); } .tour-amber { background: var(--amber); } .tour-blue { background: var(--blue); } .tour-violet { background: var(--violet); } .tour-orange { background: var(--orange); }
        .tour-closed { align-items: center; background: #111d24; border: 1px solid var(--line); border-radius: 10px; display: flex; gap: 0.8rem; justify-content: space-between; padding: 0.75rem 1rem; }
        .bit-preview { background: #091014; border: 1px solid #2b424a; border-radius: 7px; color: #ccefe5; font-family: "Cascadia Mono", Consolas, monospace; font-size: 0.78rem; line-height: 1.6; overflow-wrap: anywhere; padding: 0.85rem; }
        .stAlert { border-radius: 8px; }
        .stButton > button { border-radius: 7px; }
        .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); color: #071113; font-weight: 750; }
        .stButton > button[kind="primary"]:hover { background: #83e3d2; border-color: #83e3d2; }
        [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: 9px; }
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
        st.caption("Evidence first. Confidence visible. Review encouraged.")
        st.divider()
        for index, (number, name) in enumerate(STAGES, start=1):
            active_class = " active" if index == active_stage else ""
            marker = "●" if has_analysis and index <= active_stage else "○"
            st.markdown(f"<div class='sidebar-stage{active_class}'><span class='sidebar-stage-number'>{number} &nbsp; {marker}</span><br><span class='sidebar-stage-name'>{name}</span></div>", unsafe_allow_html=True)
        st.divider()
        if st.button("Open site tour", use_container_width=True):
            st.session_state["tour_open"] = True
            st.session_state["tour_step"] = 0
            st.rerun()
        st.caption("A quick guided walk through the workflow for juries and first-time reviewers.")


def render_site_tour() -> None:
    if not st.session_state.get("tour_open", True):
        st.markdown("<div class='tour-closed'><span><strong>Site tour paused</strong><br><span style='color:#9aadb2'>Walk through the workflow block by block.</span></span></div>", unsafe_allow_html=True)
        if st.button("Start site tour", type="primary", use_container_width=True):
            st.session_state["tour_open"] = True
            st.session_state["tour_step"] = 0
            st.rerun()
        return

    step_index = max(0, min(st.session_state.get("tour_step", 0), len(TOUR_STEPS) - 1))
    step = TOUR_STEPS[step_index]
    st.progress((step_index + 1) / len(TOUR_STEPS), text=f"Site tour  {step_index + 1} of {len(TOUR_STEPS)}")
    st.markdown(
        f"""
        <div class="tour-shell">
            <div class="tour-kicker"><span class="tour-dot tour-{step['color']}"></span>{step['eyebrow']}</div>
            <div class="tour-title">{step['number']} / {step['title']}</div>
            <div class="tour-body">{step['body']}</div>
            <div class="tour-action"><strong>Try this:</strong> {step['action']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    back, close, forward = st.columns([1, 1, 1])
    with back:
        if st.button("Back", disabled=step_index == 0, use_container_width=True):
            st.session_state["tour_step"] = step_index - 1
            st.rerun()
    with close:
        if st.button("Skip tour", use_container_width=True):
            st.session_state["tour_open"] = False
            st.rerun()
    with forward:
        label = "Finish" if step_index == len(TOUR_STEPS) - 1 else "Next"
        if st.button(label, type="primary", use_container_width=True):
            if step_index == len(TOUR_STEPS) - 1:
                st.session_state["tour_open"] = False
            else:
                st.session_state["tour_step"] = step_index + 1
            st.rerun()
