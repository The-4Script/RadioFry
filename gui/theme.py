import streamlit as st


def apply_radiofry_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-start: #07111f;
            --bg-mid: #0b1830;
            --bg-end: #101e3d;
            --panel: rgba(15, 23, 42, 0.72);
            --panel-strong: rgba(15, 23, 42, 0.92);
            --ink: #e2ecff;
            --muted: #a6bedf;
            --line: rgba(148, 163, 184, 0.2);
            --accent: #59d0ff;
            --accent-2: #8b5cf6;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
        }

        .stApp {
            background: linear-gradient(135deg, var(--bg-start), var(--bg-mid) 35%, var(--bg-end));
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: rgba(11, 17, 31, 0.96);
            border-right: 1px solid var(--line);
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--line);
            border-radius: 12px 12px 0 0;
            color: var(--muted);
            padding: 0.65rem 1rem;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(89, 208, 255, 0.2), rgba(139, 92, 246, 0.2));
            color: var(--ink);
            border-color: rgba(89, 208, 255, 0.4);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            color: var(--ink);
            letter-spacing: -0.03em;
        }

        p, li, div, label {
            color: var(--ink);
        }

        .hero-panel {
            background: linear-gradient(135deg, rgba(89, 208, 255, 0.12), rgba(139, 92, 246, 0.18));
            border: 1px solid rgba(89, 208, 255, 0.28);
            border-radius: 20px;
            padding: 1.5rem 1.5rem 1rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
        }

        .glass-card {
            background: rgba(15, 23, 42, 0.68);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(8, 15, 26, 0.18);
            margin-bottom: 1rem;
        }

        .highlight {
            display: inline-block;
            background: rgba(52, 211, 153, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.35);
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            color: #d9fff0;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #081e2d;
            font-weight: 800;
        }

        .metric-box {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.78), rgba(15, 23, 42, 0.46));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }

        .info-pill {
            background: rgba(89, 208, 255, 0.12);
            border: 1px solid rgba(89, 208, 255, 0.25);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            color: var(--ink);
        }

        div[data-testid="stMetricValue"] {
            color: white;
            font-weight: 700;
        }

        .stAlert {
            border-radius: 16px;
            border: 1px solid rgba(89, 208, 255, 0.25);
        }
    </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_explainer(title: str, layman: str, expert: str, next_step: str) -> None:
    st.subheader(title)
    tabs = st.tabs(["Layman view", "Expert view", "What happens next"])
    with tabs[0]:
        st.markdown(f"<div class='glass-card'>{layman}</div>", unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(f"<div class='glass-card'>{expert}</div>", unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(f"<div class='glass-card'>{next_step}</div>", unsafe_allow_html=True)
