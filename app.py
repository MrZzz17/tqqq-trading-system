"""
TQQQ Trading System by MrZzz
"""

import streamlit as st

st.set_page_config(
    page_title="TQQQ Trading System | MrZzz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: #0a0f1a;
    }

    /* Metric cards — glass morphism */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 0.78em !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #f0f0f0 !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85em !important;
    }

    /* Tab styling — full width, premium */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255, 255, 255, 0.025);
        border-radius: 16px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        width: 100%;
        display: flex;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #9ca3af;
        font-weight: 600;
        font-size: 1.05em;
        padding: 14px 20px;
        transition: all 0.25s ease;
        flex: 1;
        justify-content: center;
        letter-spacing: 0.01em;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #d1d5db;
        background: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.10)) !important;
        color: #c7d2fe !important;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.12);
        font-weight: 700;
    }

    /* Buttons — gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6em 1.5em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
    }

    /* Dataframes — clean dark */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* Expanders — premium feel */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.04) !important;
        border-color: rgba(99, 102, 241, 0.15) !important;
    }
    .streamlit-expanderContent {
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-top: none;
        border-radius: 0 0 12px 12px;
        background: rgba(255, 255, 255, 0.01);
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #6366f1;
    }

    /* Hide chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Info/success/warning boxes */
    .stAlert {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* Sidebar — dark glass */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* Scrollbar — minimal */
    ::-webkit-scrollbar {
        width: 5px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    /* Links */
    a {
        color: #818cf8 !important;
        text-decoration: none !important;
    }
    a:hover {
        color: #a5b4fc !important;
    }

    /* Number styling */
    .big-number {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.04) !important;
    }

    /* Column gaps */
    [data-testid="column"] {
        padding: 0 8px;
    }
</style>
""", unsafe_allow_html=True)

import views.tqqq_dashboard as tqqq_page
tqqq_page.render()
