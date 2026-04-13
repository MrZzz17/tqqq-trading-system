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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #15202B 0%, #1a2836 50%, #15202B 100%);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(29, 161, 242, 0.06);
        border: 1px solid rgba(29, 161, 242, 0.15);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #8899A6 !important;
        font-size: 0.85em !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #E7E9EA !important;
        font-weight: 700 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(25, 39, 52, 0.8);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8899A6;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(29, 161, 242, 0.15) !important;
        color: #1DA1F2 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1DA1F2, #0d8bd9);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d8bd9, #0a6fad);
        transform: translateY(-1px);
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(29, 161, 242, 0.05);
        border-radius: 8px;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #1DA1F2;
    }

    /* Hide deploy button and main menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Info/success/warning boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #192734;
        border-right: 1px solid rgba(29, 161, 242, 0.1);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #15202B;
    }
    ::-webkit-scrollbar-thumb {
        background: #38444D;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

import views.tqqq_dashboard as tqqq_page
tqqq_page.render()
