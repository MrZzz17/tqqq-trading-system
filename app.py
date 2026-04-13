"""
TQQQ Trading System
===================
A rules-based TQQQ swing trading dashboard with CAN SLIM stock screening.
Built for personal use with an IBD Digital subscription.

Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="TQQQ Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("TQQQ Trading System")
st.sidebar.markdown("---")

# --- Main page redirects to TQQQ dashboard ---
import pages.tqqq_dashboard as tqqq_page
tqqq_page.render()
