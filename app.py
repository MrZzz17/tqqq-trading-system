"""
TQQQ Trading System
===================
A rules-based TQQQ swing trading dashboard with CAN SLIM stock screening.

Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="TQQQ Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import views.tqqq_dashboard as tqqq_page
tqqq_page.render()
