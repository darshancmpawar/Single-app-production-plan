"""Centralized Streamlit styling for the app."""

import streamlit as st


APP_STYLE = """
<style>
.stApp {background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #111827 100%); color: #e5e7eb;}
div[data-testid="stTabs"] button[role="tab"] {border-radius: 10px; padding: 0.5rem 1rem; font-weight: 600;}
div[data-testid="stTabs"] button[aria-selected="true"] {background: #22c55e !important; color: #052e16 !important;}
.block-container {padding-top: 1.5rem;}
.config-card {background: rgba(15, 23, 42, 0.65); border: 1px solid #334155; border-radius: 14px; padding: 12px 16px; margin-bottom: 10px;}
</style>
"""


def apply_app_styling() -> None:
    st.markdown(APP_STYLE, unsafe_allow_html=True)
