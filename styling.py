"""Centralized Streamlit styling for the app."""

import streamlit as st


APP_STYLE = """
<style>
[data-testid="stToolbar"], header, #MainMenu, footer {visibility: hidden; height: 0; position: fixed;}
.stApp {
  background: radial-gradient(circle at top right, #e0f2fe 0%, #f8fafc 35%, #eef2ff 100%);
  color: #0f172a;
}
.block-container {padding-top: 1.2rem; max-width: 1200px;}
h1, h2, h3 {color: #0b3b5a;}

div[data-testid="stTabs"] button[role="tab"] {
  border-radius: 10px;
  padding: 0.5rem 1rem;
  font-weight: 600;
  border: 1px solid #cbd5e1;
  background: #ffffff;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  background: #0284c7 !important;
  color: #ffffff !important;
  border-color: #0369a1 !important;
}
.config-card {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid #cbd5e1;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
}
</style>
"""


def apply_app_styling() -> None:
    st.markdown(APP_STYLE, unsafe_allow_html=True)
