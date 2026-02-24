"""Centralized Streamlit styling for the app."""

import streamlit as st


APP_STYLE = """
<style>
[data-testid="stToolbar"], header, #MainMenu, footer {visibility: hidden; height: 0; position: fixed;}
.stApp {
  background: radial-gradient(circle at top right, #111827 0%, #0b1220 38%, #020617 100%);
  color: #e5e7eb;
}
.block-container {padding-top: 1.2rem; max-width: 1200px;}
h1, h2, h3 {color: #f8fafc;}

/* Inputs and labels */
label, .stMarkdown, .stCaption, p, span {color: #d1d5db !important;}

/* Tabs */
div[data-testid="stTabs"] button[role="tab"] {
  border-radius: 10px;
  padding: 0.5rem 1rem;
  font-weight: 600;
  border: 1px solid #334155;
  background: #0f172a;
  color: #cbd5e1;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  background: #22c55e !important;
  color: #052e16 !important;
  border-color: #16a34a !important;
}

/* Card */
.config-card {
  background: rgba(15, 23, 42, 0.92);
  border: 1px solid #334155;
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 20px rgba(2, 6, 23, 0.45);
}
</style>
"""


def apply_app_styling() -> None:
    st.markdown(APP_STYLE, unsafe_allow_html=True)
