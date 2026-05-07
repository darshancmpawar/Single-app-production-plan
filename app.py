"""
UI — Streamlit app with client dropdown. Dynamic UI per client capabilities.
"""
import os
import warnings
import logging

import pandas as pd
import streamlit as st

from client_database import CLIENT_LIST, name_to_key, get_info
from client_logic import get_logic
from Logic_Definer import save_client_configuration
from ml_core import (
    artifacts_exist,
    clear_cache,
    load_map,
    norm,
    fmt_cols,
    cats_match,
    get_nv_cats,
    predict,
    train_model,
)
from planner import (
    build_row,
    client_plan,
    fixed_pp_client_plan,
    vendor_plan,
    aggressive_plan,
    special_day_mg,
    gavg,
    classify,
    mg5,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# ═══════════════ PAGE CONFIG — must be the first Streamlit call ═══════════════
st.set_page_config(
    page_title="Production Planner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════ DARK THEME CSS ═══════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── VARIABLES ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b27;
    --card:      #1e2533;
    --card2:     #242d40;
    --border:    #2d3748;
    --accent:    #00d4aa;
    --adim:      rgba(0,212,170,.12);
    --purple:    #7c4dff;
    --pdim:      rgba(124,77,255,.12);
    --gold:      #f59e0b;
    --gdim:      rgba(245,158,11,.12);
    --text:      #e2e8f0;
    --tsec:      #94a3b8;
    --tmut:      #4a5568;
    --grad:      linear-gradient(135deg,#00d4aa 0%,#7c4dff 100%);
    --grad-rev:  linear-gradient(135deg,#7c4dff 0%,#00d4aa 100%);
    --err:       #ef4444;
}

/* ── GLOBAL ── */
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── APP BACKGROUND ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--bg) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.main .block-container {
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 1380px !important;
}

/* ── HEADINGS ── */
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -.02em !important;
}
p, [data-testid="stMarkdownContainer"] p {
    color: var(--tsec) !important;
}
label {
    color: var(--tsec) !important;
    font-size: .83rem !important;
    font-weight: 500 !important;
    letter-spacing: .01em !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--adim) !important;
}

/* ── TEXT INPUT ── */
.stTextInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    padding: .55rem .9rem !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--adim) !important;
}
.stTextInput input::placeholder { color: var(--tmut) !important; }

/* ── NUMBER INPUT ── */
.stNumberInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
.stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--adim) !important;
}
.stNumberInput [data-testid="stNumberInputContainer"] button {
    background: var(--card2) !important;
    border-color: var(--border) !important;
    color: var(--tsec) !important;
}

/* ── DATE INPUT ── */
.stDateInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
.stDateInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--adim) !important;
}

/* ── MULTISELECT ── */
[data-testid="stMultiSelect"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    transition: border-color .15s ease !important;
}
[data-testid="stMultiSelect"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--adim) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: var(--adim) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

/* ── TEXTAREA ── */
.stTextArea textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--adim) !important;
}
.stTextArea textarea::placeholder { color: var(--tmut) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: var(--grad) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .55rem 1.75rem !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    letter-spacing: .02em !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
    box-shadow: 0 4px 14px rgba(0,212,170,.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,170,.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--tsec) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: .9rem !important;
    padding: .5rem 1.5rem !important;
    transition: all .2s ease !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--grad) !important;
    color: #fff !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

/* ── TOGGLE ── */
[data-testid="stToggle"] label { color: var(--tsec) !important; }

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label { color: var(--tsec) !important; }

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--tmut) !important;
    font-size: .72rem !important;
    text-transform: uppercase !important;
    letter-spacing: .07em !important;
    font-weight: 500 !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid transparent !important;
}
div[data-testid="stAlert"][data-type="success"] {
    background: rgba(0,212,170,.08) !important;
    border-color: rgba(0,212,170,.3) !important;
    border-left: 3px solid var(--accent) !important;
}
div[data-testid="stAlert"][data-type="warning"] {
    background: rgba(245,158,11,.08) !important;
    border-left: 3px solid var(--gold) !important;
}
div[data-testid="stAlert"][data-type="error"] {
    background: rgba(239,68,68,.08) !important;
    border-left: 3px solid var(--err) !important;
}
div[data-testid="stAlert"][data-type="info"] {
    background: var(--pdim) !important;
    border-left: 3px solid var(--purple) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] > div {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── DATA EDITOR ── */
[data-testid="stDataEditor"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] > div { color: var(--accent) !important; }

/* ── DIVIDER ── */
hr {
    border-color: var(--border) !important;
    opacity: .6 !important;
    margin: 1.25rem 0 !important;
}

/* ── CAPTION ── */
[data-testid="stCaptionContainer"] {
    color: var(--tmut) !important;
    font-size: .78rem !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4a5568; }

/* ── POPOVER / DROPDOWN ── */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] li { color: var(--text) !important; }
[data-baseweb="menu"] li:hover { background: var(--card2) !important; }

/* ═══════ CUSTOM LAYOUT COMPONENTS ═══════ */

/* plan result header strip */
.plan-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .85rem 1.25rem;
    margin-bottom: .4rem;
}
.plan-title {
    color: var(--text);
    font-weight: 600;
    font-size: 1rem;
}

/* MG chip badges */
.mg-chip {
    background: var(--adim);
    color: var(--accent);
    border: 1px solid rgba(0,212,170,.4);
    border-radius: 20px;
    padding: .22rem .95rem;
    font-size: .82rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    white-space: nowrap;
}
.mg-chip-nv {
    background: var(--pdim);
    color: #a78bfa;
    border: 1px solid rgba(124,77,255,.4);
    border-radius: 20px;
    padding: .22rem .95rem;
    font-size: .82rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    white-space: nowrap;
}
.mg-chip-gold {
    background: var(--gdim);
    color: var(--gold);
    border: 1px solid rgba(245,158,11,.4);
    border-radius: 20px;
    padding: .22rem .95rem;
    font-size: .82rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    white-space: nowrap;
}

/* section header bar */
.sec-head {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .6rem .9rem;
    background: var(--card);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
    margin: 1.4rem 0 .8rem;
}
.sec-head span { color: var(--text); font-weight: 600; font-size: .9rem; }

/* context info bar */
.info-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .55rem 1rem;
    color: var(--tsec);
    font-size: .85rem;
    margin: .4rem 0 .8rem;
}
.info-bar strong { color: var(--text); }

/* hero header */
.hero-wrap {
    padding: .4rem 0 1.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.hero-title {
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.95rem;
    font-weight: 700;
    letter-spacing: -.035em;
    line-height: 1.15;
    display: inline-block;
}
.hero-sub {
    color: var(--tmut);
    font-size: .85rem;
    margin-top: .25rem;
}
</style>
""", unsafe_allow_html=True)


# ── helper (tekion 2-group final veg MG) ──
def _two_group_fv(results, is_nv, nv_cat, logic):
    nv, st2, rest = classify(results, nv_cat, logic.star_categories)
    raw = (gavg(st2) + gavg(rest)) / 2
    return raw if is_nv else max(logic.adjust_vendor_mg(raw), 0)


# ═══════════════ HEADER ═══════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🍽️ Per Pax Production Planner</div>
    <div class="hero-sub">Cafeteria AI &nbsp;·&nbsp; Real-time menu intelligence &amp; operational planning</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════ CLIENT SELECTOR ═══════════════
sel_col, mode_col = st.columns([2, 3])
with sel_col:
    sel = st.selectbox("Select Client", CLIENT_LIST, key="client_sel")

CK   = name_to_key(sel)
INFO = get_info(CK)
L    = get_logic(CK)

with mode_col:
    if L.has_embeddings:
        badge_bg, badge_color, badge_label = "rgba(0,212,170,.12)", "#00d4aa", "Embedding Model"
    else:
        badge_bg, badge_color, badge_label = "rgba(245,158,11,.12)", "#f59e0b", "Multiplier Only"
    st.markdown(f"""
    <div style="padding-top:1.7rem;display:flex;align-items:center;gap:.75rem;">
        <span style="background:{badge_bg};color:{badge_color};border:1px solid {badge_color};
                     border-radius:20px;padding:.28rem 1rem;font-size:.8rem;font-weight:600;">
            {badge_label}
        </span>
        <span style="color:#4a5568;font-size:.82rem;">
            Active: <strong style="color:#94a3b8;">{sel}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)


# config mismatch guard
db_mode = INFO.get("has_embeddings", None)
if db_mode is not None and bool(db_mode) != bool(L.has_embeddings):
    st.error(
        f"Config mismatch for {sel}: "
        f"client_database.has_embeddings={db_mode} vs client_logic.has_embeddings={L.has_embeddings}. "
        "Please align configs."
    )
    st.stop()


def k(name: str) -> str:
    return f"{CK}::{name}"


def kcat(cat: str) -> str:
    return norm(str(cat)).replace(" ", "_").replace("/", "_").replace("-", "_")


def _default_client_config():
    return {
        "client_name": sel,
        "menu_categories": list(L.fixed_categories),
        "nonveg_mode": getattr(L, "custom_nonveg_mode", ("Optional" if L.has_nonveg_toggle else "Not Needed")),
        "star_categories": list(L.star_categories),
        "slab_adjustments": list(getattr(L, "slab_adjustments", [])),
        "additional_requirements": getattr(L, "additional_requirements", ""),
    }


cfg_key = k("client_config")
if cfg_key not in st.session_state:
    st.session_state[cfg_key] = _default_client_config()


# ═══════════════ TABS ═══════════════
tab1, tab2 = st.tabs(["⚙️   Config Client", "📊   Generate Production Plan"])


# ═══════════════════════════════════════════════════════
#  TAB 1 — CLIENT CONFIGURATION
# ═══════════════════════════════════════════════════════
with tab1:
    cfg = st.session_state[cfg_key]

    st.markdown('<div class="sec-head"><span>🏢 Client Identity</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cfg["client_name"] = st.text_input(
            "Client Name",
            value=cfg.get("client_name", sel),
            key=k("cfg_client_name"),
        )
    with c2:
        cfg["nonveg_mode"] = st.selectbox(
            "Non-Veg Switch",
            ["Required", "Optional", "Not Needed"],
            index=["Required", "Optional", "Not Needed"].index(cfg.get("nonveg_mode", "Optional")),
            key=k("cfg_nonveg_mode"),
        )

    st.markdown('<div class="sec-head"><span>🗂️ Menu Configuration</span></div>', unsafe_allow_html=True)
    available_categories = sorted({str(x) for x in list(L.fixed_categories)})
    default_menu = [c for c in cfg.get("menu_categories", []) if c in available_categories]
    if not default_menu:
        default_menu = [c for c in L.fixed_categories if c in available_categories]

    cfg["menu_categories"] = st.multiselect(
        "Menu Categories",
        options=available_categories,
        default=default_menu,
        key=k("cfg_menu_categories"),
    )

    cfg["star_categories"] = st.multiselect(
        "⭐  Star Items  —  priority categories for vendor MG calculation",
        options=cfg["menu_categories"] if cfg["menu_categories"] else available_categories,
        default=[x for x in cfg.get("star_categories", []) if x in (cfg["menu_categories"] or available_categories)],
        key=k("cfg_star_categories"),
    )

    st.markdown('<div class="sec-head"><span>📐 Slab-wise MG Adjustment</span></div>', unsafe_allow_html=True)
    slab_default = cfg.get("slab_adjustments", [])
    seed_rows = slab_default if slab_default else [{"min_mg": None, "max_mg": None, "adjustment_pct": None}]
    slab_df = pd.DataFrame(seed_rows, columns=["min_mg", "max_mg", "adjustment_pct"])
    slab_edit = st.data_editor(
        slab_df,
        num_rows="dynamic",
        use_container_width=True,
        key=k("cfg_slab_editor"),
        column_config={
            "min_mg":          st.column_config.NumberColumn("Min MG",       min_value=0, format="%.0f"),
            "max_mg":          st.column_config.NumberColumn("Max MG",       min_value=0, format="%.0f"),
            "adjustment_pct":  st.column_config.NumberColumn("Adjustment %", format="%.1f"),
        },
    )
    cfg["slab_adjustments"] = [
        {
            "min_mg":          float(r["min_mg"]),
            "max_mg":          float(r["max_mg"]),
            "adjustment_pct":  float(r["adjustment_pct"]),
        }
        for _, r in slab_edit.dropna(subset=["min_mg", "max_mg", "adjustment_pct"]).iterrows()
    ]

    st.markdown('<div class="sec-head"><span>📝 Additional Requirements</span></div>', unsafe_allow_html=True)
    cfg["additional_requirements"] = st.text_area(
        "Client-specific notes or constraints",
        value=cfg.get("additional_requirements", ""),
        key=k("cfg_additional_requirements"),
        height=100,
        placeholder="e.g. Always add 10% buffer on Mondays...",
    )

    st.session_state[cfg_key] = cfg

    st.markdown("<br>", unsafe_allow_html=True)
    save_col, _ = st.columns([1, 4])
    with save_col:
        if st.button("💾  Save Client Config", key=k("cfg_save_btn"), use_container_width=True):
            if not cfg.get("client_name", "").strip():
                st.error("Client Name is required.")
            elif not cfg.get("menu_categories"):
                st.error("At least one menu category is required.")
            else:
                save_client_configuration(CK, cfg)
                st.success("Client configuration saved successfully.")


# ─── slab helper (module-level so the plan tab can use it) ───
def _apply_slab_adjustment(base_mg: float, slabs: list[dict]) -> float:
    mg = float(base_mg)
    for slab in slabs:
        lo  = slab.get("min_mg")
        hi  = slab.get("max_mg")
        pct = slab.get("adjustment_pct")
        if lo is None or hi is None or pct is None:
            continue
        if lo <= mg <= hi:
            return max(mg + (mg * pct / 100.0), 1.0)
    return mg


# ═══════════════════════════════════════════════════════
#  TAB 2 — GENERATE PRODUCTION PLAN
# ═══════════════════════════════════════════════════════
def _render_generate_production_plan_tab():

    # ── TOASTTAB (multiplier-only path) ──────────────────────────────────
    if not L.has_embeddings:
        from datetime import date as _d, datetime
        try:
            from zoneinfo import ZoneInfo
            _today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            _today = _d.today()

        st.markdown('<div class="sec-head"><span>📅 Date &amp; MG Input</span></div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns([1, 1, 2])
        with d1:
            sd = st.date_input("Date", value=_today, key=k("toast_date"))
        with d2:
            cmg = st.number_input("Client MG", min_value=1, step=1, value=L.default_mg, key=k("toast_cmg"))
        with d3:
            st.markdown("<br>", unsafe_allow_html=True)
            r5 = st.checkbox("Round to nearest 5", value=True, key=k("toast_r5"))

        adj = L.toasttab_adjust(cmg)
        vmg = mg5(adj) if r5 else int(adj)

        st.markdown(
            f'<div class="info-bar">Vendor MG for <strong>{sd.strftime("%A, %d %b %Y")}</strong></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Client MG", f"{cmg}")
        with m2:
            st.metric("Adjusted", f"{adj:.0f}")
        with m3:
            st.metric("Vendor MG", f"{vmg}")
        st.stop()

    # ── ENSURE MODEL ARTIFACTS ───────────────────────────────────────────
    def _ensure():
        if artifacts_exist(CK, L.encoder_columns):
            return
        ds = INFO.get("dataset")
        if not ds:
            st.error(f"No dataset configured for {sel}.")
            st.stop()
        if not os.path.exists(ds):
            st.error(f"Dataset file not found for {sel}: {ds}")
            st.stop()
        with st.spinner(f"Training model for {sel} — this may take a minute…"):
            try:
                _, rmse = train_model(CK, ds, L)
                clear_cache(CK)
                st.success(f"Model trained for {sel}. RMSE: {rmse:.4f}")
            except Exception as e:
                st.error(f"Training failed for {sel}: {e}")
                st.stop()

    _ensure()

    # ── LOAD MAPPINGS ────────────────────────────────────────────────────
    i2s    = load_map(CK, "item_to_subcat")
    c2s    = load_map(CK, "cat_to_subs")
    i2s_lc = {norm(k0): v for k0, v in i2s.items()}

    cfg = st.session_state.get(cfg_key, _default_client_config())
    configured_categories = cfg.get("menu_categories") or list(L.fixed_categories)
    configured_star        = {norm(x) for x in (cfg.get("star_categories") or [])}
    nonveg_mode            = cfg.get("nonveg_mode") or getattr(L, "custom_nonveg_mode", "Optional")
    slab_adjustments       = cfg.get("slab_adjustments") or []

    # ── DATE & CONTEXT ───────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span>📅 Date &amp; Context</span></div>', unsafe_allow_html=True)

    if L.has_special_day:
        dc1, dc2, dc3 = st.columns(3)
    else:
        dc1, dc2 = st.columns([1, 2])

    with dc1:
        sel_date = st.date_input("Service Date", key=k("sel_date"))
    day_name = sel_date.strftime("%A")
    day_norm = norm(day_name)
    month    = sel_date.strftime("%B")

    if L.has_special_day:
        with dc2:
            sdt = st.selectbox(
                "Day Type",
                ["Regular", "Previous Day of Holiday", "Next Day of Holiday", "Holiday"],
                key=k("day_type"),
            )
        with dc3:
            ht = st.selectbox(
                "Holiday Type",
                ["Not Applicable", "Non-Important Holiday", "Compulsory Holiday", "Important Holiday"],
                key=k("holiday_type"),
            )
    else:
        sdt = "Regular"
        ht  = "Not Applicable"

    cdt = L.canonicalize_day_type(sdt)
    cht = L.canonicalize_holiday_type(ht)

    # context info bar
    ctx_parts = [f"<strong>{day_name}</strong>, {sel_date.strftime('%d %b %Y')}"]
    if L.has_special_day and sdt != "Regular":
        ctx_parts.append(f"<strong>{sdt}</strong> &nbsp;·&nbsp; {ht}")
    st.markdown(
        f'<div class="info-bar">📆 {" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(ctx_parts)}</div>',
        unsafe_allow_html=True,
    )

    # ── NON-VEG STATE ────────────────────────────────────────────────────
    is_nv       = False
    meal_day    = "veg"
    nv_item     = None
    nv_cat      = None
    nv_rows     = []
    nv_mg_total = 0

    if nonveg_mode == "Required":
        is_nv    = True
        meal_day = "nonveg"
        st.info("Non-Veg is marked as **required** from Config Client.")
    elif nonveg_mode == "Optional" and L.has_nonveg_toggle:
        is_nv    = st.toggle("🍗 Non-Veg Day?", value=True, key=k("is_nv"))
        meal_day = "nonveg" if is_nv else "veg"

    entries = []
    menu    = []

    show_nv = nonveg_mode != "Not Needed" and (
        (L.has_nonveg_toggle and is_nv)
        or (not L.has_nonveg_toggle and L.nonveg_item_count > 0 and L.has_vendor_plan)
    )

    # ── NON-VEG ITEMS ────────────────────────────────────────────────────
    if show_nv:
        nv_opts = get_nv_cats(CK, INFO["dataset"], L) if INFO.get("dataset") else []
        if nv_opts:
            st.markdown('<div class="sec-head"><span>🍗 Non-Veg Items</span></div>', unsafe_allow_html=True)
            for ni in range(L.nonveg_item_count):
                nvc_col, nvi_col = st.columns(2)
                with nvc_col:
                    nvc = st.selectbox(
                        f"Non-Veg Category{f' #{ni + 1}' if L.nonveg_item_count > 1 else ''}",
                        nv_opts,
                        key=k(f"nvc_{ni}"),
                    )
                with nvi_col:
                    nvi = st.text_input(f"Non-Veg Item  ({nvc}):", key=k(f"nvi_{ni}"))

                if not nvi:
                    continue

                item_norm = norm(nvi)
                can_nvc   = L.canonicalize_category(nvc)

                if item_norm in i2s_lc:
                    sc = i2s_lc[item_norm]
                    st.caption(f"✅ Sub-category: {sc}")
                else:
                    opts = c2s.get(can_nvc, [])
                    sc = (
                        st.selectbox(f"Sub-cat for '{nvi}':", opts, key=k(f"nvsc_{ni}"))
                        if opts
                        else st.text_input(f"Sub-cat for '{nvi}':", key=k(f"nvsc_{ni}_t"))
                    )

                if L.has_nonveg_toggle:
                    item_mg        = st.number_input(
                        f"Client MG for '{nvi}':",
                        min_value=1, step=1, value=L.default_mg, key=k(f"nvmg_{ni}"),
                    )
                    needs_shared_mg = False
                else:
                    item_mg        = None
                    needs_shared_mg = True

                entries.append({
                    "item": nvi, "subcat": sc, "category": can_nvc,
                    "mg": item_mg, "display_category": can_nvc, "needs_shared_mg": needs_shared_mg,
                })
                menu.append(nvi)
                nv_rows.append({"item": nvi, "cat": can_nvc, "mg": (item_mg or 0)})
                nv_mg_total += (item_mg or 0)

    if nv_rows:
        nv_item = nv_rows[0]["item"]
        nv_cat  = nv_rows[0]["cat"]
        if not L.has_nonveg_toggle:
            is_nv    = True
            meal_day = "nonveg"
    else:
        nv_item = None
        nv_cat  = None

    # ── SHARED CLIENT MG ─────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span>👥 Meal Group Size</span></div>', unsafe_allow_html=True)
    mg_col, _ = st.columns([1, 3])
    with mg_col:
        cmg_input = st.number_input(
            "Shared Client MG",
            min_value=1, step=1, value=L.default_mg, key=k("shared_cmg"),
        )
    cmg = _apply_slab_adjustment(cmg_input, slab_adjustments)
    if cmg != cmg_input:
        st.caption(f"Slab adjusted MG: **{cmg:.1f}**  (base: {cmg_input})")

    # patch nonveg entries that use shared MG
    for e in entries:
        if e["needs_shared_mg"] or e["mg"] is None:
            e["mg"] = cmg

    # ── MENU ITEMS ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-head"><span>🍱 Menu Items</span></div>', unsafe_allow_html=True)
    star_ui  = configured_star or {"flavour rice", "flavoured rice", "veg curry", "veg gravy"}
    north_pp = None

    for cat in configured_categories:
        is_star = norm(cat) in star_ui
        lbl     = f"{'⭐ ' if is_star else ''}{cat}"
        item    = st.text_input(lbl, key=k(f"item_{kcat(cat)}"))
        if not item:
            continue

        menu.append(item)
        item_norm = norm(item)
        icat = L.category_display_map.get(cat, cat)
        cc   = L.canonicalize_category(icat)

        if cc == "salad":
            sc = "salad"
        elif item_norm in i2s_lc:
            sc = i2s_lc[item_norm]
            st.caption(f"✅ Sub-category: {sc}")
        else:
            opts = c2s.get(cc, [])
            sc = (
                st.selectbox(f"Sub-cat for '{item}' ({cat}):", opts, key=k(f"sc_{kcat(cat)}"))
                if opts
                else st.text_input(f"Sub-cat for '{item}':", key=k(f"sc_{kcat(cat)}_t"))
            )

        entries.append({
            "item": item,
            "subcat": sc,
            "category": cc,
            "mg": cmg,
            "display_category": cat,
            "needs_shared_mg": False,
        })

    if cfg.get("additional_requirements"):
        st.caption(f"ℹ️ {cfg['additional_requirements']}")

    # ── PREDICT BUTTON ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        predict_clicked = st.button("🔮  Predict", key=k("predict_btn"), use_container_width=True)

    # ── RESULTS ──────────────────────────────────────────────────────────
    if predict_clicked:
        st.markdown("<br>", unsafe_allow_html=True)
        date_str = sel_date.strftime("%a %d %b %Y")
        extra    = f" &nbsp;·&nbsp; {sdt}" if (L.has_special_day and sdt != "Regular") else ""
        st.markdown(
            f'<div class="info-bar">📋 Prediction Results &nbsp;·&nbsp; <strong>{sel}</strong>'
            f' &nbsp;·&nbsp; {date_str}{extra}</div>',
            unsafe_allow_html=True,
        )

        if not entries:
            st.warning("Add at least one menu item.")
            st.stop()

        results = []
        with st.spinner("Running predictions…"):
            for e in entries:
                item = e["item"]
                sc   = e["subcat"]
                cat  = e["category"]
                img  = e["mg"]
                dcat = e["display_category"]

                # Rippling: South Veg dry copies North's PP
                if dcat == "South Veg dry" and north_pp is not None:
                    pp, tq = north_pp
                elif norm(cat) == "salad":
                    pp = L.salad_per_pax
                    tq = pp * img
                else:
                    pr = predict(
                        CK, item, menu, img, sc, cat, day_norm, L,
                        day_type=cdt, holiday_type=cht, meal_day=meal_day,
                    )
                    if not pr.ok:
                        st.error(f"❌ {pr.error} Skipping '{item}'.")
                        continue
                    if pr.fallback:
                        st.warning(f"⚠️ '{item}' unseen — fallback via '{pr.fallback_item}'.")
                    pp = pr.per_pax
                    tq = pr.total_qty

                if dcat == "North Veg dry":
                    north_pp = (pp, tq)

                row = build_row(
                    pp, tq,
                    L.canonicalize_category(dcat) if dcat != cat else cat,
                    item, is_nv, nv_cat, L,
                )
                if dcat in ("North Veg dry", "South Veg dry"):
                    row["Category"] = dcat
                results.append(row)

        if not results:
            st.stop()

        df = pd.DataFrame(results)
        df["Total Qty"]  = df["Total Qty"].round(1)
        df["Vendor MG"]  = df["Vendor MG"].round(0)

        # ── CLIENT PLAN ──────────────────────────────────────────────────
        if L.fixed_pp_map:
            cp = fixed_pp_client_plan(df, L.fixed_pp_map, cmg)
        else:
            cp = client_plan(df)

        if L.has_nonveg_toggle and is_nv and nv_item:
            veg_mg = max(cmg - nv_mg_total, 0)
            mg_html = (
                f'<span class="mg-chip">Veg MG: {veg_mg:.0f}</span>'
                f'&nbsp;<span class="mg-chip-nv">Non-Veg MG: {nv_mg_total:.0f}</span>'
            )
        else:
            mg_html = f'<span class="mg-chip">Client MG: {cmg:.0f}</span>'

        st.markdown(
            f'<div class="plan-strip"><span class="plan-title">📋 Client Production Plan</span>{mg_html}</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(fmt_cols(cp, ["Client PP", "Total Qty"]), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── VENDOR PLAN ──────────────────────────────────────────────────
        if L.has_vendor_plan:
            vp, vmg, nvmg = vendor_plan(df, results, cmg, is_nv, nv_cat, L, weekday=day_norm)

            if L.has_separate_nonveg_mg and nvmg > 0:
                vmg_html = (
                    f'<span class="mg-chip">Veg MG: {vmg}</span>'
                    f'&nbsp;<span class="mg-chip-nv">Non-Veg MG: {nvmg}</span>'
                )
            elif L.vendor_mg_method == "day_based":
                vmg_html = f'<span class="mg-chip-gold">Day-Based MG: {vmg}</span>'
            else:
                vmg_html = f'<span class="mg-chip">Vendor MG: {vmg}</span>'

            st.markdown(
                f'<div class="plan-strip"><span class="plan-title">🤝 Vendor Production Plan</span>{vmg_html}</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(fmt_cols(vp, ["Vendor PP", "Ordered Qty"]), use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── AGGRESSIVE PLAN ──────────────────────────────────────────
            if L.has_aggressive_plan:
                if L.vendor_mg_method == "tekion_2group":
                    fv_mg  = _two_group_fv(results, is_nv, nv_cat, L)
                    avg_nv = gavg([r for r in results if cats_match(r["Category"], nv_cat)])
                    ag, at, an, av = aggressive_plan(vp, results, fv_mg, avg_nv, is_nv, nv_cat, L, method_groups=2)
                else:
                    ag, at, an, av = aggressive_plan(vp, results, vmg, 0, is_nv, nv_cat, L, method_groups=3)

                if L.has_separate_nonveg_mg and an > 0:
                    ag_html = (
                        f'<span class="mg-chip">Veg MG: {av}</span>'
                        f'&nbsp;<span class="mg-chip-nv">Non-Veg MG: {an}</span>'
                    )
                else:
                    ag_html = f'<span class="mg-chip-gold">Aggressive MG: {at}</span>'

                st.markdown(
                    f'<div class="plan-strip"><span class="plan-title">🚀 Aggressive Vendor Plan</span>{ag_html}</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(fmt_cols(ag, ["Vendor PP", "Ordered Qty"]), use_container_width=True, hide_index=True)
                st.markdown("<br>", unsafe_allow_html=True)

    # ── SPECIAL DAY (independent button) ─────────────────────────────────
    if L.has_special_day and sdt != "Regular":
        st.divider()
        sd_col, _ = st.columns([1, 4])
        with sd_col:
            apply_sd = st.button("🎯  Apply Special Day Logic", key=k("apply_special_day"), use_container_width=True)
        if apply_sd:
            vmg_sd, pct = special_day_mg(cmg, sdt, ht, day_name, L)
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Reduction", f"{pct}%")
            with s2:
                st.metric("Adjusted Vendor MG", f"{vmg_sd:.0f}")


with tab2:
    _render_generate_production_plan_tab()
