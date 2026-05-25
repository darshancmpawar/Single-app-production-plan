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
from ml_core import (
    artifacts_exist, load_map, norm, fmt_cols,
    cats_match, get_nv_cats, predict,
)
from planner import (
    build_row, client_plan, fixed_pp_client_plan, vendor_plan,
    aggressive_plan, special_day_mg, avg_vendor_mg, classify_rows, round_to_nearest_5,
)
from concurrency import predict_gate, GateBusy, MAX_CONCURRENT_PREDICTS

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# ═══════════════ PAGE CONFIG — must be the first Streamlit call ═══════════════
st.set_page_config(
    page_title="Production Planner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════ CSS ═══════════════
with open(os.path.join(os.path.dirname(__file__), "styles.css")) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)



# ── helper (tekion 2-group final veg MG) ──
def _two_group_fv(results, is_nv, nv_cat, logic):
    _, star_rows, rest_rows = classify_rows(results, nv_cat, logic.star_categories)
    raw = (avg_vendor_mg(star_rows) + avg_vendor_mg(rest_rows)) / 2
    return raw if is_nv else max(logic.adjust_vendor_mg(raw), 0)


# ═══════════════ HEADER ═══════════════
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">🍽️</div>
    <div class="app-header-text">
        <div class="app-title">Per Pax Production Planner</div>
        <div class="app-sub">Cafeteria AI &nbsp;·&nbsp; Real-time menu intelligence &amp; operational planning</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════ CLIENT SELECTOR ═══════════════
sel_col, meta_col = st.columns([2, 3])
with sel_col:
    sel = st.selectbox("Select Client", CLIENT_LIST, key="client_sel")

CK   = name_to_key(sel)
INFO = get_info(CK)
L    = get_logic(CK)

with meta_col:
    if L.has_embeddings:
        badge_cls, badge_label = "badge-blue", "Embedding Model"
    else:
        badge_cls, badge_label = "badge-amber", "Multiplier Only"
    st.markdown(f"""
    <div class="client-meta">
        <span class="badge {badge_cls}">{badge_label}</span>
        <span class="client-name">Active: <strong>{sel}</strong></span>
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


def key(name: str) -> str:
    """Return a Streamlit widget key namespaced to the active client."""
    return f"{CK}::{name}"


def cat_key(cat: str) -> str:
    """Return a safe Streamlit widget key derived from a category name."""
    return norm(str(cat)).replace(" ", "_").replace("/", "_").replace("-", "_")


# ═══════════════════════════════════════════════════════
#  GENERATE PRODUCTION PLAN
# ═══════════════════════════════════════════════════════
def _render_generate_production_plan_tab():

    # ── TOASTTAB (multiplier-only) ──────────────────────────────────────
    if not L.has_embeddings:
        from datetime import date as _d, datetime
        try:
            from zoneinfo import ZoneInfo
            _today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            _today = _d.today()

        st.markdown('<div class="sec"><span class="sec-icon">📅</span><span class="sec-label">Date &amp; MG Input</span></div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns([1, 1, 2])
        with d1:
            sd = st.date_input("Date", value=_today, key=key("toast_date"))
        with d2:
            cmg = st.number_input("Client MG", min_value=1, step=1, value=L.default_mg, key=key("toast_cmg"))
        with d3:
            st.markdown("<br>", unsafe_allow_html=True)
            r5 = st.checkbox("Round to nearest 5", value=True, key=key("toast_r5"))

        adj = L.toasttab_adjust(cmg)
        vmg = round_to_nearest_5(adj) if r5 else int(adj)

        st.markdown(
            f'<div class="ctx">📆 &nbsp;<strong>{sd.strftime("%A, %d %b %Y")}</strong> &nbsp;·&nbsp; Vendor MG result</div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        with m1:  st.metric("Client MG",  f"{cmg}")
        with m2:  st.metric("Adjusted",   f"{adj:.0f}")
        with m3:  st.metric("Vendor MG",  f"{vmg}")
        st.stop()

    # ── ENSURE MODEL ARTIFACTS ──────────────────────────────────────────
    # The runtime never trains. Artifacts are produced offline by
    # `scripts/train.py` (run locally or via the Train Models GitHub Action).
    # If they're missing here, fail fast with a clear repair path.
    def _ensure():
        if artifacts_exist(CK, L.encoder_columns):
            return
        st.error(
            f"Model artifacts for **{sel}** are missing.\n\n"
            "They are built offline by the **Train Models** GitHub Actions "
            "workflow. Trigger it via **Actions → Train Models → Run "
            f"workflow** (client: `{CK}`), wait for the auto-commit, then "
            "refresh this page."
        )
        with st.expander("Expected artifact files"):
            expected = [
                f"artifacts/{CK}_per_pax_tf_model.keras",
                f"artifacts/{CK}_item_to_subcat.pkl",
                f"artifacts/{CK}_item_to_cat.pkl",
                f"artifacts/{CK}_cat_to_subs.pkl",
            ] + [f"artifacts/{CK}_{c}_encoder.pkl" for c in L.encoder_columns]
            st.code("\n".join(expected))
        st.stop()

    _ensure()

    # ── LOAD MAPPINGS ───────────────────────────────────────────────────
    i2s    = load_map(CK, "item_to_subcat")
    c2s    = load_map(CK, "cat_to_subs")
    i2s_lc = {norm(k0): v for k0, v in i2s.items()}

    configured_categories = list(L.fixed_categories)
    configured_star       = {norm(x) for x in L.star_categories}
    nonveg_mode           = getattr(L, "custom_nonveg_mode", "Optional" if L.has_nonveg_toggle else "Not Needed")

    # ── DATE & CONTEXT ──────────────────────────────────────────────────
    st.markdown('<div class="sec"><span class="sec-icon">📅</span><span class="sec-label">Date &amp; Context</span></div>', unsafe_allow_html=True)

    if L.has_special_day:
        dc1, dc2, dc3 = st.columns(3)
    else:
        dc1, dc2 = st.columns([1, 2])

    with dc1:
        sel_date = st.date_input("Service Date", key=key("sel_date"))
    day_name = sel_date.strftime("%A")
    day_norm = norm(day_name)
    month    = sel_date.strftime("%B")

    if L.has_special_day:
        with dc2:
            sdt = st.selectbox(
                "Day Type",
                ["Regular", "Previous Day of Holiday", "Next Day of Holiday", "Holiday"],
                key=key("day_type"),
            )
        with dc3:
            ht = st.selectbox(
                "Holiday Type",
                ["Not Applicable", "Non-Important Holiday", "Compulsory Holiday", "Important Holiday"],
                key=key("holiday_type"),
            )
    else:
        sdt = "Regular"
        ht  = "Not Applicable"

    cdt = L.canonicalize_day_type(sdt)
    cht = L.canonicalize_holiday_type(ht)

    # context pill
    ctx_parts = [f"<strong>{day_name}</strong>, {sel_date.strftime('%d %b %Y')}"]
    if L.has_special_day and sdt != "Regular":
        ctx_parts.append(f"<strong>{sdt}</strong> &nbsp;·&nbsp; {ht}")
    st.markdown(
        f'<div class="ctx">📆 &nbsp;{"&emsp;|&emsp;".join(ctx_parts)}</div>',
        unsafe_allow_html=True,
    )

    # ── NON-VEG STATE ───────────────────────────────────────────────────
    is_nv       = False
    meal_day    = "veg"
    nv_item     = None
    nv_cat      = None
    nv_rows     = []
    nv_mg_total = 0

    if nonveg_mode == "Required":
        is_nv    = True
        meal_day = "nonveg"
    elif nonveg_mode == "Optional" and L.has_nonveg_toggle:
        is_nv    = st.toggle("🍗 Non-Veg Day?", value=True, key=key("is_nv"))
        meal_day = "nonveg" if is_nv else "veg"

    entries = []
    menu    = []

    show_nv = nonveg_mode != "Not Needed" and (
        (L.has_nonveg_toggle and is_nv)
        or (not L.has_nonveg_toggle and L.nonveg_item_count > 0 and L.has_vendor_plan)
    )

    # ── NON-VEG ITEMS ───────────────────────────────────────────────────
    if show_nv:
        nv_opts = get_nv_cats(CK, INFO["dataset"], L) if INFO.get("dataset") else []
        if nv_opts:
            st.markdown('<div class="sec"><span class="sec-icon">🍗</span><span class="sec-label">Non-Veg Items</span></div>', unsafe_allow_html=True)
            is_biryani_day = False
            if L.has_biryani_toggle:
                is_biryani_day = st.toggle("🍛 Biryani Day?", value=False, key=key("is_biryani_day"))
            nv_count = 1 if is_biryani_day else L.nonveg_item_count
            for ni in range(nv_count):
                nvc_col, nvi_col = st.columns(2)
                with nvc_col:
                    if is_biryani_day:
                        nvc = "non veg biryani"
                        st.text_input("Non-Veg Category", value="Non Veg Biryani", disabled=True, key=key("nvc_biryani_label"))
                    else:
                        nvc = st.selectbox(
                            f"Non-Veg Category{f' #{ni + 1}' if L.nonveg_item_count > 1 else ''}",
                            nv_opts,
                            key=key(f"nvc_{ni}"),
                        )
                with nvi_col:
                    nvi = st.text_input(f"Non-Veg Item  ({nvc}):", key=key(f"nvi_{ni}"))

                if not nvi:
                    continue

                item_norm = norm(nvi)
                can_nvc   = L.canonicalize_category(nvc)

                if item_norm in i2s_lc:
                    sc = i2s_lc[item_norm]
                    st.caption(f"✓ Sub-category: {sc}")
                else:
                    opts = c2s.get(can_nvc, [])
                    sc = (
                        st.selectbox(f"Sub-cat for '{nvi}':", opts, key=key(f"nvsc_{ni}"))
                        if opts else st.text_input(f"Sub-cat for '{nvi}':", key=key(f"nvsc_{ni}_t"))
                    )

                if L.has_nonveg_toggle:
                    item_mg         = st.number_input(
                        f"Client MG for '{nvi}':",
                        min_value=1, step=1, value=L.default_mg, key=key(f"nvmg_{ni}"),
                    )
                    needs_shared_mg = False
                else:
                    item_mg         = None
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

    # ── SHARED CLIENT MG ────────────────────────────────────────────────
    st.markdown('<div class="sec"><span class="sec-icon">👥</span><span class="sec-label">Meal Group Size</span></div>', unsafe_allow_html=True)
    mg_col, _ = st.columns([1, 3])
    with mg_col:
        cmg_input = st.number_input(
            "Shared Client MG",
            min_value=1, step=1, value=L.default_mg, key=key("shared_cmg"),
        )
    cmg = cmg_input

    # patch nonveg entries that use shared MG
    for e in entries:
        if e["needs_shared_mg"] or e["mg"] is None:
            e["mg"] = cmg

    # ── MENU ITEMS ──────────────────────────────────────────────────────
    st.markdown('<div class="sec"><span class="sec-icon">🍱</span><span class="sec-label">Menu Items</span></div>', unsafe_allow_html=True)
    star_ui  = configured_star or {"flavour rice", "flavoured rice", "veg curry", "veg gravy"}
    north_pp = None

    for cat in configured_categories:
        is_star = norm(cat) in star_ui
        lbl     = f"{'⭐ ' if is_star else ''}{cat}"
        item    = st.text_input(lbl, key=key(f"item_{cat_key(cat)}"))
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
            st.caption(f"✓ Sub-category: {sc}")
        else:
            opts = c2s.get(cc, [])
            sc = (
                st.selectbox(f"Sub-cat for '{item}' ({cat}):", opts, key=key(f"sc_{cat_key(cat)}"))
                if opts else st.text_input(f"Sub-cat for '{item}':", key=key(f"sc_{cat_key(cat)}_t"))
            )

        entries.append({
            "item": item, "subcat": sc, "category": cc,
            "mg": cmg, "display_category": cat, "needs_shared_mg": False,
        })

    # ── PREDICT BUTTON ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        predict_clicked = st.button("Predict", key=key("predict_btn"), use_container_width=True)

    # ── RESULTS ─────────────────────────────────────────────────────────
    if predict_clicked:
        st.markdown("<br>", unsafe_allow_html=True)
        date_str = sel_date.strftime("%a %d %b %Y")
        extra    = f" &nbsp;·&nbsp; {sdt}" if (L.has_special_day and sdt != "Regular") else ""
        st.markdown(
            f'<div class="ctx">📋 &nbsp;<strong>{sel}</strong> &nbsp;·&nbsp; {date_str}{extra}</div>',
            unsafe_allow_html=True,
        )

        if not entries:
            st.warning("Add at least one menu item.")
            st.stop()

        # predict_gate caps simultaneous predict loops process-wide so the
        # 1-CPU / 1-GB Streamlit Cloud tier doesn't OOM under concurrent users.
        # Excess sessions queue here briefly; on timeout they get a clean retry msg.
        # We hold the gate ONLY across the predict loop, not the plan-render
        # below, so non-ML rendering doesn't keep slots occupied.
        results = []
        try:
            with predict_gate():
                with st.spinner("Running predictions…"):
                    for e in entries:
                        item = e["item"]
                        sc   = e["subcat"]
                        cat  = e["category"]
                        img  = e["mg"]
                        dcat = e["display_category"]

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
                                st.error(f"❌ {pr.error}  Skipping '{item}'.")
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
        except GateBusy as ge:
            st.warning(str(ge))
            st.stop()

        if not results:
            st.stop()

        df = pd.DataFrame(results)
        df["Total Qty"] = df["Total Qty"].round(1)
        df["Vendor MG"] = df["Vendor MG"].round(0)

        # ── CLIENT PLAN ─────────────────────────────────────────────────
        if L.fixed_pp_map:
            cp = fixed_pp_client_plan(df, L.fixed_pp_map, cmg)
        else:
            cp = client_plan(df)

        if L.has_nonveg_toggle and is_nv and nv_item:
            veg_mg   = max(cmg - nv_mg_total, 0)
            mg_chips = (
                f'<span class="chip chip-blue">Veg MG: {veg_mg:.0f}</span>'
                f'<span class="chip chip-purple">Non-Veg MG: {nv_mg_total:.0f}</span>'
            )
        else:
            mg_chips = f'<span class="chip chip-blue">Client MG: {cmg:.0f}</span>'

        st.markdown(
            f'<div class="plan-card"><span class="plan-label">📋 Client Production Plan</span>'
            f'<div class="plan-chips">{mg_chips}</div></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(fmt_cols(cp, ["Client PP", "Total Qty"]), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── VENDOR PLAN ─────────────────────────────────────────────────
        if L.has_vendor_plan:
            vp, vmg, nvmg = vendor_plan(df, results, cmg, is_nv, nv_cat, L, weekday=day_norm)

            if L.has_separate_nonveg_mg and nvmg > 0:
                vmg_chips = (
                    f'<span class="chip chip-blue">Veg MG: {vmg}</span>'
                    f'<span class="chip chip-purple">Non-Veg MG: {nvmg}</span>'
                )
            elif L.vendor_mg_method == "day_based":
                vmg_chips = f'<span class="chip chip-amber">Day-Based MG: {vmg}</span>'
            else:
                vmg_chips = f'<span class="chip chip-blue">Vendor MG: {vmg}</span>'

            st.markdown(
                f'<div class="plan-card plan-card-vendor"><span class="plan-label">🤝 Vendor Production Plan</span>'
                f'<div class="plan-chips">{vmg_chips}</div></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(fmt_cols(vp, ["Vendor PP", "Ordered Qty"]), use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── AGGRESSIVE PLAN ──────────────────────────────────────────
            if L.has_aggressive_plan:
                if L.vendor_mg_method == "tekion_2group":
                    fv_mg  = _two_group_fv(results, is_nv, nv_cat, L)
                    avg_nv = avg_vendor_mg([r for r in results if cats_match(r["Category"], nv_cat)])
                    ag, at, an, av = aggressive_plan(vp, results, fv_mg, avg_nv, is_nv, nv_cat, L, method_groups=2)
                else:
                    ag, at, an, av = aggressive_plan(vp, results, vmg, 0, is_nv, nv_cat, L, method_groups=3)

                if L.has_separate_nonveg_mg and an > 0:
                    ag_chips = (
                        f'<span class="chip chip-blue">Veg MG: {av}</span>'
                        f'<span class="chip chip-purple">Non-Veg MG: {an}</span>'
                    )
                else:
                    ag_chips = f'<span class="chip chip-amber">Aggressive MG: {at}</span>'

                st.markdown(
                    f'<div class="plan-card plan-card-aggressive"><span class="plan-label">🚀 Aggressive Vendor Plan</span>'
                    f'<div class="plan-chips">{ag_chips}</div></div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(fmt_cols(ag, ["Vendor PP", "Ordered Qty"]), use_container_width=True, hide_index=True)
                st.markdown("<br>", unsafe_allow_html=True)

    # ── SPECIAL DAY (independent button) ────────────────────────────────
    if L.has_special_day and sdt != "Regular":
        st.divider()
        sd_col, _ = st.columns([1, 4])
        with sd_col:
            apply_sd = st.button("Apply Special Day Logic", key=key("apply_special_day"), use_container_width=True)
        if apply_sd:
            vmg_sd, pct = special_day_mg(cmg, sdt, ht, day_name, L)
            s1, s2 = st.columns(2)
            with s1:  st.metric("Reduction",          f"{pct}%")
            with s2:  st.metric("Adjusted Vendor MG", f"{vmg_sd:.0f}")


_render_generate_production_plan_tab()
