"""
UI — Streamlit app with client dropdown. Dynamic UI per client capabilities.
"""
import os
import warnings
import logging
import re

import pandas as pd
import streamlit as st

from client_database import name_to_key, get_info, get_client_list
from client_logic import get_logic
from Logic_Definer import save_client_configuration
from styling import apply_app_styling
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


# helper for any client using the "tekion_2group" vendor MG method
def _two_group_fv(results, is_nv, nv_cat, logic):
    _, st2, rest = classify(results, nv_cat, logic.star_categories)
    raw = (gavg(st2) + gavg(rest)) / 2
    return raw if is_nv else max(logic.adjust_vendor_mg(raw), 0)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-")


# ═══════════════ PAGE LAYOUT ═══════════════
st.set_page_config(page_title="Production Plan", layout="wide")
st.title("Per Pax Quantity & Production Plan Prediction")

apply_app_styling()


tab1, tab2 = st.tabs(["Config New Client", "Generate Production Plan"])

with tab1:
    st.markdown("<div class='config-card'>", unsafe_allow_html=True)
    st.subheader("Configure New Client")
    st.caption("Tab 1 is only for defining new client configuration.")

    client_name = st.text_input("Client Name", key="new_cfg_client_name").strip()
    auto_slug = _slugify(client_name)
    st.text_input("Client Key (slug)", value=auto_slug, disabled=True, help="Auto-generated from Client Name")

    cfg = {
        "client_key": auto_slug,
        "client_name": client_name,
        "menu_categories": st.multiselect(
            "Menu Category's",
            options=[
                "Flavour Rice", "Indian Bread", "White Rice", "Veg Dry", "Veg Curry", "Dal", "Sambar", "Rasam", "Salad",
            ],
            key="new_cfg_menu_categories",
        ),
        "nonveg_mode": st.selectbox(
            "Non-Veg Switch Required?",
            ["Required", "Optional", "Not Needed"],
            key="new_cfg_nonveg_mode",
        ),
        "star_categories": st.multiselect(
            "Star Item List",
            options=st.session_state.get("new_cfg_menu_categories", []),
            key="new_cfg_star_categories",
        ),
        "additional_requirements": st.text_area(
            "Another requirement input to configure a client",
            key="new_cfg_additional_requirements",
        ),
        "custom_bump_pct": st.number_input(
            "Custom Bump % (for aggressive plan)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            key="new_cfg_custom_bump_pct",
        ),
    }

    st.markdown("#### Slab wise adjustment")
    slab_edit = st.data_editor(
        pd.DataFrame(columns=["min_mg", "max_mg", "adjustment_pct"]),
        num_rows="dynamic",
        use_container_width=True,
        key="new_cfg_slab_editor",
    )
    cfg["slab_adjustments"] = [
        {
            "min_mg": float(r["min_mg"]),
            "max_mg": float(r["max_mg"]),
            "adjustment_pct": float(r["adjustment_pct"]),
        }
        for _, r in slab_edit.dropna(subset=["min_mg", "max_mg", "adjustment_pct"]).iterrows()
    ]

    if st.button("Save Client Config", key="new_cfg_save_btn"):
        if not cfg["client_key"]:
            st.error("Client Key is required.")
        elif not cfg["client_name"]:
            st.error("Client Name is required.")
        elif not cfg["menu_categories"]:
            st.error("At least one menu category is required.")
        else:
            save_client_configuration(cfg["client_key"], cfg)
            st.success("New client configuration saved in modular override files.")

    st.markdown("</div>", unsafe_allow_html=True)


def _apply_slab_adjustment(base_mg: float, slabs: list[dict]) -> float:
    mg = float(base_mg)
    for slab in slabs:
        lo = slab.get("min_mg")
        hi = slab.get("max_mg")
        pct = slab.get("adjustment_pct")
        if lo is None or hi is None or pct is None:
            continue
        if lo <= mg <= hi:
            return max(mg + (mg * pct / 100.0), 1.0)
    return mg


# ═══════════════ TOASTTAB (multiplier-only, separate UI) ═══════════════
def _render_generate_production_plan_tab():
    sel = st.selectbox("Select Client", get_client_list(), key="client_sel")
    CK = name_to_key(sel)
    INFO = get_info(CK)
    L = get_logic(CK)

    db_mode = INFO.get("has_embeddings", None)
    if db_mode is not None and bool(db_mode) != bool(L.has_embeddings):
        st.error(
            f"Config mismatch for {sel}: "
            f"client_database.has_embeddings={db_mode} vs client_logic.has_embeddings={L.has_embeddings}. "
            f"Please align configs."
        )
        st.stop()

    def k(name: str) -> str:
        return f"{CK}::{name}"

    def kcat(cat: str) -> str:
        return norm(str(cat)).replace(" ", "_").replace("/", "_").replace("-", "_")

    st.caption(f"Client: **{sel}** | Mode: **{'Embedding' if L.has_embeddings else 'Multiplier-only'}**")

    def _default_client_config():
        return {
            "client_name": sel,
            "menu_categories": list(L.fixed_categories),
            "nonveg_mode": getattr(L, "custom_nonveg_mode", ("Optional" if L.has_nonveg_toggle else "Not Needed")),
            "star_categories": list(L.star_categories),
            "slab_adjustments": list(getattr(L, "slab_adjustments", [])),
            "additional_requirements": getattr(L, "additional_requirements", ""),
            "custom_bump_pct": float(getattr(L, "custom_bump_pct", 0.0)),
        }

    cfg_key = k("client_config")
    if cfg_key not in st.session_state:
        st.session_state[cfg_key] = _default_client_config()
    if not L.has_embeddings:
        from datetime import date as _d, datetime
    
        try:
            from zoneinfo import ZoneInfo
    
            _today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            _today = _d.today()
    
        c1, c2 = st.columns(2)
        with c1:
            sd = st.date_input("Date", value=_today, key=k("toast_date"))
        with c2:
            cmg = st.number_input("Client MG", min_value=1, step=1, value=L.default_mg, key=k("toast_cmg"))
    
        r5 = st.checkbox("Round to nearest 5", value=True, key=k("toast_r5"))
        adj = L.toasttab_adjust(cmg)
        vmg = mg5(adj) if r5 else int(adj)
    
        st.divider()
        st.subheader(f"Vendor MG for {sd:%A, %d %b %Y}")
        a, b, c = st.columns(3)
        with a:
            st.metric("Client MG", f"{cmg}")
        with b:
            st.metric("Adjusted", f"{adj:.0f}")
        with c:
            st.metric("Vendor MG", f"{vmg}")
        st.stop()
    
    
    # ═══════════════ ENSURE ARTIFACTS ═══════════════
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
    
        st.warning(f"Training model for {sel}...")
        try:
            _, rmse = train_model(CK, ds, L)
            clear_cache(CK)
            st.success(f"Model trained for {sel}. RMSE: {rmse:.4f}")
        except Exception as e:
            st.error(f"Training failed for {sel}: {e}")
            st.stop()
    
    
    _ensure()
    
    
    # ═══════════════ LOAD MAPPINGS ═══════════════
    i2s = load_map(CK, "item_to_subcat")
    c2s = load_map(CK, "cat_to_subs")
    i2s_lc = {norm(k0): v for k0, v in i2s.items()}
    
    
    cfg = st.session_state.get(cfg_key, _default_client_config())
    configured_categories = cfg.get("menu_categories") or list(L.fixed_categories)
    configured_star = {norm(x) for x in (cfg.get("star_categories") or [])}
    nonveg_mode = cfg.get("nonveg_mode") or getattr(L, "custom_nonveg_mode", "Optional")
    slab_adjustments = cfg.get("slab_adjustments") or []

    # ═══════════════ INPUTS ═══════════════
    sel_date = st.date_input("Select today's date:", key=k("sel_date"))
    day_name = sel_date.strftime("%A")
    day_norm = norm(day_name)
    month = sel_date.strftime("%B")
    
    # Day type / holiday (if client supports special day)
    if L.has_special_day:
        sdt = st.selectbox(
            "Day Type",
            ["Regular", "Previous Day of Holiday", "Next Day of Holiday", "Holiday"],
            key=k("day_type"),
        )
        ht = st.selectbox(
            "Holiday Type",
            ["Not Applicable", "Non-Important Holiday", "Compulsory Holiday", "Important Holiday"],
            key=k("holiday_type"),
        )
    else:
        sdt = "Regular"
        ht = "Not Applicable"
    
    cdt = L.canonicalize_day_type(sdt)
    cht = L.canonicalize_holiday_type(ht)
    
    # Nonveg toggle/state
    is_nv = False
    meal_day = "veg"
    nv_item = None
    nv_cat = None
    nv_rows = []
    nv_mg_total = 0
    
    if nonveg_mode == "Required":
        is_nv = True
        meal_day = "nonveg"
        st.info("Non-Veg is marked as required from Config Client.")
    elif nonveg_mode == "Optional" and L.has_nonveg_toggle:
        is_nv = st.toggle("🍗 Non-Veg Day?", value=True, key=k("is_nv"))
        meal_day = "nonveg" if is_nv else "veg"
    
    entries = []  # each entry: {"item","subcat","category","mg","display_category","needs_shared_mg"}
    menu = []
    
    # Nonveg item(s) — shown if toggle on OR if client always has nonveg
    show_nv = nonveg_mode != "Not Needed" and ((L.has_nonveg_toggle and is_nv) or (
        not L.has_nonveg_toggle and L.nonveg_item_count > 0 and L.has_vendor_plan
    ))
    
    if show_nv:
        nv_opts = get_nv_cats(CK, INFO["dataset"], L) if INFO.get("dataset") else []
        if nv_opts:
            for ni in range(L.nonveg_item_count):
                nvc = st.selectbox(
                    f"Non-Veg Category{f' #{ni + 1}' if L.nonveg_item_count > 1 else ''}",
                    nv_opts,
                    key=k(f"nvc_{ni}"),
                )
                nvi = st.text_input(f"Non-Veg Item for {nvc}:", key=k(f"nvi_{ni}"))
    
                if not nvi:
                    continue
    
                item_norm = norm(nvi)
                can_nvc = L.canonicalize_category(nvc)
    
                if item_norm in i2s_lc:
                    sc = i2s_lc[item_norm]
                    st.text(f"✅ Sub-category: {sc}")
                else:
                    opts = c2s.get(can_nvc, [])
                    sc = (
                        st.selectbox(f"Sub-cat for '{nvi}':", opts, key=k(f"nvsc_{ni}"))
                        if opts
                        else st.text_input(f"Sub-cat for '{nvi}':", key=k(f"nvsc_{ni}_t"))
                    )
    
                if L.has_nonveg_toggle:
                    item_mg = st.number_input(
                        f"Client MG for '{nvi}':",
                        min_value=1,
                        step=1,
                        value=L.default_mg,
                        key=k(f"nvmg_{ni}"),
                    )
                    needs_shared_mg = False
                else:
                    item_mg = None
                    needs_shared_mg = True
    
                entries.append(
                    {
                        "item": nvi,
                        "subcat": sc,
                        "category": can_nvc,
                        "mg": item_mg,
                        "display_category": can_nvc,
                        "needs_shared_mg": needs_shared_mg,
                    }
                )
                menu.append(nvi)
    
                nv_rows.append({"item": nvi, "cat": can_nvc, "mg": (item_mg or 0)})
                nv_mg_total += (item_mg or 0)
    
    if nv_rows:
        nv_item = nv_rows[0]["item"]  # primary NV item for UI text
        nv_cat = nv_rows[0]["cat"]  # primary NV category for grouping logic

        # if client has no toggle but NV items are entered, mark as nonveg day
        if not L.has_nonveg_toggle:
            is_nv = True
            meal_day = "nonveg"
    
    
    cmg_input = st.number_input("Shared Client MG:", min_value=1, step=1, value=L.default_mg, key=k("shared_cmg"))
    cmg = _apply_slab_adjustment(cmg_input, slab_adjustments)
    if cmg != cmg_input:
        st.caption(f"Slab adjusted Client MG: {cmg:.1f} (base: {cmg_input})")
    
    # patch nonveg entries that need shared MG
    for e in entries:
        if e["needs_shared_mg"] or e["mg"] is None:
            e["mg"] = cmg
    
    
    st.subheader("Enter Menu Items by Category")
    star_ui = configured_star or {"flavour rice", "flavoured rice", "veg curry", "veg gravy"}
    north_pp = None  # for Rippling
    
    for cat in configured_categories:
        lbl = f"Item name for {cat}:"
        if norm(cat) in star_ui:
            lbl = f"⭐ {lbl}"
    
        item = st.text_input(lbl, key=k(f"item_{kcat(cat)}"))
        if not item:
            continue
    
        menu.append(item)
        item_norm = norm(item)
    
        icat = L.category_display_map.get(cat, cat)  # display -> internal
        cc = L.canonicalize_category(icat)
    
        if cc == "salad":
            sc = "salad"
        elif item_norm in i2s_lc:
            sc = i2s_lc[item_norm]
            st.text(f"✅ Sub-category: {sc}")
        else:
            opts = c2s.get(cc, [])
            sc = (
                st.selectbox(f"Sub-cat for '{item}' ({cat}):", opts, key=k(f"sc_{kcat(cat)}"))
                if opts
                else st.text_input(f"Sub-cat for '{item}':", key=k(f"sc_{kcat(cat)}_t"))
            )
    
        entries.append(
            {
                "item": item,
                "subcat": sc,
                "category": cc,             # internal canonical category
                "mg": cmg,
                "display_category": cat,    # UI/display category
                "needs_shared_mg": False,
            }
        )
    
    
    if cfg.get("additional_requirements"):
        st.caption(f"Additional requirement: {cfg['additional_requirements']}")

    # ═══════════════ PREDICT ═══════════════
    if st.button("Predict", key=k("predict_btn")):
        st.markdown(f"### Prediction Results — {sel}")
        st.markdown(f"**Date:** {sel_date} | **Day:** {day_name} | **Month:** {month}")
        if L.has_special_day:
            st.markdown(f"**Day Type:** {sdt} | **Holiday Type:** {ht}")
        st.markdown("---")
    
        if not entries:
            st.warning("Add at least one menu item.")
            st.stop()
    
        results = []
        for e in entries:
            item = e["item"]
            sc = e["subcat"]
            cat = e["category"]               # internal category
            img = e["mg"]
            dcat = e["display_category"]      # display category
    
            # Rippling: South Veg dry copies North's PP
            if dcat == "South Veg dry" and north_pp is not None:
                pp, tq = north_pp
            elif norm(cat) == "salad":
                pp = L.salad_per_pax
                tq = pp * img
            else:
                pr = predict(
                    CK,
                    item,
                    menu,
                    img,
                    sc,
                    cat,
                    day_norm,
                    L,
                    day_type=cdt,
                    holiday_type=cht,
                    meal_day=meal_day,
                )
                if not pr.ok:
                    st.error(f"❌ {pr.error} Skipping '{item}'.")
                    continue
                if pr.fallback:
                    st.warning(f"⚠️ '{item}' unseen, fallback '{pr.fallback_item}'.")
                pp = pr.per_pax
                tq = pr.total_qty
    
            if dcat == "North Veg dry":
                north_pp = (pp, tq)
    
            row = build_row(
                pp,
                tq,
                L.canonicalize_category(dcat) if dcat != cat else cat,
                item,
                is_nv,
                nv_cat,
                L,
            )
    
            # keep display category for Rippling
            if dcat in ("North Veg dry", "South Veg dry"):
                row["Category"] = dcat
    
            results.append(row)
    
        if not results:
            st.stop()
    
        df = pd.DataFrame(results)
        df["Total Qty"] = df["Total Qty"].round(1)
        df["Vendor MG"] = df["Vendor MG"].round(0)
    
        # ── CLIENT PLAN ──
        if L.fixed_pp_map:
            cp = fixed_pp_client_plan(df, L.fixed_pp_map, cmg)
        else:
            cp = client_plan(df)
    
        st.markdown("#### 📋 Client Production Plan")
        if L.has_nonveg_toggle and is_nv and nv_item:
            st.write(f"**Veg MG: {max(cmg - nv_mg_total, 0)} | Non-Veg MG: {nv_mg_total}**")
        else:
            st.write(f"**Client MG: {cmg}")
    
        st.table(fmt_cols(cp, ["Client PP", "Total Qty"]))
        st.markdown("---")
    
        # ── VENDOR PLAN ──
        if L.has_vendor_plan:
            vp, vmg, nvmg = vendor_plan(df, results, cmg, is_nv, nv_cat, L, weekday=day_norm)
    
            st.markdown("#### 🤝 Vendor Production Plan")
            if L.has_separate_nonveg_mg and nvmg > 0:
                st.write(f"**Veg Vendor MG: {vmg} | Non-Veg Vendor MG: {nvmg}**")
            else:
                label = "Vendor MG (day-based)" if L.vendor_mg_method == "day_based" else "Vendor MG"
                st.write(f"**{label}: {vmg}")
    
            st.table(fmt_cols(vp, ["Vendor PP", "Ordered Qty"]))
            st.markdown("---")
    
            # ── AGGRESSIVE PLAN ──
            if L.has_aggressive_plan:
                if L.vendor_mg_method == "tekion_2group":
                    fv_mg = _two_group_fv(results, is_nv, nv_cat, L)
                    avg_nv = gavg([r for r in results if cats_match(r["Category"], nv_cat)])
                    ag, at, an, av = aggressive_plan(vp, results, fv_mg, avg_nv, is_nv, nv_cat, L, method_groups=2)
                elif L.vendor_mg_method in ("3group", "day_based"):
                    ag, at, an, av = aggressive_plan(vp, results, vmg, 0, is_nv, nv_cat, L, method_groups=3)
                else:
                    ag, at, an, av = aggressive_plan(vp, results, vmg, 0, is_nv, nv_cat, L, method_groups=3)
    
                st.markdown("#### 🚀 Aggressive Vendor Production Plan")
                if L.has_separate_nonveg_mg and an > 0:
                    st.write(f"**Veg Aggressive MG: {av} | Non-Veg Aggressive MG: {an}**")
                else:
                    st.write(f"**Adjusted Aggressive MG: {at}")
    
                st.table(fmt_cols(ag, ["Vendor PP", "Ordered Qty"]))
                st.markdown("---")
    
    
    # ── SPECIAL DAY (separate button, independent of Predict click) ──
    if L.has_special_day and sdt != "Regular":
        if st.button("Apply Special Day Logic", key=k("apply_special_day")):
            vmg_sd, pct = special_day_mg(cmg, sdt, ht, day_name, L)
            st.write(f"**Reduction: {pct}%**")
            st.success(f"🎯 Adjusted Vendor MG: **{vmg_sd:.2f}**")

with tab2:
    _render_generate_production_plan_tab()
