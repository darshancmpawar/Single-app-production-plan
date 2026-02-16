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
    nv, st2, rest = classify(results, nv_cat, logic.star_categories)
    raw = (gavg(st2) + gavg(rest)) / 2
    return raw if is_nv else max(logic.adjust_vendor_mg(raw), 0)


# ═══════════════ CLIENT SELECTOR ═══════════════
st.title("Per Pax Quantity & Production Plan Prediction")
sel = st.selectbox("Select Client", CLIENT_LIST, key="client_sel")
CK = name_to_key(sel)
INFO = get_info(CK)
L = get_logic(CK)


# Transitional guard: client_logic is source of truth for mode.
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


# ═══════════════ TOASTTAB (multiplier-only, separate UI) ═══════════════
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

if L.has_nonveg_toggle:
    is_nv = st.toggle("🍗 Non-Veg Day?", value=True, key=k("is_nv"))
    meal_day = "nonveg" if is_nv else "veg"

entries = []  # each entry: {"item","subcat","category","mg","display_category","needs_shared_mg"}
menu = []

# Nonveg item(s) — shown if toggle on OR if client always has nonveg
show_nv = (L.has_nonveg_toggle and is_nv) or (
    not L.has_nonveg_toggle and L.nonveg_item_count > 0 and L.has_vendor_plan
)

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
else:
    nv_item = None
    nv_cat = None


cmg = st.number_input("Shared Client MG:", min_value=1, step=1, value=L.default_mg, key=k("shared_cmg"))

# patch nonveg entries that need shared MG
for e in entries:
    if e["needs_shared_mg"] or e["mg"] is None:
        e["mg"] = cmg


st.subheader("Enter Menu Items by Category")
star_ui = {"flavour rice", "flavoured rice", "veg curry", "veg gravy"}
north_pp = None  # for Rippling

for cat in L.fixed_categories:
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
