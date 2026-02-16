"""
ML Core — dynamic model training + prediction for any client's feature schema.

The model architecture is built dynamically from the client's encoder_columns list.
All artifacts are prefixed with the client key (e.g. tekion_per_pax_tf_model.keras).
"""
import os, math, random
from collections import defaultdict
import numpy as np, pandas as pd, tensorflow as tf, joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.dirname(__file__))
SEED = 42; CTX_LEN = 10; EPOCHS = 20; BATCH = 32; VAL_SPLIT = 0.1; TEST_SIZE = 0.2
EMB_DIMS = {"menu_items":8,"sub_category":4,"category":4,"weekday":2,"day_type":2,"holiday_type":2,"meal_day":2}

# ── helpers ──
def norm(s): return s.strip().lower() if isinstance(s, str) else s

def token(v):
    """Normalize any raw value to a safe categorical token."""
    if pd.isna(v):
        return "unknown"
    t = norm(v) if isinstance(v, str) else norm(str(v))
    return t if t else "unknown"

def cats_match(a,b): return a is not None and b is not None and norm(a)==norm(b)

def floor005(v): return math.floor(v/0.005)*0.005
def fmt_cols(df, cols):
    df=df.copy()
    for c in cols:
        if c in df.columns: df[c]=df[c].map(lambda x:f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x,(int,float)) else x)
    return df

# ── artifact paths (client-prefixed) ──
def _ap(ck,fn): return os.path.join(ARTIFACT_DIR,f"{ck}_{fn}")

def artifacts_exist(ck, encoder_columns):
    fns = ["per_pax_tf_model.keras"] + [f"{c}_encoder.pkl" for c in encoder_columns] + ["item_to_subcat.pkl","item_to_cat.pkl","cat_to_subs.pkl"]
    return all(os.path.exists(_ap(ck,f)) for f in fns)

# ── cache ──
_C={}
def clear_cache(ck=None):
    if ck: [_C.pop(k) for k in [k for k in _C if k.startswith(f"{ck}:")]]
    else: _C.clear()
def _lc(k,fn):
    if k not in _C: _C[k]=fn()
    return _C[k]

def load_model(ck): return _lc(f"{ck}:mdl", lambda: tf.keras.models.load_model(_ap(ck,"per_pax_tf_model.keras")))
def load_enc(ck,n): return _lc(f"{ck}:e:{n}", lambda: joblib.load(_ap(ck,f"{n}_encoder.pkl")))
def load_map(ck,n): return _lc(f"{ck}:m:{n}", lambda: joblib.load(_ap(ck,f"{n}.pkl")))

def encode_safe(le, v, d=0):
    v = token(v)
    cl = set(getattr(le, "classes_", []))
    if v in cl:
        return int(le.transform([v])[0])
    if "other" in cl:
        return int(le.transform(["other"])[0])
    if "unknown" in cl:
        return int(le.transform(["unknown"])[0])
    return d


def resolve_ci(name,known):
    m={k.lower():k for k in known}; return m.get(name.lower(),name)

def fallback_by_sub(sub,le,i2s):
    sn=norm(sub)
    for it,sc in i2s.items():
        if norm(sc)==sn and it in le.classes_: return it
    return None

# ── dataset ──
def load_dataset(ck, path, logic):
    df=pd.read_excel(path)
    df.columns=df.columns.str.strip().str.lower().str.replace(" ","_")
    dispatch={"category":logic.canonicalize_category,"day_type":logic.canonicalize_day_type,"holiday_type":logic.canonicalize_holiday_type,"meal_day":logic.canonicalize_meal_day}
    for c in ["menu_items","sub_category","category","day_type","holiday_type","meal_day","meal_type"]:
        if c in df.columns: df[c]=df[c].map(dispatch.get(c,norm))
    return df

def build_cat_to_subs(i2c,i2s):
    r=defaultdict(set)
    for it,c in i2c.items():
        s=i2s.get(it)
        if c and s: r[norm(c)].add(norm(s))
    return {k:sorted(v) for k,v in r.items()}

def get_nv_cats(ck,path,logic):
    df=load_dataset(ck,path,logic)
    if "meal_type" not in df.columns: return []
    return sorted(df.loc[df["meal_type"]=="non veg","category"].dropna().unique())

# ── DYNAMIC training ──
def train_model(ck, path, logic, verbose=0):
    np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)
    df = load_dataset(ck, path, logic)

    # -------- schema validation (fail fast, clear message) --------
    required = {"date", "menu_items", "sub_category", "category", "ideal_pp"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(
            f"[{ck}] Dataset missing required columns: {missing_required}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # Encoder columns that are allowed to be auto-filled
    autofill_defaults = {
        "day_type": "regular",
        "holiday_type": "not applicable",
        "meal_day": "veg",
    }

    # Any other missing encoder column should fail explicitly
    missing_encoder = [
        c for c in logic.encoder_columns
        if c not in df.columns and c not in ("weekday",) and c not in autofill_defaults
    ]
    if missing_encoder:
        raise ValueError(
            f"[{ck}] Missing encoder columns required by logic: {missing_encoder}. "
            f"Either add them in dataset or adjust encoder_columns."
        )

    # Parse/validate date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_date = int(df["date"].isna().sum())
    if bad_date > 0:
        raise ValueError(f"[{ck}] Column 'date' has {bad_date} invalid/null values after parsing.")

    # Parse/validate target
    df["ideal_pp"] = pd.to_numeric(df["ideal_pp"], errors="coerce")
    bad_pp = int(df["ideal_pp"].isna().sum())
    if bad_pp > 0:
        raise ValueError(f"[{ck}] Column 'ideal_pp' has {bad_pp} invalid/null numeric values.")

    # Normalize core text columns
    for c in ["menu_items", "sub_category", "category"]:
        df[c] = df[c].map(token).astype(str)

    df["weekday"] = df["date"].dt.day_name().map(norm)
    df["menu_items_list"] = df.groupby("date")["menu_items"].transform(list)


    # fill missing optional encoder columns
    for c in logic.encoder_columns:
        if c not in df.columns and c in autofill_defaults:
            df[c] = autofill_defaults[c]


    i2s=df.set_index("menu_items")["sub_category"].to_dict()
    i2c=df.set_index("menu_items")["category"].to_dict()
    joblib.dump(i2s,_ap(ck,"item_to_subcat.pkl"))
    joblib.dump(i2c,_ap(ck,"item_to_cat.pkl"))
    joblib.dump(build_cat_to_subs(i2c,i2s),_ap(ck,"cat_to_subs.pkl"))

    le_map = {}
    for c in logic.encoder_columns:
        vals = df[c].map(token).astype(str)

        le = LabelEncoder()
        fit_vals = pd.concat([vals, pd.Series(["other"])], ignore_index=True)  # force OOV bucket
        le.fit(fit_vals)

        df[f"{c}_idx"] = le.transform(vals)
        joblib.dump(le, _ap(ck, f"{c}_encoder.pkl"))
        le_map[c] = le


    # context
    i2i={it:ix for ix,it in enumerate(le_map["menu_items"].classes_)}
    pad=len(i2i)
    df["ctx"]=df["menu_items_list"].map(lambda ml:[i2i[i] for i in ml if i in i2i][:CTX_LEN]+[pad]*(CTX_LEN-len([i2i[i] for i in ml if i in i2i][:CTX_LEN])))

    # non-menu feature columns in order
    extra_cols=[c for c in logic.encoder_columns if c!="menu_items"]
    X = [np.array(df["menu_items_idx"]),
     np.array(df["ctx"].tolist())] + [np.array(df[f"{c}_idx"]) for c in extra_cols]
    y = np.array(df["ideal_pp"], dtype=float)

    # Group split by date to avoid leakage from same-day menu context
    date_groups = df["date"].dt.normalize().astype("string").fillna("nat")
    unique_dates = date_groups.unique().tolist()

    if len(unique_dates) >= 2:
        rng = np.random.RandomState(SEED)
        unique_dates = np.array(unique_dates, dtype=object)
        rng.shuffle(unique_dates)

        n_test_dates = max(1, int(round(len(unique_dates) * TEST_SIZE)))
        if n_test_dates >= len(unique_dates):
            n_test_dates = len(unique_dates) - 1  # keep at least 1 train date

        test_dates = set(unique_dates[:n_test_dates])
        test_mask = date_groups.isin(test_dates).to_numpy()
        train_mask = ~test_mask

        # safety fallback if split degenerates
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            idx = np.arange(len(df))
            rng.shuffle(idx)
            cut = max(1, int(round(len(idx) * (1 - TEST_SIZE))))
            tr_idx, te_idx = idx[:cut], idx[cut:]
            Xtr = [arr[tr_idx] for arr in X]
            Xte = [arr[te_idx] for arr in X]
            ytr, yte = y[tr_idx], y[te_idx]
        else:
            Xtr = [arr[train_mask] for arr in X]
            Xte = [arr[test_mask] for arr in X]
            ytr, yte = y[train_mask], y[test_mask]
    else:
        # very small dataset fallback
        idx = np.arange(len(df))
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * (1 - TEST_SIZE))))
        tr_idx, te_idx = idx[:cut], idx[cut:]
        Xtr = [arr[tr_idx] for arr in X]
        Xte = [arr[te_idx] for arr in X]
        ytr, yte = y[tr_idx], y[te_idx]


    # dynamic model
    inputs=[tf.keras.Input(shape=(1,),name="item"), tf.keras.Input(shape=(CTX_LEN,),name="ctx")]
    embs=[]
    embs.append(tf.keras.layers.Flatten()(tf.keras.layers.Embedding(len(le_map["menu_items"].classes_)+1,8)(inputs[0])))
    embs.append(tf.keras.layers.GlobalAveragePooling1D()(tf.keras.layers.Embedding(len(le_map["menu_items"].classes_)+1,8)(inputs[1])))
    for c in extra_cols:
        inp=tf.keras.Input(shape=(1,),name=c); inputs.append(inp)
        dim=EMB_DIMS.get(c,2)
        embs.append(tf.keras.layers.Flatten()(tf.keras.layers.Embedding(len(le_map[c].classes_)+1,dim)(inp)))

    x=tf.keras.layers.Concatenate()(embs)
    x=tf.keras.layers.Dense(64,activation="relu")(x)
    x=tf.keras.layers.Dropout(0.3)(x)
    x=tf.keras.layers.Dense(32,activation="relu")(x)
    out=tf.keras.layers.Dense(1)(x)
    model=tf.keras.Model(inputs=inputs,outputs=out)
    model.compile(optimizer="adam",loss="mse")
    model.fit(Xtr,ytr,epochs=EPOCHS,batch_size=BATCH,validation_split=VAL_SPLIT,verbose=verbose)
    preds=model.predict(Xte,verbose=0).flatten()
    rmse=float(np.sqrt(mean_squared_error(yte,preds)))
    model.save(_ap(ck,"per_pax_tf_model.keras"))
    return model, rmse

# ── DYNAMIC prediction ──
class PredResult:
    __slots__=("per_pax","total_qty","fallback","fallback_item","error")
    def __init__(self,pp=None,tq=None,fb=False,fi=None,err=None):
        self.per_pax=pp;self.total_qty=tq;self.fallback=fb;self.fallback_item=fi;self.error=err
    @property
    def ok(self): return self.per_pax is not None

def predict(ck, item, menu, mg, subcat, cat, weekday, logic, day_type=None, holiday_type=None, meal_day=None):
    mdl=load_model(ck)
    ile=load_enc(ck,"menu_items"); i2s=load_map(ck,"item_to_subcat")
    extra_cols=[c for c in logic.encoder_columns if c!="menu_items"]

    t=(item or "").strip(); mi=[(i or "").strip() for i in menu]
    known=set(ile.classes_); pad=len(ile.classes_)
    t=resolve_ci(t,known)

    fb=False; fi=None
    if t not in known:
        fi=fallback_by_sub(norm(subcat),ile,i2s)
        if fi: fb=True; idx=int(ile.transform([fi])[0])
        else: return PredResult(err=f"No fallback for sub-category '{subcat}'.")
    else: idx=int(ile.transform([t])[0])

    ctx=[]
    for i in mi:
        if i.lower()==t.lower(): continue
        ex=resolve_ci(i,known)
        if ex in known: ctx.append(int(ile.transform([ex])[0]))
    padded=ctx[:CTX_LEN]+[pad]*(CTX_LEN-len(ctx[:CTX_LEN]))

    # build feature arrays in same order as training
    vals={"sub_category":norm(subcat),"category":logic.canonicalize_category(cat),"weekday":norm(weekday),
          "day_type":logic.canonicalize_day_type(day_type or "regular"),
          "holiday_type":logic.canonicalize_holiday_type(holiday_type or "not applicable"),
          "meal_day":logic.canonicalize_meal_day(meal_day or "veg")}

    feed=[np.array([idx]), np.array([padded])]
    for c in extra_cols:
        le=load_enc(ck,c)
        feed.append(np.array([encode_safe(le,vals.get(c,""))]))

    raw=mdl.predict(feed,verbose=0)[0][0]
    pp=floor005(float(raw)); tq=pp*mg
    return PredResult(pp=pp,tq=tq,fb=fb,fi=fi)
