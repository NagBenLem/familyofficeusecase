
# streamlit_app.py (robust schema version, no type hints)
import os
import numpy as np
import pandas as pd
import streamlit as st

# Optional joblib (not required to run)
try:
    import joblib
except Exception:
    joblib = None

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Hybrid Recommender Demo", layout="wide")

# ----------------------------
# Helpers: schema normalization & safe access
# ----------------------------
def norm_cols(df):
    # Lowercase, strip, replace spaces with underscores; drop BOM chars.
    df = df.copy()
    df.columns = [c.replace('\ufeff', '').strip().lower().replace(" ", "_") for c in df.columns]
    return df

def rename_known(df, mapping):
    df = df.copy()
    for k, v in mapping.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

def ensure_col(df, col, default):
    if col not in df.columns:
        df[col] = default
        st.warning("Column '%s' missing; created with default '%s'" % (col, default))
    return df

def find_first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ----------------------------
# Sidebar: Data & settings
# ----------------------------
st.sidebar.header("Data & Settings")
data_dir = st.sidebar.text_input("Data folder", value=".", help="Folder containing the 5 CSVs")
k_retr = st.sidebar.slider("Retrieval K (fanout)", 5, 100, 25, step=5)
topn = st.sidebar.slider("Top-N to display", 5, 50, 10, step=5)

gate_kyc = st.sidebar.checkbox("Gate: KYC Complete", value=True)
gate_shariah = st.sidebar.checkbox("Gate: Shariah compliance", value=False)
gate_ticket = st.sidebar.checkbox("Gate: Ticket overlap", value=False)

st.sidebar.markdown("---")
use_grouped_split = st.sidebar.checkbox("Use grouped split by client for metrics", value=True)
retrain_in_app = st.sidebar.checkbox("Train/refresh ranker in app (if artifacts missing)", value=True)

@st.cache_data(show_spinner=False)
def load_csvs(folder):
    def _read(name):
        path = os.path.join(folder, "%s.csv" % name)
        df = pd.read_csv(path)
        return norm_cols(df)

    # Load & normalize column names
    cps  = _read("client_profiles")
    dmd  = _read("deal_metadata")
    cns  = _read("constraints")
    eng  = _read("engagement_logs")
    cdm  = _read("client_deal_matches")

    # Lightweight synonym renames (non-destructive)
    cps = rename_known(cps, {
        "clientid": "client_id", "client_id_": "client_id", "client": "client_id",
        "preferred_regions": "regions", "region_prefs": "regions", "preferred_sector": "preferred_sectors",
        "gp": "gp_name"
    })
    dmd = rename_known(dmd, {
        "dealid": "deal_id", "deal_id_": "deal_id", "deal": "deal_id",
        "gp": "gp_name", "fundstage": "fund_stage", "fund_stage_name": "fund_stage"
    })
    cns = rename_known(cns, {"clientid": "client_id"})
    eng = rename_known(eng, {
        "clientid": "client_id", "meeting_notes": "last_meeting_notes", "notes": "last_meeting_notes"
    })
    cdm = rename_known(cdm, {"clientid": "client_id", "dealid": "deal_id", "is_subscribed": "subscribed"})

    # Hard requirements check with helpful errors
    required = [
        ("client_profiles", cps, ["client_id"]),
        ("deal_metadata", dmd, ["deal_id"]),
        ("client_deal_matches", cdm, ["client_id", "deal_id", "subscribed"]),
    ]
    for name, df, cols in required:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError("%s.csv is missing required column(s): %s. Present columns: %s" % (name, missing, list(df.columns)[:20]))

    # Optional columns defaults
    if "kyc_status" not in cns.columns:
        cns["kyc_status"] = "Complete"  # default to allow demo; adjust to your data
        st.info("constraints.csv lacks 'kyc_status'; defaulted to 'Complete' for demo.")
    if "shariah_compliant" not in cns.columns:
        cns["shariah_compliant"] = 0
    if "shariah_filter" not in dmd.columns:
        dmd["shariah_filter"] = 0

    # Ensure engagement has client_id; otherwise create empty notes for all clients
    if "client_id" not in eng.columns:
        st.warning("engagement_logs.csv lacks 'client_id'; creating empty notes for all clients.")
        eng = pd.DataFrame({"client_id": cps["client_id"].dropna().unique(), "last_meeting_notes": ""})
    else:
        # Ensure last_meeting_notes exists (from synonyms or create)
        if "last_meeting_notes" not in eng.columns:
            cand = find_first(eng, ["meeting_notes", "notes", "rm_notes", "last_notes"])
            if cand:
                eng.rename(columns={cand: "last_meeting_notes"}, inplace=True)
            else:
                eng["last_meeting_notes"] = ""

    return cps, dmd, cns, eng, cdm

try:
    client_profiles, deal_metadata, constraints, engagement_logs, client_deal_matches = load_csvs(data_dir)
except Exception as e:
    st.error("Failed to load/validate CSVs from '%s'.\n\n%s" % (data_dir, e))
    st.stop()

# ----------------------------
# Domain utilities
# ----------------------------
def in_list(val, list_str):
    if pd.isna(list_str): return 0
    return 1 if str(val) in [s.strip() for s in str(list_str).split("|")] else 0

def _join(parts, sep=" | "):
    return sep.join([str(p).strip() for p in parts if pd.notna(p) and str(p).strip()])

def ensure_pref_flags(df):
    if "sector_pref_match" not in df.columns:
        df["sector_pref_match"] = df.apply(lambda r: in_list(r.get("sector"), r.get("preferred_sectors")), axis=1)
    if "region_pref_match" not in df.columns:
        df["region_pref_match"] = df.apply(lambda r: in_list(r.get("region"), r.get("regions")), axis=1)
    return df

def apply_gates(df, gate_kyc=True, gate_shariah=True, gate_ticket=False):
    mask = pd.Series(True, index=df.index)
    if gate_kyc and "kyc_status" in df.columns:
        mask &= df["kyc_status"].eq("Complete")
    if gate_shariah and {"shariah_filter","shariah_compliant"}.issubset(df.columns):
        sf = pd.to_numeric(df["shariah_filter"], errors="coerce")
        sc = pd.to_numeric(df["shariah_compliant"], errors="coerce")
        mask &= ~((sf == 1) & (sc != 1))
    if gate_ticket and {"min_ticket","max_ticket","deal_size"}.issubset(df.columns):
        mn = pd.to_numeric(df["min_ticket"], errors="coerce")
        mx = pd.to_numeric(df["max_ticket"], errors="coerce")
        ds = pd.to_numeric(df["deal_size"], errors="coerce")
        mask &= ~( (mn.notna() & (mn > ds)) | (mx.notna() & (mx < ds)) )
    return df[mask]

def reasons_row(r):
    rs = []
    if r.get("sector_pref_match")==1: rs.append("sector preference match")
    if r.get("region_pref_match")==1: rs.append("region preference match")
    return ", ".join(rs) if rs else "high overall score"

def get_training_schema(rank_pipe):
    pre = rank_pipe.named_steps["pre"]
    schema = {"num_cols": [], "cat_cols": [], "txt_col": None}
    for name, trans, cols in pre.transformers_:
        if name == "num":
            schema["num_cols"] = list(cols)
        elif name == "cat":
            schema["cat_cols"] = list(cols)
        elif name == "txt":
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)) and len(cols):
                schema["txt_col"] = cols[0]
            else:
                schema["txt_col"] = cols
    return schema

def prepare_for_ranker(df, rank_pipe):
    out = df.copy()
    sch = get_training_schema(rank_pipe)
    for c in sch["num_cols"]:
        if c not in out.columns: out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in sch["cat_cols"]:
        if c not in out.columns: out[c] = "__MISSING__"
    if sch["txt_col"]:
        if sch["txt_col"] not in out.columns: out[sch["txt_col"]] = ""
        out[sch["txt_col"]] = out[sch["txt_col"]].fillna("")
    return out

# ----------------------------
# Retrieval (TF-IDF)
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_retrieval(client_profiles, engagement_logs, deal_metadata, max_features=5000, ngram=(1,2)):
    # Client text
    notes = engagement_logs[["client_id","last_meeting_notes"]].copy()
    clients_tv = client_profiles.merge(notes, on="client_id", how="left")
    clients_tv["last_meeting_notes"] = clients_tv["last_meeting_notes"].fillna("")
    clients_tv["client_text"] = clients_tv.apply(lambda r: _join([
        "preferred_sectors: %s" % r.get('preferred_sectors',''),
        "preferred_regions: %s" % r.get('regions',''),
        "notes: %s" % r.get('last_meeting_notes',''),
    ]), axis=1)

    # Deal text
    deals_tv = deal_metadata.copy()
    deals_tv["deal_text"] = deals_tv.apply(lambda r: _join([
        "sector: %s" % r.get('sector',''),
        "region: %s" % r.get('region',''),
        "gp_name: %s" % r.get('gp_name',''),
        "fund_stage: %s" % r.get('fund_stage',''),
    ]), axis=1)

    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
    corpus = pd.concat([clients_tv["client_text"], deals_tv["deal_text"]], axis=0).fillna("")
    vec.fit(corpus)
    from sklearn.preprocessing import normalize as _normalize
    C = _normalize(vec.transform(clients_tv["client_text"].fillna("")))
    D = _normalize(vec.transform(deals_tv["deal_text"].fillna("")))
    S = C @ D.T

    client_ids = clients_tv["client_id"].to_numpy()
    deal_ids   = deals_tv["deal_id"].to_numpy()

    return vec, S, client_ids, deal_ids, clients_tv, deals_tv

vec_retr, S, CLIENT_IDS, DEAL_IDS, clients_tv, deals_tv = build_retrieval(
    client_profiles, engagement_logs, deal_metadata
)

def suggest_deals_for_client(client_id, K):
    idx = np.where(CLIENT_IDS == client_id)[0]
    if len(idx)==0: return pd.DataFrame(columns=["client_id","deal_id","retr_score"])
    i = int(idx[0])
    row = S.getrow(i).toarray().ravel()
    k = min(K, len(row))
    top = np.argpartition(-row, k-1)[:k]; top = top[np.argsort(-row[top])]
    return pd.DataFrame({"client_id": client_id, "deal_id": DEAL_IDS[top], "retr_score": row[top]})

def suggest_clients_for_deal(deal_id, K):
    idx = np.where(DEAL_IDS == deal_id)[0]
    if len(idx)==0: return pd.DataFrame(columns=["deal_id","client_id","retr_score"])
    j = int(idx[0])
    col = S.getcol(j).toarray().ravel()
    k = min(K, len(col))
    top = np.argpartition(-col, k-1)[:k]; top = top[np.argsort(-col[top])]
    return pd.DataFrame({"deal_id": deal_id, "client_id": CLIENT_IDS[top], "retr_score": col[top]})

# ----------------------------
# Training table & ranker
# ----------------------------
def build_training_table():
    Xy = (client_deal_matches
            .merge(client_profiles, on="client_id", how="left")
            .merge(constraints,    on="client_id", how="left")
            .merge(engagement_logs,on="client_id", how="left")
            .merge(deal_metadata,  on="deal_id",   how="left"))
    # preference flags
    Xy["sector_pref_match"] = Xy.apply(lambda r: in_list(r.get("sector"), r.get("preferred_sectors")), axis=1)
    Xy["region_pref_match"] = Xy.apply(lambda r: in_list(r.get("region"), r.get("regions")), axis=1)
    if "last_meeting_notes" in Xy.columns:
        Xy["last_meeting_notes"] = Xy["last_meeting_notes"].fillna("")
    y = Xy["subscribed"].astype(int)
    X = Xy.drop(columns=["subscribed"])
    return X, y

@st.cache_resource(show_spinner=False)
def build_ranker(X, y):
    # split
    if use_grouped_split and "client_id" in X.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=X["client_id"]))
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    CAND_NUM = [
        "client_sector_score","engagement_score","risk_score","app_views","docs_downloaded",
        "sector_pref_match","region_pref_match","region_match","past_gp_investment","ticket_fit"
    ]
    num_cols = [c for c in CAND_NUM if c in X_tr.columns]
    cat_cols = [c for c in ["sector","region","gp_name","fund_stage"] if c in X_tr.columns]
    txt_col  = "last_meeting_notes" if "last_meeting_notes" in X_tr.columns else None

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])
    txt_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="", keep_empty_features=True)),
        ("to1d", FunctionTransformer(lambda x: x.reshape(-1), validate=False)),
        ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1,2))),
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_pipe, num_cols))
    if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
    if txt_col:  transformers.append(("txt", txt_pipe, [txt_col]))

    if not transformers:
        raise ValueError("No usable features in TRAIN. Check your joined table and feature lists.")

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    ranker = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            penalty="elasticnet",
            l1_ratio=0.2,
            solver="saga",
        )),
    ])
    ranker.fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, ranker.predict_proba(X_te)[:,1])
    ap  = average_precision_score(y_te, ranker.predict_proba(X_te)[:,1])
    return ranker, auc, ap

X, y = build_training_table()
if retrain_in_app or (joblib is None):
    ranker, auc, ap = build_ranker(X, y)
else:
    try:
        ranker = joblib.load("lr_ranker.joblib")
        auc = ap = None
    except Exception:
        ranker, auc, ap = build_ranker(X, y)

st.sidebar.markdown("---")
st.sidebar.write("Ranker evaluation")
if auc is not None and ap is not None:
    st.sidebar.write("AUC: **%.3f**  |  AP: **%.3f**" % (auc, ap))
else:
    st.sidebar.write("Loaded from artifact or trained in-app.")

# ----------------------------
# Scoring APIs
# ----------------------------
def rank_for_client(client_id, topn, K):
    ret = suggest_deals_for_client(client_id, K)
    if ret.empty: return ret
    pairs = (ret
        .merge(client_profiles, on="client_id", how="left")
        .merge(constraints,    on="client_id", how="left")
        .merge(engagement_logs,on="client_id", how="left")
        .merge(deal_metadata,  on="deal_id",   how="left")
    )
    if "last_meeting_notes" in pairs.columns:
        pairs["last_meeting_notes"] = pairs["last_meeting_notes"].fillna("")
    pairs = ensure_pref_flags(pairs)
    ready = prepare_for_ranker(pairs, ranker)
    pairs["rank_score"] = ranker.predict_proba(ready)[:,1]
    pairs = apply_gates(pairs, gate_kyc=gate_kyc, gate_shariah=gate_shariah, gate_ticket=gate_ticket)
    out = pairs.sort_values("rank_score", ascending=False).head(topn).copy()
    out["reasons"] = out.apply(reasons_row, axis=1)
    cols = ["client_id","deal_id","retr_score","rank_score","reasons"]
    for c in ["kyc_status","shariah_filter","shariah_compliant"]:
        if c in out.columns: cols.append(c)
    return out[cols]

def rank_for_deal(deal_id, topn, K):
    ret = suggest_clients_for_deal(deal_id, K)
    if ret.empty: return ret
    pairs = (ret
        .merge(client_profiles, on="client_id", how="left")
        .merge(constraints,    on="client_id", how="left")
        .merge(engagement_logs,on="client_id", how="left")
        .merge(deal_metadata,  on="deal_id",   how="left")
    )
    if "last_meeting_notes" in pairs.columns:
        pairs["last_meeting_notes"] = pairs["last_meeting_notes"].fillna("")
    pairs = ensure_pref_flags(pairs)
    ready = prepare_for_ranker(pairs, ranker)
    pairs["rank_score"] = ranker.predict_proba(ready)[:,1]
    pairs = apply_gates(pairs, gate_kyc=gate_kyc, gate_shariah=gate_shariah, gate_ticket=gate_ticket)
    out = pairs.sort_values("rank_score", ascending=False).head(topn).copy()
    out["reasons"] = out.apply(reasons_row, axis=1)
    cols = ["deal_id","client_id","retr_score","rank_score","reasons"]
    for c in ["kyc_status","shariah_filter","shariah_compliant"]:
        if c in out.columns: cols.append(c)
    return out[cols]

# ----------------------------
# UI
# ----------------------------
st.title("Hybrid Recommendation Demo (Retrieval → Ranking)")
st.caption("Uses only the 5 CSVs. TF-IDF retrieval + Elastic-Net Logistic Regression ranking. Gates: KYC/Shariah.")

mode = st.radio("Mode", ["Deals → Clients (Targeting)", "Clients → Deals (Assist)"], horizontal=True)

colA, colB = st.columns(2)
with colA:
    if mode.startswith("Deals"):
        deal_id = st.selectbox("Pick a deal_id", sorted(pd.unique(deal_metadata["deal_id"].dropna())), index=0 if len(deal_metadata)>0 else None)
    else:
        client_id = st.selectbox("Pick a client_id", sorted(pd.unique(client_profiles["client_id"].dropna())), index=0 if len(client_profiles)>0 else None)

with colB:
    st.write("Gates:", "KYC ✅" if gate_kyc else "KYC ❌", "|", "Shariah ✅" if gate_shariah else "Shariah ❌", "|", "Ticket ✅" if gate_ticket else "Ticket ❌")

if mode.startswith("Deals"):
    if deal_id is not None:
        df = rank_for_deal(int(deal_id), topn=topn, K=k_retr)
        st.subheader("Top clients for this deal")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="targeting_list_deal_%s.csv" % deal_id, mime="text/csv")
else:
    if client_id is not None:
        df = rank_for_client(int(client_id), topn=topn, K=k_retr)
        st.subheader("Top deals for this client")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="recommendations_client_%s.csv" % client_id, mime="text/csv")

st.markdown("---")
with st.expander("Debug / schema"):
    st.write("client_profiles columns:", list(client_profiles.columns))
    st.write("deal_metadata columns:", list(deal_metadata.columns))
    st.write("constraints columns:", list(constraints.columns))
    st.write("engagement_logs columns:", list(engagement_logs.columns))
    st.write("client_deal_matches columns:", list(client_deal_matches.columns))
