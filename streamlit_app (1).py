
# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

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
# Utilities
# ----------------------------
def in_list(val, list_str):
    if pd.isna(list_str): return 0
    return 1 if str(val) in [s.strip() for s in str(list_str).split("|")] else 0

def _join(parts, sep=" | "):
    return sep.join([str(p).strip() for p in parts if pd.notna(p) and str(p).strip()])

def get_training_schema(rank_pipe):
    pre = rank_pipe.named_steps["pre"]
    schema = {"num_cols": [], "cat_cols": [], "txt_col": None}
    for name, trans, cols in pre.transformers_:
        if name == "num":
            schema["num_cols"] = list(cols)
        elif name == "cat":
            schema["cat_cols"] = list(cols)
        elif name == "txt":
            # normalize to single string
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                schema["txt_col"] = cols[0] if len(cols) else None
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
        # simple overlap: client's [min,max] must intersect deal_size
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
        path = os.path.join(folder, f"{name}.csv")
        return pd.read_csv(path)
    cps  = _read("client_profiles")
    dmd  = _read("deal_metadata")
    cns  = _read("constraints")
    eng  = _read("engagement_logs")
    cdm  = _read("client_deal_matches")
    return cps, dmd, cns, eng, cdm

try:
    client_profiles, deal_metadata, constraints, engagement_logs, client_deal_matches = load_csvs(data_dir)
except Exception as e:
    st.error(f"Failed to load CSVs from '{data_dir}'. Make sure the 5 files exist. Error: {e}")
    st.stop()

# ----------------------------
# Retrieval (TF-IDF) build
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_retrieval(client_profiles, engagement_logs, deal_metadata, max_features=5000, ngram=(1,2)):
    # Client text
    notes = (engagement_logs[["client_id","last_meeting_notes"]]
             if "last_meeting_notes" in engagement_logs.columns
             else engagement_logs[["client_id"]].assign(last_meeting_notes=""))
    clients_tv = client_profiles.merge(notes, on="client_id", how="left")
    clients_tv["client_text"] = clients_tv.apply(lambda r: _join([
        f"preferred_sectors: {r.get('preferred_sectors','')}",
        f"preferred_regions: {r.get('regions','')}",
        f"notes: {r.get('last_meeting_notes','')}",
    ]), axis=1)

    # Deal text
    deals_tv = deal_metadata.copy()
    if "deal_id" not in deals_tv.columns:
        st.warning("Column 'deal_id' not found in deal_metadata; please ensure it exists.")
    deals_tv["deal_text"] = deals_tv.apply(lambda r: _join([
        f"sector: {r.get('sector','')}",
        f"region: {r.get('region','')}",
        f"gp_name: {r.get('gp_name','')}",
        f"fund_stage: {r.get('fund_stage','')}",
    ]), axis=1)

    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
    corpus = pd.concat([clients_tv["client_text"], deals_tv["deal_text"]], axis=0).fillna("")
    vec.fit(corpus)
    C = normalize(vec.transform(clients_tv["client_text"].fillna("")))
    D = normalize(vec.transform(deals_tv["deal_text"].fillna("")))
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
# Ranking: build/jit-train
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_ranker(X, y):
    # TRAIN/TEST split
    if use_grouped_split and "client_id" in X.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=X["client_id"]))
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    # Feature lists (from TRAIN)
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

    # Metrics
    auc = roc_auc_score(y_te, ranker.predict_proba(X_te)[:,1])
    ap  = average_precision_score(y_te, ranker.predict_proba(X_te)[:,1])

    return ranker, auc, ap

# Build training table X,y from the five CSVs
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
    if "subscribed" not in Xy.columns:
        st.error("Column 'subscribed' missing from client_deal_matches; needed for training labels.")
        st.stop()
    y = Xy["subscribed"].astype(int)
    X = Xy.drop(columns=["subscribed"])
    return X, y

X, y = build_training_table()
if retrain_in_app:
    ranker, auc, ap = build_ranker(X, y)
else:
    # If you already exported a trained model; otherwise fallback to training
    try:
        ranker = joblib.load("lr_ranker.joblib")
        auc = ap = None
    except Exception:
        ranker, auc, ap = build_ranker(X, y)

st.sidebar.markdown("---")
st.sidebar.write("Ranker evaluation")
if auc is not None and ap is not None:
    st.sidebar.write(f"AUC: **{auc:.3f}**  |  AP: **{ap:.3f}**")
else:
    st.sidebar.write("Loaded from artifact.")

# ----------------------------
# Scoring helpers
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
    return out[["client_id","deal_id","retr_score","rank_score","reasons","kyc_status"] + 
               [c for c in ["shariah_filter","shariah_compliant"] if c in out.columns]]

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
    return out[["deal_id","client_id","retr_score","rank_score","reasons","kyc_status"] + 
               [c for c in ["shariah_filter","shariah_compliant"] if c in out.columns]]

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
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"targeting_list_deal_{deal_id}.csv", mime="text/csv")
else:
    if client_id is not None:
        df = rank_for_client(int(client_id), topn=topn, K=k_retr)
        st.subheader("Top deals for this client")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"recommendations_client_{client_id}.csv", mime="text/csv")

st.markdown("---")
with st.expander("Debug / context"):
    st.write("client_profiles rows:", len(client_profiles), "| deal_metadata rows:", len(deal_metadata))
    st.write("Constraints cols:", list(constraints.columns))
    st.write("Engagement logs cols:", list(engagement_logs.columns))
    st.write("Client-deal matches rows:", len(client_deal_matches))
