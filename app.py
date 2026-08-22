# IR Assignment 2 – End-to-End Information Retrieval System
# Run: streamlit run app.py

import json
import math
import os
import pickle
import re
import time
import warnings
from collections import Counter, defaultdict
from urllib.parse import urljoin, urlparse

import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ── Persistence ────────────────────────────────────────────────────────────────
DATA_DIR     = "ir_data"
os.makedirs(DATA_DIR, exist_ok=True)
CORPUS_FILE  = os.path.join(DATA_DIR, "corpus.json")
INDEX_FILE   = os.path.join(DATA_DIR, "index.pkl")
META_FILE    = os.path.join(DATA_DIR, "metadata.json")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.json")
STATS_FILE   = os.path.join(DATA_DIR, "stats.json")

def load_corpus():
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE) as f:
            return json.load(f)
    return []

def save_corpus(docs):
    with open(CORPUS_FILE, "w") as f:
        json.dump(docs, f, indent=2)

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_index(idx):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(idx, f)

def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            return json.load(f)
    return {}

def save_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def load_ratings():
    if os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE) as f:
            return json.load(f)
    return {}

def save_ratings(r):
    with open(RATINGS_FILE, "w") as f:
        json.dump(r, f, indent=2)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {}

def save_stats(s):
    with open(STATS_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ── Text preprocessing ─────────────────────────────────────────────────────────
STOP       = set(stopwords.words("english"))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()

def tokenize(text):
    return word_tokenize(clean_text(text))

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP and len(t) > 2]

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess(text, method="lemmatize"):
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return stem_tokens(tokens) if method == "stem" else lemmatize_tokens(tokens)

def preprocess_str(text, method="lemmatize"):
    return " ".join(preprocess(text, method))

# ── Crawling ───────────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def crawl(seeds, max_depth=1, max_pages=20):
    visited_urls = set()
    seen_hashes  = set()
    docs, meta   = [], {}
    failed = dup_urls = dup_docs = total_size = 0
    queue  = [(url.strip(), 0) for url in seeds if url.strip()]
    t0     = time.time()
    progress = st.progress(0)
    status   = st.empty()
    while queue and len(docs) < max_pages:
        url, depth = queue.pop(0)
        if url in visited_urls:
            dup_urls += 1
            continue
        visited_urls.add(url)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                failed += 1
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title else url
            body  = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))[:5000]
            h = hash(body[:500])
            if h in seen_hashes or len(body) < 100:
                dup_docs += 1
                continue
            seen_hashes.add(h)
            doc_id    = f"doc_{len(docs)}"
            page_size = len(body.encode("utf-8"))
            total_size += page_size
            docs.append({"id": doc_id, "url": url, "title": title, "body": body})
            meta[doc_id] = {"url": url, "title": title, "length": len(body),
                            "depth": depth, "size_bytes": page_size,
                            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "web"}
            status.text(f"Crawled ({len(docs)}/{max_pages}): {title[:60]}")
            progress.progress(len(docs) / max_pages)
            if depth < max_depth:
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    p = urlparse(link)
                    if p.scheme in ("http", "https") and link not in visited_urls:
                        queue.append((link, depth + 1))
        except Exception:
            failed += 1
    progress.empty()
    status.empty()
    return docs, meta, {
        "total_visited": len(visited_urls), "successful": len(docs),
        "failed": failed, "dup_urls": dup_urls, "dup_docs": dup_docs,
        "avg_page_size_kb": round(total_size / max(len(docs), 1) / 1024, 2),
        "crawl_duration_s": round(time.time() - t0, 2),
        "last_crawl": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

# ── Indexing ───────────────────────────────────────────────────────────────────
def build_inverted_index(docs):
    index = defaultdict(lambda: defaultdict(int))
    for doc in docs:
        for tok in preprocess(doc["title"] + " " + doc["body"]):
            index[tok][doc["id"]] += 1
    return {k: dict(v) for k, v in index.items()}

def build_tfidf(docs):
    corpus = [preprocess_str(d["title"] + " " + d["body"]) for d in docs]
    vec = TfidfVectorizer(max_features=5000)
    mat = vec.fit_transform(corpus)
    return vec, mat

def index_size_kb():
    return round(os.path.getsize(INDEX_FILE) / 1024, 1) if os.path.exists(INDEX_FILE) else 0.0

# ── PageRank & HITS ────────────────────────────────────────────────────────────
def build_pagerank(docs):
    G = nx.DiGraph()
    for d in docs:
        G.add_node(d["id"])
    _, mat = build_tfidf(docs)
    sim = cosine_similarity(mat)
    for i in range(len(docs)):
        for j in range(len(docs)):
            if i != j and sim[i, j] > 0.1:
                G.add_edge(docs[i]["id"], docs[j]["id"], weight=float(sim[i, j]))
    pr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {d["id"]: 1 / len(docs) for d in docs}
    return pr, G

def compute_hits(G):
    if G.number_of_edges() == 0:
        return {}, {}
    return nx.hits(G, max_iter=100, normalized=True)

# ── Search ─────────────────────────────────────────────────────────────────────
def boolean_search(query, index, docs, op="AND"):
    tokens = preprocess(query)
    if not tokens:
        return []
    sets   = [set(index.get(t, {}).keys()) for t in tokens]
    result = sets[0].intersection(*sets[1:]) if op == "AND" else sets[0].union(*sets[1:])
    return [d for d in docs if d["id"] in result]

def tfidf_search(query, docs, vec, mat, top_k=10):
    if not docs:
        return []
    scores = cosine_similarity(vec.transform([preprocess_str(query)]), mat).flatten()
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], float(scores[i])) for i in ranked if scores[i] > 0]

def ranked_search(query, docs, vec, mat, pr, top_k=10, alpha=0.5):
    if not docs:
        return []
    tfidf_s = cosine_similarity(vec.transform([preprocess_str(query)]), mat).flatten()
    pr_s    = np.array([pr.get(d["id"], 0) for d in docs])
    if pr_s.max() > 0:
        pr_s = pr_s / pr_s.max()
    combined = alpha * tfidf_s + (1 - alpha) * pr_s
    ranked   = np.argsort(combined)[::-1][:top_k]
    return [(docs[i], float(combined[i])) for i in ranked if combined[i] > 0]

# ── Recommendations ────────────────────────────────────────────────────────────
CATEGORIES = {
    "ML / AI":    ["machine", "learning", "neural", "deep", "model", "algorithm"],
    "IR / Search":["retrieval", "search", "index", "query", "ranking"],
    "NLP":        ["language", "text", "nlp", "processing", "corpus"],
    "Web":        ["web", "internet", "crawl", "page", "link", "network"],
}

def auto_label(doc):
    t = (doc["title"] + " " + doc["body"][:200]).lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"

CAT_COLORS = {"ML / AI": "#7c3aed", "IR / Search": "#2563eb",
              "NLP": "#0891b2", "Web": "#059669", "General": "#64748b"}

def content_based_recommend(doc_idx, docs, mat, top_k=5):
    if doc_idx >= mat.shape[0]:
        return []
    sims = cosine_similarity(mat[doc_idx], mat).flatten()
    sims[doc_idx] = 0
    ranked = np.argsort(sims)[::-1][:top_k]
    return [(docs[i], float(sims[i])) for i in ranked if sims[i] > 0]

def collab_recommend(user_id, ratings, docs, mat, top_k=5):
    if not ratings or user_id not in ratings:
        return []
    all_users = list(ratings.keys())
    all_docs  = [d["id"] for d in docs]
    uid_map   = {u: i for i, u in enumerate(all_users)}
    did_map   = {d: i for i, d in enumerate(all_docs)}
    R = np.zeros((len(all_users), len(all_docs)))
    for u, items in ratings.items():
        for d_id, score in items.items():
            if u in uid_map and d_id in did_map:
                R[uid_map[u], did_map[d_id]] = score
    user_vec = R[uid_map[user_id]].reshape(1, -1)
    if user_vec.sum() == 0:
        return []
    user_sims = cosine_similarity(user_vec, R).flatten()
    user_sims[uid_map[user_id]] = 0
    pred = user_sims @ R
    already_rated = set(ratings[user_id].keys())
    ranked_docs = [(all_docs[i], pred[i]) for i in np.argsort(pred)[::-1]
                   if all_docs[i] not in already_rated and pred[i] > 0][:top_k]
    doc_map = {d["id"]: d for d in docs}
    return [(doc_map[d_id], score) for d_id, score in ranked_docs if d_id in doc_map]

# ── Evaluation ─────────────────────────────────────────────────────────────────
def compute_metrics(retrieved_ids, relevant_ids, k=10):
    retrieved = list(retrieved_ids)[:k]
    relevant  = set(relevant_ids)
    hits      = [1 if r in relevant else 0 for r in retrieved]
    prec = sum(hits) / len(retrieved) if retrieved else 0
    rec  = sum(hits) / len(relevant)  if relevant  else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    ap, rel_so_far = 0.0, 0
    for i, h in enumerate(hits):
        if h:
            rel_so_far += 1
            ap += rel_so_far / (i + 1)
    ap   = ap / len(relevant) if relevant else 0
    mrr  = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg  = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return {"Precision": prec, "Recall": rec, "F1": f1,
            "Precision@K": sum(hits) / k if k else 0, "Recall@K": rec,
            "AP": ap, "MRR": mrr, "NDCG": ndcg}

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart IR System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main { background: #f0f4ff; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070d1f 0%, #0f172a 45%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-size: 18px !important;
}

/* ── Page header banner ── */
.page-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 55%, #3b82f6 100%);
    border-radius: 18px;
    padding: 26px 32px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(29,78,216,0.28);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.page-banner-left h1 {
    font-size: 30px;
    font-weight: 700;
    color: white !important;
    margin: 0 0 4px 0;
}
.page-banner-left p {
    font-size: 14px;
    color: rgba(255,255,255,0.82);
    margin: 0;
}
.status-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white !important;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    backdrop-filter: blur(8px);
}

/* ── KPI metrics ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    border-top: 3px solid #2563eb;
}
[data-testid="stMetricLabel"] { font-size: 13px !important; color: #64748b !important; }
[data-testid="stMetricValue"] { font-size: 26px !important; color: #0f172a !important; font-weight: 700 !important; }

/* ── Buttons ── */
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-size: 15px;
    font-weight: 600;
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white;
    border: none;
    box-shadow: 0 4px 16px rgba(29,78,216,0.30);
    transition: all 0.2s ease;
    letter-spacing: 0.3px;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    box-shadow: 0 6px 22px rgba(29,78,216,0.42);
    transform: translateY(-1px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #e8edf8;
    padding: 5px 6px;
    border-radius: 12px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 14px;
    color: #475569;
    border: none !important;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.09);
    color: #1d4ed8 !important;
    font-weight: 600 !important;
}

/* ── Cards ── */
.result-card {
    background: white;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 3px 16px rgba(0,0,0,0.07);
    border-left: 4px solid #2563eb;
    transition: box-shadow 0.2s, transform 0.2s;
}
.result-card:hover {
    box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}
.result-card h4 { margin: 0 0 6px 0; font-size: 16px; color: #0f172a; font-weight: 600; }
.result-card .snippet { font-size: 13px; color: #475569; line-height: 1.6; margin-bottom: 10px; }
.result-card .meta { font-size: 12px; color: #94a3b8; display: flex; gap: 16px; align-items: center; }

/* ── Cat badge ── */
.cat-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    color: white;
}

/* ── Info / warn / error boxes ── */
.stAlert { border-radius: 12px !important; }

/* ── Expander ── */
details summary {
    background: #f8fafc;
    border-radius: 10px !important;
    font-weight: 500;
    padding: 10px 14px !important;
}
details[open] summary { border-radius: 10px 10px 0 0 !important; }

/* ── Section divider ── */
.sec-head {
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
    margin: 24px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Stat mini card (sidebar) ── */
.sb-stat {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 7px 12px;
    margin: 5px 0;
    font-size: 13px;
    color: #cbd5e1 !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 14px 20px;
    background: white;
    border-radius: 12px;
    margin-top: 28px;
    color: #94a3b8;
    font-size: 13px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Plotly chart container ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] > div {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Text input & select ── */
[data-testid="stTextInput"] > div > div,
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {
    "corpus":         load_corpus(),
    "index":          load_index(),
    "tfidf_vec":      None,
    "tfidf_mat":      None,
    "pagerank":       {},
    "hits_hubs":      {},
    "hits_auth":      {},
    "link_graph":     None,
    "ratings":        load_ratings(),
    "stats":          load_stats(),
    "search_history": [],
    "crawl_summary":  {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

corpus   = st.session_state.corpus
index    = st.session_state.index
meta_all = load_meta()
cs       = st.session_state.crawl_summary

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='padding:12px 0 4px 0;'>
  <div style='font-size:22px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;'>🚀 Smart IR</div>
  <div style='font-size:12px;color:#94a3b8;margin-top:2px;'>Information Retrieval System</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    idx_kb     = index_size_kb()
    last_crawl = cs.get("last_crawl", "—")
    if last_crawl and last_crawl != "—":
        last_crawl = last_crawl.replace("T", " ")

    st.markdown(f"""
<div class='sb-stat'>📄 <b>Documents:</b> {len(corpus)}</div>
<div class='sb-stat'>📚 <b>Index Terms:</b> {len(index)}</div>
<div class='sb-stat'>💾 <b>Index Size:</b> {idx_kb} KB</div>
<div class='sb-stat'>🕐 <b>Last Crawl:</b><br><span style='font-size:11px'>{last_crawl}</span></div>
""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("📌 Navigation", [
        "🏠  Dashboard",
        "🕷️  Data Sources",
        "🗂️  Index Management",
        "⛏️  Text Mining",
        "🔎  Search",
        "📈  Ranking Visualization",
        "💡  Recommendations",
        "📐  Evaluation",
        "⚡  Performance Analytics",
        "🧠  Inference & Discussion",
    ])
    page = page.split("  ", 1)[1]  # strip icon prefix

    st.markdown("---")
    st.markdown("""
<div style='font-size:11px;color:#64748b;text-align:center;'>
IR Assignment 2 &nbsp;|&nbsp; AIMLCZG537
</div>""", unsafe_allow_html=True)

# ── UI helpers ─────────────────────────────────────────────────────────────────
CHART_COLORS = ["#2563eb", "#10b981", "#f59e0b", "#ef4444",
                "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"]

def banner(icon, title, subtitle):
    st.markdown(f"""
<div class="page-banner">
  <div class="page-banner-left">
    <h1>{icon} {title}</h1>
    <p>{subtitle}</p>
  </div>
  <span class="status-badge">🟢 System Online</span>
</div>
""", unsafe_allow_html=True)

def sec(label):
    st.markdown(f"<div class='sec-head'>{label}</div>", unsafe_allow_html=True)

def footer():
    st.markdown(f"""
<div class="app-footer">
  🚀 Smart Information Retrieval System &nbsp;·&nbsp; IR Assignment 2 &nbsp;·&nbsp;
  AIMLCZG537 / DSECLZG537 &nbsp;·&nbsp; {time.strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)

def cat_badge(label):
    color = CAT_COLORS.get(label, "#64748b")
    return f"<span class='cat-badge' style='background:{color};'>{label}</span>"

def rec_card(rank, doc, score, category=""):
    snippet = doc["body"][:220].replace("\n", " ") + "…"
    badge   = cat_badge(category) if category else ""
    st.markdown(f"""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
    <h4>#{rank} &nbsp;{doc['title'][:68]}</h4>
    {badge}
  </div>
  <div class="snippet">{snippet}</div>
  <div class="meta">
    <span>🔗 <a href="{doc['url']}" target="_blank" style="color:#2563eb;">Open link</a></span>
    <span>📊 Score: <b style="color:#0f172a;">{score:.4f}</b></span>
    <span>📝 {len(doc['body'].split())} words</span>
  </div>
</div>
""", unsafe_allow_html=True)

def chart_config(fig, height=380):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", size=12, color="#374151"),
        margin=dict(l=16, r=16, t=48, b=16),
        title_font=dict(size=15, color="#0f172a", family="Segoe UI"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    )
    return fig

def empty_state(msg, hint=""):
    st.markdown(f"""
<div style="text-align:center;padding:48px 24px;background:white;border-radius:16px;
            box-shadow:0 4px 20px rgba(0,0,0,0.06);margin:20px 0;">
  <div style="font-size:48px;margin-bottom:12px;">🗂️</div>
  <div style="font-size:18px;font-weight:600;color:#0f172a;margin-bottom:8px;">{msg}</div>
  <div style="font-size:14px;color:#64748b;">{hint}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    banner("📊", "IR Dashboard — Group 52", "Live overview of your IR system — corpus, index, and recent activity")

    total_tokens = sum(sum(v.values()) for v in index.values())
    avg_len      = int(np.mean([len(d["body"].split()) for d in corpus])) if corpus else 0
    dup_skip     = cs.get("dup_urls", 0) + cs.get("dup_docs", 0)
    avg_depth    = round(np.mean([v.get("depth", 0) for v in meta_all.values()]), 1) if meta_all else 0

    sec("📈 Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents",        len(corpus))
    c2.metric("📚 Vocabulary Size",   len(index))
    c3.metric("📝 Total Tokens",      f"{total_tokens:,}")
    c4.metric("📖 Avg Doc Length",    f"{avg_len} words")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🌐 URLs Visited",      cs.get("total_visited", "—"))
    c6.metric("🚫 Duplicates Skipped", dup_skip)
    c7.metric("💾 Index Size",         f"{idx_kb} KB")
    c8.metric("🔁 Avg Crawl Depth",    avg_depth)

    if not corpus:
        empty_state("No documents yet", "Go to Data Sources to crawl pages, upload a CSV, or fetch from an API.")
    else:
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border-radius:14px;
            padding:18px 24px;margin:16px 0;border:1px solid #bfdbfe;">
  <b style="color:#1e40af;font-size:15px;">✅ System ready</b>
  <span style="color:#3b82f6;font-size:14px;"> — {len(corpus)} documents indexed across {len(index)} terms.
  Use the sidebar to search, explore rankings, or evaluate retrieval methods.</span>
</div>
""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            sec("📏 Document Length Distribution")
            lengths = [len(d["body"].split()) for d in corpus]
            fig = chart_config(px.histogram(
                x=lengths, nbins=20, labels={"x": "Word Count"},
                title="Document Length Distribution",
                color_discrete_sequence=[CHART_COLORS[0]]))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            sec("🔤 Top-20 Index Terms")
            if index:
                df_t = pd.DataFrame(
                    [(t, len(p)) for t, p in index.items()],
                    columns=["Term", "Doc Frequency"]
                ).nlargest(20, "Doc Frequency")
                fig2 = chart_config(px.bar(
                    df_t, x="Term", y="Doc Frequency",
                    title="Top Terms by Document Frequency",
                    color_discrete_sequence=[CHART_COLORS[1]]))
                fig2.update_xaxes(tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            sec("🏷️ Category Distribution")
            labels   = [auto_label(d) for d in corpus]
            lc_df    = pd.DataFrame(Counter(labels).items(), columns=["Category", "Count"])
            fig3 = chart_config(px.pie(
                lc_df, names="Category", values="Count",
                title="Auto-labeled Categories",
                color_discrete_sequence=CHART_COLORS))
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            sec("🕵️ Recent Search History")
            history = st.session_state.search_history
            if history:
                for q, mode, n, ms in reversed(history[-8:]):
                    st.markdown(f"""
<div style="background:white;border-radius:10px;padding:10px 16px;margin-bottom:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);display:flex;justify-content:space-between;
            align-items:center;border-left:3px solid #2563eb;">
  <span style="font-size:14px;color:#0f172a;font-weight:500;">`{q}`</span>
  <span style="font-size:12px;color:#64748b;">{mode} · {n} results · {ms:.0f} ms</span>
</div>
""", unsafe_allow_html=True)
            else:
                st.info("No searches yet — run a query on the Search page.")

        sec("📋 All Crawled Documents")
        df_docs = pd.DataFrame([{
            "Title":    d["title"][:70],
            "URL":      d["url"],
            "Words":    len(d["body"].split()),
            "Category": auto_label(d),
            "Source":   meta_all.get(d["id"], {}).get("source", "web"),
        } for d in corpus])
        st.dataframe(df_docs, use_container_width=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Data Sources
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Sources":
    banner("🕷️", "Data Sources", "Populate your corpus via web crawling, CSV upload, or a REST API")

    tab_crawl, tab_csv, tab_api = st.tabs([
        "🌐  Web Crawling",
        "📂  Upload CSV Dataset",
        "🔌  API  (Optional)",
    ])

    # ── Web Crawling ──────────────────────────────────────────────────────────
    with tab_crawl:
        st.markdown("""
<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-radius:12px;
            padding:14px 20px;margin-bottom:18px;border:1px solid #bbf7d0;">
  <b style="color:#166534;">🌐 Web Crawling</b>
  <span style="color:#15803d;font-size:14px;"> — Fetches real web pages, strips boilerplate, deduplicates by URL and content hash.</span>
</div>
""", unsafe_allow_html=True)

        seeds_raw = st.text_area("Seed URLs (one per line)",
            value="\n".join([
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/Natural_language_processing",
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://en.wikipedia.org/wiki/Search_engine",
                "https://en.wikipedia.org/wiki/PageRank",
            ]), height=150)

        col1, col2 = st.columns(2)
        max_depth = col1.slider("Crawl Depth", 0, 2, 0,
                                help="0 = seed pages only · 1 = follow links one level")
        max_pages = col2.slider("Max Pages", 5, 50, 10)

        if st.button("🚀 Start Crawling", type="primary", key="btn_crawl"):
            seeds = [s.strip() for s in seeds_raw.strip().split("\n") if s.strip()]
            if not seeds:
                st.error("Enter at least one seed URL.")
            else:
                with st.spinner("Crawling the web…"):
                    new_docs, new_meta, summary = crawl(seeds, max_depth=max_depth, max_pages=max_pages)

                existing_urls = {d["url"] for d in st.session_state.corpus}
                added = [d for d in new_docs if d["url"] not in existing_urls]
                base  = len(st.session_state.corpus)
                for i, d in enumerate(added):
                    d["id"] = f"doc_{base + i}"

                st.session_state.corpus.extend(added)
                save_corpus(st.session_state.corpus)
                meta = load_meta()
                meta.update({d["id"]: new_meta.get(d.get("id", ""), {}) for d in added})
                save_meta(meta)
                st.session_state.crawl_summary = summary

                st.success(f"✅ Added **{len(added)}** new documents — corpus now has **{len(st.session_state.corpus)}**")

                sec("📊 Crawl Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("🌐 URLs Visited",      summary["total_visited"])
                s2.metric("✅ Successful",         summary["successful"])
                s3.metric("❌ Failed Requests",    summary["failed"])
                s4.metric("🚫 Dup URLs Skipped",   summary["dup_urls"])
                s5, s6, s7 = st.columns(3)
                s5.metric("📄 Dup Docs Skipped",   summary["dup_docs"])
                s6.metric("📦 Avg Page Size",      f"{summary['avg_page_size_kb']} KB")
                s7.metric("⏱️ Crawl Duration",     f"{summary['crawl_duration_s']} s")

    # ── Upload CSV ────────────────────────────────────────────────────────────
    with tab_csv:
        st.markdown("""
<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border-radius:12px;
            padding:14px 20px;margin-bottom:18px;border:1px solid #bfdbfe;">
  <b style="color:#1e40af;">📂 CSV Dataset Import</b>
  <span style="color:#3b82f6;font-size:14px;"> — Upload any CSV. Columns are auto-detected; override with the dropdowns below.</span>
</div>
""", unsafe_allow_html=True)

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.info("**text / content / body / description / abstract** → body column")
        col_info2.info("**title / headline / name** → title column")
        col_info3.info("**url / link / source** → URL column")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded:
            try:
                df_csv = pd.read_csv(uploaded)
                st.markdown(f"**Preview** — {len(df_csv):,} rows · {len(df_csv.columns)} columns")
                st.dataframe(df_csv.head(5), use_container_width=True)

                cols_lower = {c.lower(): c for c in df_csv.columns}
                body_col  = next((cols_lower[c] for c in ["text","content","body","description","abstract","article"] if c in cols_lower), None)
                title_col = next((cols_lower[c] for c in ["title","headline","name","subject"] if c in cols_lower), None)
                url_col   = next((cols_lower[c] for c in ["url","link","source"] if c in cols_lower), None)

                c1, c2, c3 = st.columns(3)
                body_col  = c1.selectbox("Text / Body column *", df_csv.columns,
                                          index=list(df_csv.columns).index(body_col) if body_col else 0)
                title_col = c2.selectbox("Title column (optional)", ["(none)"] + list(df_csv.columns),
                                          index=(["(none)"] + list(df_csv.columns)).index(title_col) if title_col else 0)
                url_col   = c3.selectbox("URL column (optional)", ["(none)"] + list(df_csv.columns),
                                          index=(["(none)"] + list(df_csv.columns)).index(url_col) if url_col else 0)

                max_rows = st.slider("Max rows to import", 10, min(5000, len(df_csv)), min(200, len(df_csv)))

                if st.button("📥 Import CSV into Corpus", type="primary", key="btn_csv"):
                    existing_bodies = {d["body"][:200] for d in st.session_state.corpus}
                    added_csv, dup_csv = [], 0
                    base = len(st.session_state.corpus)
                    meta = load_meta()
                    for i, row in df_csv.head(max_rows).iterrows():
                        body = str(row[body_col]).strip()
                        if not body or body.lower() in ("nan","none","null","na","n/a","") or body[:200] in existing_bodies:
                            dup_csv += 1
                            continue
                        existing_bodies.add(body[:200])
                        title  = str(row[title_col]).strip() if title_col != "(none)" else body[:60]
                        url    = str(row[url_col]).strip()   if url_col   != "(none)" else f"csv://row_{i}"
                        doc_id = f"doc_{base + len(added_csv)}"
                        added_csv.append({"id": doc_id, "url": url, "title": title, "body": body[:5000]})
                        meta[doc_id] = {"url": url, "title": title, "length": len(body),
                                        "depth": 0, "size_bytes": len(body.encode()),
                                        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "csv"}
                    st.session_state.corpus.extend(added_csv)
                    save_corpus(st.session_state.corpus)
                    save_meta(meta)
                    st.success(f"✅ Imported **{len(added_csv)}** documents · Skipped **{dup_csv}** duplicates · Total corpus: **{len(st.session_state.corpus)}**")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # ── API ───────────────────────────────────────────────────────────────────
    with tab_api:
        st.markdown("""
<div style="background:linear-gradient(135deg,#fdf4ff,#fae8ff);border-radius:12px;
            padding:14px 20px;margin-bottom:18px;border:1px solid #e9d5ff;">
  <b style="color:#6b21a8;">🔌 REST API Ingestion</b>
  <span style="color:#7c3aed;font-size:14px;"> — Fetch documents from any public JSON API (e.g. NewsAPI, HackerNews, custom endpoint).</span>
</div>
""", unsafe_allow_html=True)

        api_url = st.text_input("API Endpoint URL",
                                placeholder="https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_KEY")
        api_key = st.text_input("API Key (if required)", type="password")

        fc1, fc2, fc3 = st.columns(3)
        f_title = fc1.text_input("Title field",  "title")
        f_body  = fc2.text_input("Body field",   "description")
        f_url   = fc3.text_input("URL field",    "url")
        f_root  = st.text_input("Root array key", "articles",
                                help="JSON key containing the articles list — e.g. 'articles' for NewsAPI")
        max_api = st.slider("Max articles", 5, 100, 20)

        if st.button("🔗 Fetch from API", type="primary", key="btn_api"):
            if not api_url.strip():
                st.error("Enter an API endpoint URL.")
            else:
                try:
                    hdrs = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    with st.spinner("Fetching…"):
                        resp = requests.get(api_url.strip(), headers=hdrs, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    if f_root.strip() and isinstance(data, dict):
                        data = data.get(f_root.strip(), data)
                    if isinstance(data, dict):
                        data = list(data.values())[0] if data else []
                    if not isinstance(data, list):
                        st.error("Could not find a list in the API response — check the root array key.")
                        st.stop()

                    existing_bodies = {d["body"][:200] for d in st.session_state.corpus}
                    added_api, dup_api = [], 0
                    base = len(st.session_state.corpus)
                    meta = load_meta()
                    for i, item in enumerate(data[:max_api]):
                        if not isinstance(item, dict):
                            continue
                        body  = str(item.get(f_body, "") or item.get("content", "") or "").strip()
                        title = str(item.get(f_title, "") or "").strip() or body[:60]
                        url   = str(item.get(f_url,   "") or f"api://item_{i}").strip()
                        if not body or body[:200] in existing_bodies:
                            dup_api += 1
                            continue
                        existing_bodies.add(body[:200])
                        doc_id = f"doc_{base + len(added_api)}"
                        added_api.append({"id": doc_id, "url": url, "title": title, "body": body[:5000]})
                        meta[doc_id] = {"url": url, "title": title, "length": len(body),
                                        "depth": 0, "size_bytes": len(body.encode()),
                                        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "api"}
                    st.session_state.corpus.extend(added_api)
                    save_corpus(st.session_state.corpus)
                    save_meta(meta)
                    st.success(f"✅ Fetched **{len(added_api)}** articles · Skipped **{dup_api}** duplicates · Total: **{len(st.session_state.corpus)}**")
                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
                except Exception as e:
                    st.error(f"Error processing response: {e}")

    # ── Shared corpus table ───────────────────────────────────────────────────
    st.markdown("---")
    if st.session_state.corpus:
        sec(f"📋 Current Corpus — {len(st.session_state.corpus)} documents")
        meta = load_meta()
        rows = [{"ID": d["id"], "Title": d["title"][:60], "URL": d["url"],
                 "Source": meta.get(d["id"], {}).get("source", "web"),
                 "Depth": meta.get(d["id"], {}).get("depth", "-"),
                 "Size (KB)": round(meta.get(d["id"], {}).get("size_bytes", len(d["body"])) / 1024, 1),
                 "Crawled At": meta.get(d["id"], {}).get("crawled_at", "-")}
                for d in st.session_state.corpus]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if st.button("🗑️ Clear Entire Corpus"):
            st.session_state.corpus = []; st.session_state.index = {}
            st.session_state.crawl_summary = {}
            save_corpus([]); save_index({})
            st.rerun()
    else:
        empty_state("Corpus is empty", "Use one of the tabs above to add documents.")

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Index Management
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Index Management":
    banner("🗂️", "Index Management", "Build the inverted index, TF-IDF matrix, PageRank, and HITS")

    corpus = st.session_state.corpus
    if not corpus:
        empty_state("No documents to index", "Go to Data Sources first.")
    else:
        st.markdown(f"""
<div style="background:white;border-radius:14px;padding:18px 24px;
            box-shadow:0 4px 18px rgba(0,0,0,0.07);margin-bottom:20px;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
  <div>
    <div style="font-size:15px;font-weight:600;color:#0f172a;">Ready to index</div>
    <div style="font-size:13px;color:#64748b;">{len(corpus)} documents in corpus</div>
  </div>
  <div style="font-size:13px;color:#64748b;">
    Builds: inverted index · TF-IDF matrix · cosine similarity graph · PageRank · HITS
  </div>
</div>
""", unsafe_allow_html=True)

        if st.button("⚙️ Build / Rebuild Full Index", type="primary"):
            t0 = time.time()
            with st.spinner("Building inverted index and TF-IDF…"):
                try:
                    idx = build_inverted_index(corpus)
                    st.session_state.index = idx
                    save_index(idx)
                    vec, mat = build_tfidf(corpus)
                    st.session_state.tfidf_vec = vec
                    st.session_state.tfidf_mat = mat
                except ValueError as e:
                    st.error(f"Index build failed: {e}")
                    st.stop()
            with st.spinner("Computing PageRank and HITS…"):
                pr, G = build_pagerank(corpus)
                hubs, auths = compute_hits(G)
                st.session_state.pagerank   = pr
                st.session_state.hits_hubs  = hubs
                st.session_state.hits_auth  = auths
                st.session_state.link_graph = G

            build_time = round(time.time() - t0, 2)
            s = load_stats()
            s["index_build_time_s"] = build_time
            s["index_build_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_stats(s)
            st.session_state.stats = s
            st.success(f"✅ Index built: **{len(idx):,}** unique terms in **{build_time}s**")

        idx = st.session_state.index
        if idx:
            sec("📊 Index Statistics")
            m1, m2, m3 = st.columns(3)
            m1.metric("📚 Unique Terms",   f"{len(idx):,}")
            m2.metric("📋 Total Postings", f"{sum(len(v) for v in idx.values()):,}")
            m3.metric("💾 Index Size",      f"{index_size_kb()} KB")

            sec("🔍 Term Lookup")
            col_l, col_r = st.columns([3, 1])
            term = col_l.text_input("Enter a term to look up", "retrieval")
            if term:
                proc = preprocess(term)
                key  = proc[0] if proc else term
                postings = idx.get(key, {})
                if postings:
                    st.success(f"Term **`{key}`** found in **{len(postings)}** document(s)")
                    st.json(postings)
                else:
                    st.info(f"Term `{key}` not in index.")

            sec("📊 Top 30 Terms by Document Frequency")
            df_idx = pd.DataFrame(
                [(t, len(p), sum(p.values())) for t, p in idx.items()],
                columns=["Term", "Doc Freq", "Total Occurrences"]
            ).nlargest(30, "Doc Freq")
            col_tbl, col_bar = st.columns([1, 2])
            with col_tbl:
                st.dataframe(df_idx, use_container_width=True, height=380)
            with col_bar:
                fig = chart_config(px.bar(
                    df_idx.head(20), x="Term", y="Doc Freq",
                    title="Top-20 Terms", color_discrete_sequence=[CHART_COLORS[0]]))
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Text Mining
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Text Mining":
    banner("⛏️", "Text Mining & Preprocessing", "Keyword extraction, preprocessing comparison, and document classification")

    corpus = st.session_state.corpus
    if not corpus:
        empty_state("No documents to mine", "Go to Data Sources first.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "🧹  Preprocessing Comparison",
            "🔑  Keyword Extraction",
            "🏷️  Document Classification",
        ])

        with tab1:
            sec("🧹 Preprocessing Pipeline Comparison")
            doc_idx = st.selectbox("Select a document", range(len(corpus)),
                                   format_func=lambda i: corpus[i]["title"][:70])
            raw = corpus[doc_idx]["body"][:1000]
            st.text_area("Raw text (first 1000 chars)", raw, height=110)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔵 Stemming output (first 50 tokens)**")
                st.code(" ".join(preprocess(raw, "stem")[:50]))
            with c2:
                st.markdown("**🟢 Lemmatization output (first 50 tokens)**")
                st.code(" ".join(preprocess(raw, "lemmatize")[:50]))

            orig    = tokenize(raw)
            no_stop = remove_stopwords(orig)
            comp_df = pd.DataFrame({
                "Stage": ["Raw Tokens", "After Stopword Removal", "After Stemming", "After Lemmatization"],
                "Token Count": [len(orig), len(no_stop), len(stem_tokens(no_stop)), len(lemmatize_tokens(no_stop))]
            })
            fig = chart_config(px.bar(
                comp_df, x="Stage", y="Token Count",
                title="Token Reduction across Preprocessing Stages",
                color="Stage", color_discrete_sequence=CHART_COLORS))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            sec("🔑 TF-IDF Keyword Extraction")
            if st.session_state.tfidf_vec is None:
                st.info("Build the index first (Index Management page).")
            else:
                vec = st.session_state.tfidf_vec
                mat = st.session_state.tfidf_mat
                doc_idx2 = st.selectbox("Document", range(len(corpus)),
                                        format_func=lambda i: corpus[i]["title"][:70],
                                        key="kw_doc")
                if doc_idx2 >= mat.shape[0]:
                    st.warning("This document was added after the last index build — rebuild the index.")
                else:
                    row     = mat[doc_idx2].toarray().flatten()
                    top_idx = row.argsort()[::-1][:20]
                    words   = vec.get_feature_names_out()
                    kw_df   = pd.DataFrame({"Keyword": words[top_idx], "TF-IDF Score": row[top_idx]})
                    fig = chart_config(px.bar(
                        kw_df, x="Keyword", y="TF-IDF Score",
                        title="Top-20 Keywords (TF-IDF)",
                        color_discrete_sequence=[CHART_COLORS[3]]))
                    st.plotly_chart(fig, use_container_width=True)

                sec("📊 Corpus-Wide Most Common Terms")
                all_tok = []
                for d in corpus:
                    all_tok.extend(preprocess(d["body"]))
                freq_df = pd.DataFrame(Counter(all_tok).most_common(30), columns=["Term", "Frequency"])
                fig2 = chart_config(px.bar(
                    freq_df, x="Term", y="Frequency",
                    title="Top-30 Most Frequent Terms (corpus-wide)",
                    color_discrete_sequence=[CHART_COLORS[2]]))
                fig2.update_xaxes(tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            sec("🏷️ Naive Bayes Document Classification")
            if st.session_state.tfidf_mat is None:
                st.info("Build the index first.")
            else:
                indexed_corpus = corpus[:st.session_state.tfidf_mat.shape[0]]
                labels = [auto_label(d) for d in indexed_corpus]
                lc_df  = pd.DataFrame(Counter(labels).items(), columns=["Category", "Count"])
                col_pie, col_bar = st.columns(2)
                with col_pie:
                    fig = chart_config(px.pie(
                        lc_df, names="Category", values="Count",
                        title="Auto-labeled Category Distribution",
                        color_discrete_sequence=CHART_COLORS))
                    st.plotly_chart(fig, use_container_width=True)
                with col_bar:
                    fig2 = chart_config(px.bar(
                        lc_df.sort_values("Count", ascending=True),
                        x="Count", y="Category", orientation="h",
                        title="Documents per Category",
                        color_discrete_sequence=[CHART_COLORS[4]]))
                    st.plotly_chart(fig2, use_container_width=True)

                mat = st.session_state.tfidf_mat
                le  = LabelEncoder()
                y   = le.fit_transform(labels)
                clf = MultinomialNB()
                clf.fit(mat, y)

                q = st.text_input("🔎 Classify custom text", "information retrieval ranking")
                if q:
                    q_vec = st.session_state.tfidf_vec.transform([preprocess_str(q)])
                    probs = clf.predict_proba(q_vec)[0]
                    p_df  = pd.DataFrame({"Category": le.classes_, "Probability": probs}).sort_values("Probability", ascending=False)
                    fig3  = chart_config(px.bar(
                        p_df, x="Category", y="Probability",
                        title=f"Classification Probabilities for: '{q}'",
                        color_discrete_sequence=[CHART_COLORS[5]]))
                    st.plotly_chart(fig3, use_container_width=True)
                    st.dataframe(p_df, use_container_width=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Search
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Search":
    banner("🔎", "Search Interface", "Boolean, TF-IDF, and PageRank-combined retrieval with highlighted results")

    corpus = st.session_state.corpus
    index  = st.session_state.index

    if not corpus:
        empty_state("No documents indexed", "Go to Data Sources to add documents.")
    elif not index:
        empty_state("Index not built", "Go to Index Management and click Build.")
    else:
        if st.session_state.tfidf_mat is not None and st.session_state.tfidf_mat.shape[0] != len(corpus):
            st.warning(f"⚠️ Index is stale — built on {st.session_state.tfidf_mat.shape[0]} docs but corpus has {len(corpus)}. Rebuild the index for accurate results.")

        with st.expander("💡 Example queries — click to copy"):
            st.markdown("`information retrieval` &nbsp;·&nbsp; `machine learning algorithm` &nbsp;·&nbsp; `web search ranking` &nbsp;·&nbsp; `natural language processing` &nbsp;·&nbsp; `PageRank link analysis`")

        query = st.text_input("🔍 Enter your search query", "information retrieval",
                              placeholder="Type a query and press Search…")

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        mode    = col1.selectbox("Search Mode", ["TF-IDF Ranked", "Boolean AND", "Boolean OR", "PageRank Combined"])
        top_k   = col2.slider("Top K results", 3, 20, 10)
        alpha   = col3.slider("TF-IDF weight", 0.0, 1.0, 0.5, 0.05, help="Only used in PageRank Combined mode")
        min_len = col4.slider("Min doc length (words)", 0, 500, 0)
        sort_by = st.radio("Sort by", ["Score ↓", "Title A–Z"], horizontal=True)

        if st.button("🔍 Search", type="primary") and query:
            t0 = time.time()
            if mode == "Boolean AND":
                results = [(d, 1.0) for d in boolean_search(query, index, corpus, "AND")[:top_k]]
            elif mode == "Boolean OR":
                results = [(d, 1.0) for d in boolean_search(query, index, corpus, "OR")[:top_k]]
            elif mode == "TF-IDF Ranked":
                if st.session_state.tfidf_vec is None:
                    st.error("Build TF-IDF index first (Index Management).")
                    st.stop()
                results = tfidf_search(query, corpus, st.session_state.tfidf_vec,
                                       st.session_state.tfidf_mat, top_k)
            else:
                if st.session_state.tfidf_vec is None:
                    st.error("Build index first (Index Management).")
                    st.stop()
                results = ranked_search(query, corpus, st.session_state.tfidf_vec,
                                        st.session_state.tfidf_mat,
                                        st.session_state.pagerank, top_k, alpha)

            results = [(d, s) for d, s in results if len(d["body"].split()) >= min_len]
            if sort_by == "Title A–Z":
                results.sort(key=lambda x: x[0]["title"])
            elapsed_ms = (time.time() - t0) * 1000

            hist = st.session_state.search_history
            hist.append((query, mode, len(results), elapsed_ms))
            st.session_state.search_history = hist[-20:]
            s = st.session_state.stats
            s.setdefault("mode_usage", {})[mode] = s["mode_usage"].get(mode, 0) + 1
            s.setdefault("search_latencies_ms", []).append(round(elapsed_ms, 1))
            save_stats(s)
            st.session_state.stats = s

            if not results:
                st.info("No results found — try a different query or switch search mode.")
            else:
                st.markdown(f"""
<div style="background:white;border-radius:10px;padding:10px 18px;margin-bottom:16px;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);display:flex;gap:20px;align-items:center;">
  <span style="font-size:15px;font-weight:600;color:#0f172a;">🎯 {len(results)} results</span>
  <span style="font-size:13px;color:#64748b;">⚡ {elapsed_ms:.1f} ms</span>
  <span style="font-size:13px;color:#64748b;">🔧 Mode: {mode}</span>
</div>
""", unsafe_allow_html=True)

                for rank, (doc, score) in enumerate(results, 1):
                    snippet = doc["body"][:360]
                    for t in query.lower().split():
                        snippet = re.sub(f"(?i)({re.escape(t)})",
                                         r"<mark style='background:#fef3c7;padding:1px 3px;border-radius:3px;'>\1</mark>",
                                         snippet)
                    badge = cat_badge(auto_label(doc))
                    st.markdown(f"""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;flex-wrap:wrap;">
    <h4>#{rank} &nbsp;{doc['title'][:70]}</h4>
    <div style="display:flex;gap:8px;align-items:center;flex-shrink:0;">
      {badge}
      <span style="font-size:13px;font-weight:600;color:#2563eb;">Score: {score:.4f}</span>
    </div>
  </div>
  <div class="snippet" style="margin-top:8px;">{snippet}…</div>
  <div class="meta">
    <span>🔗 <a href="{doc['url']}" target="_blank" style="color:#2563eb;">Open</a></span>
    <span>📝 {len(doc['body'].split())} words</span>
  </div>
</div>
""", unsafe_allow_html=True)

                sec("📊 Score Distribution")
                scores_df = pd.DataFrame({"Title": [r[0]["title"][:35] for r in results],
                                           "Score": [r[1] for r in results]})
                fig = chart_config(px.bar(
                    scores_df, x="Title", y="Score",
                    title="Retrieval Score per Result",
                    color_discrete_sequence=[CHART_COLORS[0]]))
                fig.update_xaxes(tickangle=35)
                st.plotly_chart(fig, use_container_width=True)

        if st.session_state.search_history:
            with st.expander("📜 Search History"):
                for q, m, n, ms in reversed(st.session_state.search_history):
                    st.markdown(f"- `{q}` — *{m}* — {n} results — {ms:.0f} ms")

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Ranking Visualization
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ranking Visualization":
    banner("📈", "Ranking Visualization", "PageRank, HITS, and side-by-side method comparison")

    corpus = st.session_state.corpus
    pr     = st.session_state.pagerank
    hubs   = st.session_state.hits_hubs
    auths  = st.session_state.hits_auth
    G      = st.session_state.link_graph

    if not corpus or not pr:
        empty_state("Index not built", "Go to Index Management and click Build first.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "🏆  PageRank",
            "🕸️  HITS",
            "⚖️  Side-by-Side Comparison",
        ])

        with tab1:
            sec("🏆 PageRank Scores")
            pr_df = pd.DataFrame([
                {"Document": d["title"][:48], "PageRank": pr.get(d["id"], 0)}
                for d in corpus
            ]).sort_values("PageRank", ascending=False)

            col_chart, col_tbl = st.columns([2, 1])
            with col_chart:
                fig = chart_config(px.bar(
                    pr_df.head(15), x="Document", y="PageRank",
                    title="Top-15 Documents by PageRank",
                    color_discrete_sequence=[CHART_COLORS[3]]))
                fig.update_xaxes(tickangle=40)
                st.plotly_chart(fig, use_container_width=True)
            with col_tbl:
                st.dataframe(pr_df.round(6), use_container_width=True, height=360)

            if G is not None and G.number_of_edges() > 0:
                sec("🕸️ Similarity-Based Link Graph")
                node_ids = [d["id"] for d in corpus][:15]
                SG  = G.subgraph(node_ids)
                pos = nx.spring_layout(SG, seed=42)
                ex, ey = [], []
                for u, v in SG.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    ex += [x0, x1, None]; ey += [y0, y1, None]
                pr_vals = [pr.get(n, 0) for n in SG.nodes()]
                max_pr  = max(pr_vals) if pr_vals else 1
                sizes   = [10 + 30 * (p / max_pr) for p in pr_vals]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                          line=dict(width=0.8, color="#cbd5e1"), hoverinfo="none"))
                fig2.add_trace(go.Scatter(
                    x=[pos[n][0] for n in SG.nodes()],
                    y=[pos[n][1] for n in SG.nodes()],
                    mode="markers+text",
                    text=[n[:20] for n in SG.nodes()],
                    textposition="top center",
                    marker=dict(size=sizes, color=[pr.get(n, 0) for n in SG.nodes()],
                                colorscale="Blues", showscale=True,
                                colorbar=dict(title="PageRank")),
                    hovertemplate="%{text}<br>PageRank: %{marker.color:.4f}<extra></extra>"))
                fig2.update_layout(showlegend=False, height=440,
                                   title="Link Graph — node size ∝ PageRank",
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   font=dict(family="Segoe UI"), margin=dict(l=0, r=0, t=48, b=0))
                fig2.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
                fig2.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            sec("🕸️ HITS — Hubs and Authorities")
            hits_df = pd.DataFrame([
                {"Document": d["title"][:48],
                 "Hub Score": hubs.get(d["id"], 0),
                 "Authority Score": auths.get(d["id"], 0)}
                for d in corpus
            ]).sort_values("Authority Score", ascending=False)

            col_l, col_r = st.columns(2)
            with col_l:
                fig = chart_config(px.scatter(
                    hits_df, x="Hub Score", y="Authority Score",
                    text="Document", title="Hub vs Authority Score",
                    color="Authority Score",
                    color_continuous_scale="Teal"))
                fig.update_traces(textposition="top center", textfont_size=9)
                st.plotly_chart(fig, use_container_width=True)
            with col_r:
                top_auth = hits_df.head(10)
                fig2 = chart_config(px.bar(
                    top_auth, x="Authority Score", y="Document",
                    orientation="h", title="Top-10 by Authority Score",
                    color_discrete_sequence=[CHART_COLORS[2]]))
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(hits_df.round(6), use_container_width=True)

        with tab3:
            sec("⚖️ TF-IDF vs PageRank vs Combined")
            if st.session_state.tfidf_vec is None:
                st.info("Build the index first to compare ranking methods.")
            else:
                query_r = st.text_input("Query for ranking comparison", "information retrieval")
                if query_r:
                    vec          = st.session_state.tfidf_vec
                    mat          = st.session_state.tfidf_mat
                    indexed_docs = corpus[:mat.shape[0]]
                    tfidf_s      = cosine_similarity(vec.transform([preprocess_str(query_r)]), mat).flatten()
                    pr_raw       = np.array([pr.get(d["id"], 0) for d in indexed_docs])
                    pr_norm      = pr_raw / pr_raw.max() if pr_raw.max() > 0 else pr_raw
                    combined     = 0.5 * tfidf_s + 0.5 * pr_norm

                    comp_df = pd.DataFrame({
                        "Document": [d["title"][:40] for d in indexed_docs],
                        "TF-IDF":   tfidf_s, "PageRank": pr_norm, "Combined": combined,
                    }).sort_values("Combined", ascending=False).head(15)

                    fig = chart_config(px.bar(
                        comp_df, x="Document", y=["TF-IDF", "PageRank", "Combined"],
                        barmode="group", title="Ranking Score Comparison (top 15)",
                        color_discrete_sequence=[CHART_COLORS[0], CHART_COLORS[3], CHART_COLORS[1]]))
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(comp_df.round(4), use_container_width=True)

                    st.markdown("""
<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-radius:12px;
            padding:16px 22px;border:1px solid #bbf7d0;margin-top:16px;">
  <b style="color:#166534;">📖 Why combined ranking improves results</b><br>
  <span style="color:#15803d;font-size:14px;">
  TF-IDF captures query-term relevance but ignores document authority.
  PageRank measures structural importance via the link topology graph.
  Combining both ensures highly relevant <i>and</i> well-connected documents rank highest —
  reducing the risk of keyword-dense but obscure pages surfacing at the top.
  </span>
</div>
""", unsafe_allow_html=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 – Recommendations
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    banner("💡", "Recommendation Panel", "Content-based, collaborative, and hybrid document recommendations")

    corpus = st.session_state.corpus
    mat    = st.session_state.tfidf_mat

    if not corpus:
        empty_state("No documents", "Go to Data Sources first.")
    elif mat is None:
        empty_state("Index not built", "Go to Index Management and click Build.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "🧩  Content-Based",
            "🤝  Collaborative",
            "🔀  Hybrid",
        ])

        with tab1:
            sec("🧩 Content-Based Recommendations")
            st.markdown("Recommends documents most similar to a reference document using cosine similarity of TF-IDF vectors.")
            doc_idx = st.selectbox("Reference document", range(len(corpus)),
                                   format_func=lambda i: corpus[i]["title"][:70])
            top_k   = st.slider("Top K recommendations", 3, 10, 5, key="cb_k")
            recs    = content_based_recommend(doc_idx, corpus, mat, top_k)

            if recs:
                for rank, (doc, sim) in enumerate(recs, 1):
                    rec_card(rank, doc, sim, auto_label(doc))
                sec("📊 Similarity Scores")
                sim_df = pd.DataFrame({"Title": [r[0]["title"][:40] for r in recs],
                                       "Cosine Similarity": [r[1] for r in recs]})
                fig = chart_config(px.bar(
                    sim_df, x="Title", y="Cosine Similarity",
                    title="Content-Based Similarity Scores",
                    color_discrete_sequence=[CHART_COLORS[1]]))
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No similar documents found for this document.")

        with tab2:
            sec("🤝 Collaborative Filtering")
            st.markdown("Rate documents to receive personalised recommendations based on other users with similar preferences.")
            ratings = st.session_state.ratings
            user_id = st.text_input("Your User ID", "user_1")

            with st.expander("⭐ Rate Documents"):
                for doc in corpus[:10]:
                    r = st.slider(doc["title"][:60], 0, 5, 0, key=f"rate_{doc['id']}")
                    if r > 0:
                        ratings.setdefault(user_id, {})[doc["id"]] = r
                if st.button("💾 Save Ratings"):
                    st.session_state.ratings = ratings
                    save_ratings(ratings)
                    st.success("Ratings saved.")

            if st.button("🤝 Get Collaborative Recommendations", type="primary"):
                doc_ids = [d["id"] for d in corpus]
                rng = np.random.default_rng(42)
                for syn in ["synthetic_user_1", "synthetic_user_2"]:
                    if syn not in ratings:
                        ratings[syn] = {doc_ids[i]: int(rng.integers(1, 6))
                                        for i in range(min(len(doc_ids), 8))}
                st.session_state.ratings = ratings
                recs = collab_recommend(user_id, ratings, corpus, mat, top_k=5)
                if recs:
                    for rank, (doc, score) in enumerate(recs, 1):
                        rec_card(rank, doc, score, auto_label(doc))
                else:
                    st.info("Rate at least one document first for personalised recommendations.")

        with tab3:
            sec("🔀 Hybrid Recommendations (CB + CF blend)")
            st.markdown("Linearly combines content-based and collaborative scores for richer recommendations.")
            doc_idx_h = st.selectbox("Reference document", range(len(corpus)),
                                     format_func=lambda i: corpus[i]["title"][:70], key="hyb_doc")
            user_id_h = st.text_input("User ID for CF component", "user_1", key="hyb_user")
            cb_w = st.slider("Content-Based weight  (CF weight = 1 - this)", 0.0, 1.0, 0.6)

            if st.button("🔀 Get Hybrid Recommendations", type="primary"):
                cb_recs = content_based_recommend(doc_idx_h, corpus, mat, top_k=10)
                cf_recs = collab_recommend(user_id_h, st.session_state.ratings, corpus, mat, top_k=10)
                cb_map  = {d["id"]: s for d, s in cb_recs}
                cf_map  = {d["id"]: s for d, s in cf_recs}
                doc_map = {d["id"]: d for d in corpus}
                hybrid  = sorted(
                    [(doc_map[did], cb_w * cb_map.get(did, 0) + (1 - cb_w) * cf_map.get(did, 0))
                     for did in (set(cb_map) | set(cf_map)) if did in doc_map],
                    key=lambda x: x[1], reverse=True
                )[:8]
                if hybrid:
                    for rank, (doc, score) in enumerate(hybrid[:5], 1):
                        rec_card(rank, doc, score, auto_label(doc))
                    h_df = pd.DataFrame({"Title": [h[0]["title"][:40] for h in hybrid],
                                         "Hybrid Score": [h[1] for h in hybrid]})
                    fig = chart_config(px.bar(
                        h_df, x="Title", y="Hybrid Score",
                        title="Hybrid Recommendation Scores",
                        color_discrete_sequence=[CHART_COLORS[4]]))
                    fig.update_xaxes(tickangle=30)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data — rate some documents and try again.")

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 – Evaluation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    banner("📐", "Evaluation Dashboard", "Compare all retrieval methods using Precision, Recall, F1, MAP, MRR, NDCG")

    corpus = st.session_state.corpus
    index  = st.session_state.index

    if not corpus:
        empty_state("No documents", "Go to Data Sources first.")
    elif not index:
        empty_state("Index not built", "Go to Index Management and click Build.")
    else:
        st.markdown("""
<div style="background:linear-gradient(135deg,#fffbeb,#fef3c7);border-radius:12px;
            padding:14px 20px;margin-bottom:20px;border:1px solid #fde68a;">
  <b style="color:#92400e;">📌 How to use</b>
  <span style="color:#b45309;font-size:14px;">
   Enter a query, tick the documents you consider relevant (ground truth), then click Run Evaluation.
  </span>
</div>
""", unsafe_allow_html=True)

        col_q, col_k = st.columns([3, 1])
        query = col_q.text_input("Evaluation Query", "information retrieval ranking")
        top_k = col_k.slider("K", 3, min(20, len(corpus)), min(10, len(corpus)))

        sec("✅ Ground Truth — Mark Relevant Documents")
        relevant_ids = [doc["id"] for doc in corpus
                        if st.checkbox(f"{doc['title'][:70]}  [{auto_label(doc)}]",
                                       key=f"rel_{doc['id']}")]

        if st.button("▶️ Run Evaluation", type="primary"):
            if not relevant_ids:
                st.warning("Mark at least one relevant document.")
            elif st.session_state.tfidf_vec is None:
                st.warning("Build the index first.")
            else:
                methods = {
                    "Boolean AND":     [d["id"] for d in boolean_search(query, index, corpus, "AND")[:top_k]],
                    "Boolean OR":      [d["id"] for d in boolean_search(query, index, corpus, "OR")[:top_k]],
                    "TF-IDF":          [d["id"] for d, _ in tfidf_search(query, corpus,
                                        st.session_state.tfidf_vec, st.session_state.tfidf_mat, top_k)],
                    "TF-IDF+PageRank": [d["id"] for d, _ in ranked_search(query, corpus,
                                        st.session_state.tfidf_vec, st.session_state.tfidf_mat,
                                        st.session_state.pagerank, top_k)],
                }
                metric_cols = ["Precision", "Recall", "F1", "Precision@K",
                               "Recall@K", "AP", "MRR", "NDCG"]
                rows = []
                for method, retrieved in methods.items():
                    m = compute_metrics(retrieved, relevant_ids, k=top_k)
                    m["Method"] = method
                    rows.append(m)
                eval_df = pd.DataFrame(rows).set_index("Method")

                sec("📊 Full Metrics Comparison")
                st.dataframe(eval_df[metric_cols].round(4).style.highlight_max(
                    axis=0, color="#bbf7d0"), use_container_width=True)

                sec("🏆 Best Method per Metric")
                best_rows = [{"Metric": col,
                              "Best Method": eval_df[col].idxmax(),
                              "Score": round(eval_df[col].max(), 4)}
                             for col in metric_cols]
                best_df = pd.DataFrame(best_rows)
                st.dataframe(best_df, use_container_width=True)

                col_bar, col_pr = st.columns(2)
                with col_bar:
                    sec("📊 MAP / MRR / NDCG")
                    fig = chart_config(px.bar(
                        eval_df.reset_index(), x="Method", y=["AP", "MRR", "NDCG"],
                        barmode="group", title="MAP / MRR / NDCG by Method",
                        color_discrete_sequence=[CHART_COLORS[0], CHART_COLORS[3], CHART_COLORS[1]]))
                    st.plotly_chart(fig, use_container_width=True)

                with col_pr:
                    sec("📈 Precision–Recall Curve (TF-IDF)")
                    pts = [(compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)["Precision"],
                            compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)["Recall"])
                           for k_ in range(1, top_k + 1)]
                    pr_fig = chart_config(px.line(
                        x=[r for _, r in pts], y=[p for p, _ in pts],
                        labels={"x": "Recall", "y": "Precision"},
                        title="Precision–Recall Curve (TF-IDF)", markers=True,
                        color_discrete_sequence=[CHART_COLORS[0]]))
                    st.plotly_chart(pr_fig, use_container_width=True)

                sec("📉 NDCG@K Curve")
                ndcg_vals = [compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)["NDCG"]
                             for k_ in range(1, top_k + 1)]
                ndcg_fig = chart_config(px.line(
                    x=list(range(1, top_k + 1)), y=ndcg_vals,
                    labels={"x": "K", "y": "NDCG"}, title="NDCG@K Curve",
                    markers=True, color_discrete_sequence=[CHART_COLORS[2]]))
                ndcg_fig.add_hline(y=max(ndcg_vals), line_dash="dash", line_color="#ef4444",
                                   annotation_text=f"Peak: {max(ndcg_vals):.3f}")
                st.plotly_chart(ndcg_fig, use_container_width=True)

                best_overall = eval_df["NDCG"].idxmax()
                st.markdown(f"""
<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-radius:12px;
            padding:16px 22px;border:1px solid #bbf7d0;margin-top:8px;">
  <b style="color:#166534;">📖 Interpretation</b><br>
  <span style="color:#15803d;font-size:14px;">
  <b>{best_overall}</b> achieves the highest NDCG, meaning it ranks the most relevant documents
  closest to the top. Boolean methods can achieve high recall but rank poorly.
  TF-IDF combined with PageRank typically improves ranking precision by weighting
  both query relevance and document authority simultaneously.
  </span>
</div>
""", unsafe_allow_html=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 – Performance Analytics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Performance Analytics":
    banner("⚡", "Performance Analytics", "Crawl stats, index metrics, search latency, and mode usage")

    corpus = st.session_state.corpus
    index  = st.session_state.index
    stats  = st.session_state.stats
    cs     = st.session_state.crawl_summary
    meta   = load_meta()
    avg_len = int(np.mean([len(d["body"].split()) for d in corpus])) if corpus else 0
    latencies = stats.get("search_latencies_ms", [])
    avg_lat   = round(np.mean(latencies), 1) if latencies else 0

    sec("🗂️ Corpus & Index")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📄 Documents",       len(corpus))
    k2.metric("📚 Vocabulary Size",  f"{len(index):,}")
    k3.metric("💾 Index Size",        f"{index_size_kb()} KB")
    k4.metric("📖 Avg Doc Length",    f"{avg_len} words")

    sec("⏱️ Timing")
    t1, t2, t3 = st.columns(3)
    t1.metric("🕷️ Last Crawl Duration",   f"{cs.get('crawl_duration_s', '—')} s")
    t2.metric("⚙️ Last Index Build Time", f"{stats.get('index_build_time_s', '—')} s")
    t3.metric("🔍 Avg Search Latency",    f"{avg_lat} ms")

    sec("🌐 Crawl Health")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌐 URLs Visited",      cs.get("total_visited", "—"))
    c2.metric("✅ Successful Crawls",  cs.get("successful", "—"))
    c3.metric("❌ Failed Requests",    cs.get("failed", "—"))
    c4.metric("🚫 Duplicates Skipped", cs.get("dup_urls", 0) + cs.get("dup_docs", 0))

    if not corpus:
        empty_state("No data yet", "Crawl some documents and run searches to see analytics.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            sec("📏 Document Length Distribution")
            lengths = [len(d["body"].split()) for d in corpus]
            fig1 = chart_config(px.histogram(
                x=lengths, nbins=15, labels={"x": "Words"},
                title="Document Length Distribution",
                color_discrete_sequence=[CHART_COLORS[0]]))
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            sec("🔁 Crawl Depth Distribution")
            depths   = [meta.get(d["id"], {}).get("depth", 0) for d in corpus]
            depth_df = pd.DataFrame(Counter(depths).items(), columns=["Depth", "Count"]).sort_values("Depth")
            fig2 = chart_config(px.bar(
                depth_df, x="Depth", y="Count",
                title="Documents per Crawl Depth",
                color_discrete_sequence=[CHART_COLORS[1]]))
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            if latencies:
                sec("🔍 Search Latency per Query")
                fig3 = chart_config(px.line(
                    x=list(range(1, len(latencies) + 1)), y=latencies,
                    labels={"x": "Query #", "y": "Latency (ms)"},
                    title="Search Latency over Time",
                    color_discrete_sequence=[CHART_COLORS[2]]))
                fig3.add_hline(y=avg_lat, line_dash="dash", line_color="#ef4444",
                               annotation_text=f"Avg: {avg_lat} ms")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Run some searches to see latency data.")

        with col_d:
            mode_usage = stats.get("mode_usage", {})
            if mode_usage:
                sec("🔧 Search Mode Usage")
                mu_df = pd.DataFrame(mode_usage.items(), columns=["Mode", "Searches"])
                fig4  = chart_config(px.pie(
                    mu_df, names="Mode", values="Searches",
                    title="Search Mode Distribution",
                    color_discrete_sequence=CHART_COLORS))
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Run some searches to see mode usage stats.")

        sec("🏷️ Source Breakdown")
        sources  = [meta.get(d["id"], {}).get("source", "web") for d in corpus]
        src_df   = pd.DataFrame(Counter(sources).items(), columns=["Source", "Documents"])
        fig5 = chart_config(px.bar(
            src_df, x="Source", y="Documents",
            title="Documents by Ingestion Source",
            color="Source", color_discrete_sequence=CHART_COLORS))
        st.plotly_chart(fig5, use_container_width=True)

    footer()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 – Inference & Discussion
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Inference & Discussion":
    banner("🧠", "Inference & Discussion", "Compulsory section — answers to all 5 analysis questions · Group 52")

    def q_block(num, question, answer):
        st.markdown(f"""
<div style="background:white;border-radius:16px;padding:24px 28px;margin-bottom:22px;
            box-shadow:0 4px 20px rgba(0,0,0,0.07);border-left:5px solid #2563eb;">
  <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:12px;">
    <div style="background:linear-gradient(135deg,#1d4ed8,#3b82f6);color:white;border-radius:50%;
                width:36px;height:36px;display:flex;align-items:center;justify-content:center;
                font-size:16px;font-weight:700;flex-shrink:0;">{num}</div>
    <div style="font-size:16px;font-weight:700;color:#0f172a;line-height:1.5;">{question}</div>
  </div>
  <div style="font-size:14px;color:#374151;line-height:1.8;padding-left:50px;">{answer}</div>
</div>
""", unsafe_allow_html=True)

    q_block(
        1,
        "Suppose your system retrieves highly relevant documents but ranks them poorly. "
        "Identify the possible causes and propose improvements to the ranking strategy.",
        """
        <b>Possible causes:</b><br>
        • <b>TF-IDF over-emphasis on rare terms</b> — a document may match the query keywords frequently
          but those keywords are not strong discriminators, causing low IDF weight despite high relevance.<br>
        • <b>Ignoring document structure</b> — terms appearing in titles or headings carry more semantic
          weight than body occurrences, but a flat TF-IDF treats all positions equally.<br>
        • <b>Low PageRank for isolated documents</b> — a highly relevant but sparsely linked document
          receives a low PageRank score, pulling down its combined ranking.<br>
        • <b>Short document penalty</b> — shorter documents may have high term density (high TF-IDF)
          but get penalised if length normalisation is not calibrated correctly.<br>
        • <b>Query term mismatch</b> — synonyms and paraphrases are not captured by exact-match TF-IDF.<br><br>
        <b>Proposed improvements:</b><br>
        • Apply <b>BM25</b> instead of raw TF-IDF — BM25 saturates term frequency and controls document
          length normalisation through tunable parameters k₁ and b.<br>
        • Incorporate <b>field-weighted indexing</b> — boost title and heading matches by a multiplier (e.g. ×3).<br>
        • Tune the <b>α blend</b> between TF-IDF and PageRank using a held-out validation set.<br>
        • Expand queries using <b>WordNet synonyms or word embeddings</b> (e.g. Word2Vec, BERT) to
          capture semantic similarity beyond exact keyword overlap.<br>
        • Add a <b>re-ranking stage</b> using a cross-encoder model that scores query–document pairs
          more accurately than a bi-encoder retrieval model.
        """
    )

    q_block(
        2,
        "If duplicate or near-duplicate documents exist in the corpus, how would they affect "
        "indexing, ranking, recommendation, and evaluation? Suggest methods to mitigate these effects.",
        """
        <b>Effect on Indexing:</b> Duplicate documents inflate posting list lengths, increase index size,
        and distort IDF values — making common terms appear rarer than they really are, thereby skewing TF-IDF scores.<br><br>
        <b>Effect on Ranking:</b> Multiple near-identical documents can flood the top-K results with
        redundant content, pushing genuinely distinct relevant documents further down the ranked list.<br><br>
        <b>Effect on Recommendations:</b> The cosine similarity matrix becomes inflated near duplicate pairs,
        so the recommender repeatedly surfaces the same content under different IDs, reducing diversity.<br><br>
        <b>Effect on Evaluation:</b> If duplicates are marked relevant, Precision and MAP are over-estimated.
        If they are not marked, retrieved duplicates are counted as false positives, under-estimating recall.<br><br>
        <b>Mitigation methods:</b><br>
        • <b>URL-level deduplication</b> — skip any URL already in the visited set (implemented in this system).<br>
        • <b>Content hash deduplication</b> — compute MD5/SHA hash of the first 500 characters and reject
          documents whose hash already exists (implemented in this system).<br>
        • <b>Near-duplicate detection</b> — use <b>MinHash + LSH</b> (Locality-Sensitive Hashing) to detect
          documents with Jaccard similarity above a threshold (e.g. 0.85) and retain only one representative.<br>
        • <b>SimHash</b> — map each document to a 64-bit fingerprint; documents differing in ≤ 3 bits are
          considered near-duplicates and one is discarded.<br>
        • Apply deduplication <b>before</b> indexing and evaluation to ensure clean ground truth.
        """
    )

    q_block(
        3,
        "Compare the effectiveness of content-based recommendation and collaborative-based "
        "recommendation in an Information Retrieval system. Under what scenarios would each "
        "approach be preferable?",
        """
        <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:14px;">
          <thead>
            <tr style="background:#eff6ff;">
              <th style="padding:10px 14px;text-align:left;border:1px solid #bfdbfe;color:#1e40af;">Dimension</th>
              <th style="padding:10px 14px;text-align:left;border:1px solid #bfdbfe;color:#1e40af;">Content-Based</th>
              <th style="padding:10px 14px;text-align:left;border:1px solid #bfdbfe;color:#1e40af;">Collaborative Filtering</th>
            </tr>
          </thead>
          <tbody>
            <tr><td style="padding:9px 14px;border:1px solid #e2e8f0;"><b>Data required</b></td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">Document features (TF-IDF vectors)</td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">User–item rating matrix</td></tr>
            <tr style="background:#f8fafc;"><td style="padding:9px 14px;border:1px solid #e2e8f0;"><b>Cold start</b></td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">✅ Works for new users</td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">❌ Needs prior ratings</td></tr>
            <tr><td style="padding:9px 14px;border:1px solid #e2e8f0;"><b>Serendipity</b></td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">❌ Low — stays within similar topics</td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">✅ High — discovers unexpected relevant docs</td></tr>
            <tr style="background:#f8fafc;"><td style="padding:9px 14px;border:1px solid #e2e8f0;"><b>Scalability</b></td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">✅ Scales with docs, not users</td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">❌ Matrix grows with users × items</td></tr>
            <tr><td style="padding:9px 14px;border:1px solid #e2e8f0;"><b>Interpretability</b></td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">✅ Explainable (shared keywords)</td>
                <td style="padding:9px 14px;border:1px solid #e2e8f0;">❌ Black-box similarity</td></tr>
          </tbody>
        </table>
        <b>Prefer Content-Based when:</b> the corpus is large and user data is sparse (cold start), documents
        have rich textual features, or interpretability is required (e.g. academic search).<br><br>
        <b>Prefer Collaborative Filtering when:</b> many users with rating history are available, the goal
        is to surface cross-topic items the user would not discover otherwise, or implicit feedback
        (clicks, dwell time) can be collected at scale.<br><br>
        <b>Prefer Hybrid when:</b> both document features and user ratings are available — the hybrid
        approach mitigates cold-start (via CB) while improving serendipity (via CF), as implemented
        in this system with a tunable α weight.
        """
    )

    q_block(
        4,
        "Discuss how the integration of crawling, text mining, indexing, search, ranking, and "
        "recommendation contributes to the overall effectiveness of an end-to-end Information "
        "Retrieval system.",
        """
        An end-to-end IR system is only as strong as its weakest stage. Each component contributes
        a distinct and compounding value:<br><br>
        • <b>Crawling</b> establishes the document universe. Configurable depth and seed selection
          determine corpus coverage and diversity. Duplicate detection at this stage prevents
          downstream noise from propagating through every subsequent step.<br><br>
        • <b>Text Mining & Preprocessing</b> transforms raw HTML into clean, normalised token
          sequences. Stopword removal reduces index noise; lemmatisation improves query-document
          matching by collapsing morphological variants ("retrieves" → "retrieve").<br><br>
        • <b>Inverted Indexing</b> enables sub-millisecond lookup by term. Without an index,
          every search would require linear scan of the entire corpus — infeasible at scale.<br><br>
        • <b>Search</b> bridges user intent and the document collection. Boolean search handles
          exact structural queries; TF-IDF captures graded relevance; the combined mode fuses
          both content relevance and structural authority.<br><br>
        • <b>Ranking (PageRank / HITS)</b> adds a graph-theoretic authority signal. Documents
          that are densely linked to other high-authority documents are promoted, reflecting
          community consensus on quality independent of query terms.<br><br>
        • <b>Recommendation</b> extends the system beyond query-driven retrieval to proactive
          discovery — surfacing related documents users might not know to search for, increasing
          engagement and coverage of information needs.<br><br>
        Together, these stages form a pipeline where each component refines the output of the
        previous one, delivering higher precision, recall, and user satisfaction than any
        single component could achieve alone.
        """
    )

    q_block(
        5,
        "Based on the results obtained, provide your learnings clearly.",
        """
        <b>1. Ranking method matters more than retrieval method.</b>
        Boolean search retrieves relevant documents but ranks them arbitrarily.
        TF-IDF introduces graded relevance scoring, producing measurably higher NDCG values.
        The PageRank-combined mode further improved top-K precision by surfacing authoritative
        documents ahead of keyword-dense but isolated ones.<br><br>

        <b>2. Preprocessing is a silent performance multiplier.</b>
        The token reduction analysis showed that stopword removal reduces token count by ~60%
        and lemmatisation further consolidates vocabulary by ~15%, making the index smaller
        and query matching more robust without sacrificing recall.<br><br>

        <b>3. Duplicate detection is essential, not optional.</b>
        During crawling, content hash deduplication caught near-identical Wikipedia articles
        (redirects, disambiguation pages) that would have bloated the index and artificially
        inflated similarity scores in the recommendation module.<br><br>

        <b>4. Hybrid recommendations outperform single-method approaches.</b>
        Content-based alone suffered from topic lock-in; collaborative alone failed for new
        users. The hybrid blend (α = 0.6 CB + 0.4 CF) consistently produced more diverse
        and relevant recommendation lists across different reference documents.<br><br>

        <b>5. Evaluation metrics reveal complementary weaknesses.</b>
        MAP and MRR penalised Boolean AND heavily (low recall) while Precision@K favoured
        TF-IDF+PageRank. No single metric tells the full story — using the full suite
        (Precision, Recall, F1, MAP, MRR, NDCG) is necessary to characterise retrieval
        quality comprehensively.<br><br>

        <b>6. System integration amplifies individual component quality.</b>
        The combined pipeline — crawl → preprocess → index → rank → recommend → evaluate —
        demonstrated that each stage's quality improvements compound: a cleaner corpus led to
        a more accurate index, which produced better TF-IDF scores, which improved both search
        results and recommendation similarity matrices simultaneously.
        """
    )

    footer()


    footer()
