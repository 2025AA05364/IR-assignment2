# IR Assignment 2 - Information Retrieval System
# Group 52
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

# Persistence
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

# Text preprocessing
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

# Crawling
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def crawl(seeds, max_depth=1, max_pages=20):
    visited_urls  = set()
    seen_hashes   = set()
    docs          = []
    meta          = {}
    failed        = 0
    dup_urls      = 0
    dup_docs      = 0
    total_size    = 0
    queue         = [(url.strip(), 0) for url in seeds if url.strip()]
    crawl_start   = time.time()

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
            body  = soup.get_text(separator=" ", strip=True)
            body  = re.sub(r"\s+", " ", body)[:5000]

            h = hash(body[:500])
            if h in seen_hashes or len(body) < 100:
                dup_docs += 1
                continue
            seen_hashes.add(h)

            doc_id = f"doc_{len(docs)}"
            page_size = len(body.encode("utf-8"))
            total_size += page_size
            docs.append({"id": doc_id, "url": url, "title": title, "body": body})
            meta[doc_id] = {
                "url": url, "title": title, "length": len(body),
                "depth": depth, "size_bytes": page_size,
                "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
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

    crawl_duration = round(time.time() - crawl_start, 2)
    progress.empty()
    status.empty()

    crawl_summary = {
        "total_visited": len(visited_urls),
        "successful": len(docs),
        "failed": failed,
        "dup_urls": dup_urls,
        "dup_docs": dup_docs,
        "avg_page_size_kb": round(total_size / max(len(docs), 1) / 1024, 2),
        "crawl_duration_s": crawl_duration,
        "last_crawl": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return docs, meta, crawl_summary

# Indexing
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
    if os.path.exists(INDEX_FILE):
        return round(os.path.getsize(INDEX_FILE) / 1024, 1)
    return 0.0

# PageRank & HITS
def build_pagerank(docs):
    G = nx.DiGraph()
    for d in docs:
        G.add_node(d["id"])
    vec, mat = build_tfidf(docs)
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
    hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
    return hubs, authorities

# Search
def boolean_search(query, index, docs, op="AND"):
    tokens = preprocess(query)
    if not tokens:
        return []
    sets = [set(index.get(t, {}).keys()) for t in tokens]
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
    tfidf_scores = cosine_similarity(vec.transform([preprocess_str(query)]), mat).flatten()
    pr_scores    = np.array([pr.get(d["id"], 0) for d in docs])
    if pr_scores.max() > 0:
        pr_scores = pr_scores / pr_scores.max()
    combined = alpha * tfidf_scores + (1 - alpha) * pr_scores
    ranked   = np.argsort(combined)[::-1][:top_k]
    return [(docs[i], float(combined[i])) for i in ranked if combined[i] > 0]

# Recommendations
CATEGORIES = {
    "ML/AI":    ["machine", "learning", "neural", "deep", "model", "algorithm"],
    "IR/Search":["retrieval", "search", "index", "query", "ranking"],
    "NLP":      ["language", "text", "nlp", "processing", "corpus"],
    "Web":      ["web", "internet", "crawl", "page", "link", "network"],
}

def auto_label(doc):
    t = (doc["title"] + " " + doc["body"][:200]).lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"

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

# Evaluation
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

# UI helpers
def footer():
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#94a3b8;font-size:13px;'>"
        f"IR Assignment 2 &nbsp;|&nbsp; Group 52 &nbsp;|&nbsp; "
        f"{time.strftime('%Y-%m-%d %H:%M')}"
        f"</div>",
        unsafe_allow_html=True,
    )

def rec_card(rank, doc, score, category=""):
    snippet = doc["body"][:200].replace("\n", " ") + "…"
    cat_badge = f"<span style='background:#2563eb;color:white;padding:2px 8px;border-radius:10px;font-size:12px;'>{category}</span>" if category else ""
    st.markdown(f"""
<div style='background:white;border-radius:12px;padding:16px 20px;
            box-shadow:0 4px 15px rgba(0,0,0,.07);margin-bottom:12px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <span style='font-weight:700;font-size:16px;color:#0f172a;'>#{rank} {doc['title'][:65]}</span>
    {cat_badge}
  </div>
  <div style='margin:6px 0;color:#64748b;font-size:13px;'>{snippet}</div>
  <div style='display:flex;gap:16px;margin-top:8px;font-size:13px;'>
    <span>🔗 <a href='{doc['url']}' target='_blank'>Open</a></span>
    <span>📊 Score: <b>{score:.4f}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

# Page config & global CSS
st.set_page_config(
    page_title="IR Assignment 2 - Group 52",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family:'Segoe UI',sans-serif; }
.main { background:#f5f7fb; }
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0f172a,#1e293b);
}
section[data-testid="stSidebar"] * { color:white !important; }
.block-container { padding-top:1rem; }
.big-title  { font-size:36px;font-weight:700;color:#0f172a; }
.sub-title  { color:#64748b;font-size:16px; }
div.stButton>button {
    width:100%;border-radius:10px;height:46px;font-size:15px;
    background:#2563eb;color:white;border:none;
}
div.stButton>button:hover { background:#1d4ed8; }
[data-testid="metric-container"] {
    background:white;border-radius:14px;padding:14px;
    box-shadow:0 4px 14px rgba(0,0,0,.07);
}
</style>
""", unsafe_allow_html=True)

# Session state
_defaults = {
    "corpus":       load_corpus(),
    "index":        load_index(),
    "tfidf_vec":    None,
    "tfidf_mat":    None,
    "pagerank":     {},
    "hits_hubs":    {},
    "hits_auth":    {},
    "link_graph":   None,
    "ratings":      load_ratings(),
    "stats":        load_stats(),
    "search_history": [],
    "crawl_summary":  {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
st.sidebar.markdown("# IR Assignment 2\n### Group 52\n---")

corpus     = st.session_state.corpus
index      = st.session_state.index
stats      = st.session_state.stats
meta_all   = load_meta()

st.sidebar.markdown(f"**📄 Documents:** {len(corpus)}")
st.sidebar.markdown(f"**📚 Index Terms:** {len(index)}")
idx_kb = index_size_kb()
st.sidebar.markdown(f"**💾 Index Size:** {idx_kb} KB")
last_crawl = st.session_state.crawl_summary.get("last_crawl") or meta_all and next(
    (v.get("crawled_at","–") for v in meta_all.values()), "–")
st.sidebar.markdown(f"**🕐 Last Crawl:** {last_crawl or '–'}")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Crawling",
    "Index Management",
    "Text Mining",
    "Search",
    "Ranking Visualization",
    "Recommendations",
    "Evaluation",
    "Performance Analytics",
])

# ---
# PAGE 1: Dashboard
# ---
if page == "Dashboard":
    left, right = st.columns([4, 1])
    with left:
        st.markdown("""
<div class='big-title'>📚 IR Assignment 2 - Information Retrieval System</div>
<div class='sub-title'>IR Assignment 2 - Group 52</div><br>
""", unsafe_allow_html=True)
    with right:
        st.success("🟢 Online")

    total_tokens = sum(sum(v.values()) for v in index.values())
    avg_len  = int(np.mean([len(d["body"].split()) for d in corpus])) if corpus else 0
    cs       = st.session_state.crawl_summary
    dup_skip = cs.get("dup_urls", 0) + cs.get("dup_docs", 0)

    st.markdown("## 📈 Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents",       len(corpus))
    c2.metric("📚 Vocabulary",       len(index))
    c3.metric("📝 Total Tokens",     total_tokens)
    c4.metric("📖 Avg Doc Length",   avg_len)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🌐 URLs Visited",     cs.get("total_visited", "–"))
    c6.metric("🚫 Duplicates Skipped", dup_skip)
    c7.metric("💾 Index Size (KB)",  idx_kb)
    avg_depth = round(np.mean([v.get("depth", 0) for v in meta_all.values()]), 1) if meta_all else 0
    c8.metric("🔁 Avg Crawl Depth",  avg_depth)

    st.info("""
Web crawling, text preprocessing, inverted index, TF-IDF search, PageRank, HITS, recommendations, and evaluation metrics - all in one Streamlit app.
""")

    if corpus:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Document Length Distribution")
            lengths = [len(d["body"].split()) for d in corpus]
            fig = px.histogram(x=lengths, nbins=20, labels={"x": "Word Count"},
                               title="Document Length Distribution",
                               color_discrete_sequence=["#2563EB"])
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Top-20 Index Terms")
            if index:
                df_terms = pd.DataFrame(
                    [(t, len(p)) for t, p in index.items()],
                    columns=["Term", "Doc Frequency"]
                ).nlargest(20, "Doc Frequency")
                fig2 = px.bar(df_terms, x="Term", y="Doc Frequency",
                              color_discrete_sequence=["#10B981"])
                fig2.update_xaxes(tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)

        if st.session_state.search_history:
            st.subheader("🕵️ Recent Searches")
            for q, mode, n, ms in st.session_state.search_history[-8:][::-1]:
                st.markdown(f"- `{q}` — *{mode}* — {n} results in {ms:.0f} ms")

        st.subheader("📋 Crawled Documents")
        df = pd.DataFrame([{"Title": d["title"][:70], "URL": d["url"],
                             "Length": len(d["body"].split()),
                             "Category": auto_label(d)} for d in corpus])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ No documents yet. Go to **Crawling** to fetch some pages.")

    footer()

# ---
# PAGE 2: Crawling
# ---
elif page == "Crawling":
    st.markdown("<div class='big-title'>🕷️ Data Sources</div><br>", unsafe_allow_html=True)
    st.markdown("Choose one or more sources to populate the corpus.")

    tab_crawl, tab_csv, tab_api = st.tabs([
        "☑ Web Crawling",
        "☑ Upload CSV Dataset",
        "☑ API (Optional)",
    ])

    # Tab 1: Web Crawling
    with tab_crawl:
        st.subheader("🌐 Web Crawling")
        st.markdown("Fetch pages from seed URLs. Handles duplicate URLs and duplicate documents automatically.")

        seeds_raw = st.text_area(
            "Seed URLs (one per line)",
            value="\n".join([
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/Natural_language_processing",
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://en.wikipedia.org/wiki/Search_engine",
                "https://en.wikipedia.org/wiki/PageRank",
            ]),
            height=150,
        )
        col1, col2 = st.columns(2)
        max_depth = col1.slider("Crawl Depth", 0, 2, 0,
                                help="0 = seed pages only; 1 = follow links one level deep")
        max_pages = col2.slider("Max Pages", 5, 50, 10)

        if st.button("🚀 Start Crawling", type="primary", key="btn_crawl"):
            seeds = [s.strip() for s in seeds_raw.strip().split("\n") if s.strip()]
            if not seeds:
                st.error("Enter at least one seed URL.")
            else:
                with st.spinner("Crawling…"):
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

                st.success(f"✅ Added **{len(added)}** new documents. Corpus total: **{len(st.session_state.corpus)}**")
                st.markdown("### 📊 Crawl Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("🌐 URLs Visited",     summary["total_visited"])
                s2.metric("✅ Successful",        summary["successful"])
                s3.metric("❌ Failed",            summary["failed"])
                s4.metric("🚫 Dup URLs Skipped",  summary["dup_urls"])
                s5, s6, s7 = st.columns(3)
                s5.metric("📄 Dup Docs Skipped",  summary["dup_docs"])
                s6.metric("📦 Avg Page Size",     f"{summary['avg_page_size_kb']} KB")
                s7.metric("⏱️ Duration",           f"{summary['crawl_duration_s']} s")

    # Tab 2: Upload CSV
    with tab_csv:
        st.subheader("📂 Upload CSV Dataset")
        st.markdown("""
Upload a **CSV file** with at least a text/content column.
Optionally include `title` and `url` columns — they will be auto-detected.

**Expected columns (flexible):**

| Column | Required | Description |
|--------|----------|-------------|
| `title` / `headline` / `name` | optional | Document title |
| `text` / `content` / `body` / `description` / `abstract` | **required** | Main text |
| `url` / `link` | optional | Source URL |
        """)

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded:
            try:
                df_csv = pd.read_csv(uploaded)
                st.write(f"**Preview** ({len(df_csv)} rows, {len(df_csv.columns)} columns)")
                st.dataframe(df_csv.head(5), use_container_width=True)

                # Auto-detect columns
                cols_lower = {c.lower(): c for c in df_csv.columns}
                body_candidates  = ["text", "content", "body", "description", "abstract", "article"]
                title_candidates = ["title", "headline", "name", "subject"]
                url_candidates   = ["url", "link", "source"]

                body_col  = next((cols_lower[c] for c in body_candidates if c in cols_lower), None)
                title_col = next((cols_lower[c] for c in title_candidates if c in cols_lower), None)
                url_col   = next((cols_lower[c] for c in url_candidates   if c in cols_lower), None)

                c1, c2, c3 = st.columns(3)
                body_col  = c1.selectbox("Text / Body column *", df_csv.columns,
                                          index=list(df_csv.columns).index(body_col) if body_col else 0)
                title_col = c2.selectbox("Title column (optional)",
                                          ["(none)"] + list(df_csv.columns),
                                          index=(["(none)"] + list(df_csv.columns)).index(title_col) if title_col else 0)
                url_col   = c3.selectbox("URL column (optional)",
                                          ["(none)"] + list(df_csv.columns),
                                          index=(["(none)"] + list(df_csv.columns)).index(url_col) if url_col else 0)

                max_rows = st.slider("Max rows to import", 10, min(5000, len(df_csv)), min(200, len(df_csv)))

                if st.button("📥 Import CSV into Corpus", type="primary", key="btn_csv"):
                    existing_bodies = {d["body"][:200] for d in st.session_state.corpus}
                    added_csv, dup_csv = [], 0
                    base = len(st.session_state.corpus)
                    meta = load_meta()

                    for i, row in df_csv.head(max_rows).iterrows():
                        body  = str(row[body_col]).strip()
                        # Edge case: skip NaN / null / empty cells
                        if (not body
                                or body.lower() in ("nan", "none", "null", "na", "n/a", "")
                                or body[:200] in existing_bodies):
                            dup_csv += 1
                            continue
                        existing_bodies.add(body[:200])

                        title = str(row[title_col]).strip() if title_col != "(none)" else body[:60]
                        url   = str(row[url_col]).strip()   if url_col   != "(none)" else f"csv://row_{i}"
                        doc_id = f"doc_{base + len(added_csv)}"

                        added_csv.append({"id": doc_id, "url": url, "title": title, "body": body[:5000]})
                        meta[doc_id] = {
                            "url": url, "title": title, "length": len(body),
                            "depth": 0, "size_bytes": len(body.encode()),
                            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "source": "csv",
                        }

                    st.session_state.corpus.extend(added_csv)
                    save_corpus(st.session_state.corpus)
                    save_meta(meta)

                    st.success(f"✅ Imported **{len(added_csv)}** documents from CSV. "
                               f"Skipped **{dup_csv}** duplicates. "
                               f"Corpus total: **{len(st.session_state.corpus)}**")

            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    # Tab 3: API
    with tab_api:
        st.subheader("🔌 API Data Source (Optional)")
        st.markdown("""
Fetch articles from a public REST API. The response must return a JSON array or an object
containing an array of articles. Each article should have a text/description field.
        """)

        api_url = st.text_input("API Endpoint URL",
                                placeholder="https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_KEY")
        api_key = st.text_input("API Key (if required)", type="password")

        st.markdown("**Field mapping** — tell the system which JSON keys to use:")
        fc1, fc2, fc3 = st.columns(3)
        f_title = fc1.text_input("Title field",   "title")
        f_body  = fc2.text_input("Body field",    "description")
        f_url   = fc3.text_input("URL field",     "url")
        f_root  = st.text_input("Root array key (leave blank if response is already an array)",
                                "articles",
                                help="e.g. for NewsAPI the JSON root key containing the list is 'articles'")
        max_api = st.slider("Max articles to fetch", 5, 100, 20)

        if st.button("🔗 Fetch from API", type="primary", key="btn_api"):
            if not api_url.strip():
                st.error("Enter an API endpoint URL.")
            else:
                try:
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    with st.spinner("Fetching from API…"):
                        resp = requests.get(api_url.strip(), headers=headers, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()

                    # Navigate to root array
                    if f_root.strip() and isinstance(data, dict):
                        data = data.get(f_root.strip(), data)
                    if isinstance(data, dict):
                        data = list(data.values())[0] if data else []
                    if not isinstance(data, list):
                        st.error("Could not find a list in the API response. Check the root array key.")
                        st.stop()

                    existing_bodies = {d["body"][:200] for d in st.session_state.corpus}
                    added_api, dup_api = [], 0
                    base = len(st.session_state.corpus)
                    meta = load_meta()

                    for i, item in enumerate(data[:max_api]):
                        if not isinstance(item, dict):
                            continue
                        body  = str(item.get(f_body,  "") or item.get("content", "") or "").strip()
                        title = str(item.get(f_title, "") or "").strip() or body[:60]
                        url   = str(item.get(f_url,   "") or f"api://item_{i}").strip()

                        if not body or body[:200] in existing_bodies:
                            dup_api += 1
                            continue
                        existing_bodies.add(body[:200])

                        doc_id = f"doc_{base + len(added_api)}"
                        added_api.append({"id": doc_id, "url": url, "title": title, "body": body[:5000]})
                        meta[doc_id] = {
                            "url": url, "title": title, "length": len(body),
                            "depth": 0, "size_bytes": len(body.encode()),
                            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "source": "api",
                        }

                    st.session_state.corpus.extend(added_api)
                    save_corpus(st.session_state.corpus)
                    save_meta(meta)

                    st.success(f"✅ Fetched **{len(added_api)}** articles from API. "
                               f"Skipped **{dup_api}** duplicates. "
                               f"Corpus total: **{len(st.session_state.corpus)}**")

                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
                except Exception as e:
                    st.error(f"Error processing API response: {e}")

    # Shared corpus table
    st.markdown("---")
    if st.session_state.corpus:
        st.subheader(f"📋 Current Corpus ({len(st.session_state.corpus)} documents)")
        meta = load_meta()
        rows = [{"ID": d["id"], "Title": d["title"][:60], "URL": d["url"],
                 "Source": meta.get(d["id"], {}).get("source", "web"),
                 "Depth": meta.get(d["id"], {}).get("depth", "-"),
                 "Size (KB)": round(meta.get(d["id"], {}).get("size_bytes", len(d["body"])) / 1024, 1),
                 "Crawled At": meta.get(d["id"], {}).get("crawled_at", "-")}
                for d in st.session_state.corpus]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button("🗑️ Clear Corpus"):
            st.session_state.corpus        = []
            st.session_state.index         = {}
            st.session_state.crawl_summary = {}
            save_corpus([])
            save_index({})
            st.rerun()

    footer()

# ---
# PAGE 3: Index Management
# ---
elif page == "Index Management":
    st.markdown("<div class='big-title'>🗂️ Index Management</div><br>", unsafe_allow_html=True)
    corpus = st.session_state.corpus

    if not corpus:
        st.warning("⚠️ Crawl some documents first.")
    else:
        if st.button("⚙️ Build / Rebuild Index", type="primary"):
            t0 = time.time()
            with st.spinner("Building inverted index & TF-IDF…"):
                try:
                    idx = build_inverted_index(corpus)
                    st.session_state.index = idx
                    save_index(idx)
                    vec, mat = build_tfidf(corpus)
                    st.session_state.tfidf_vec = vec
                    st.session_state.tfidf_mat = mat
                except ValueError as e:
                    st.error(f"❌ Index build failed: {e}. "
                             "Ensure documents contain sufficient non-stopword text.")
                    st.stop()

            t1 = time.time()
            with st.spinner("Computing PageRank & HITS…"):
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
            st.success(f"✅ Index built: **{len(idx)}** unique terms in **{build_time}s**")

        idx = st.session_state.index
        if idx:
            m1, m2, m3 = st.columns(3)
            m1.metric("Unique Terms",   len(idx))
            m2.metric("Total Postings", sum(len(v) for v in idx.values()))
            m3.metric("Index Size",     f"{index_size_kb()} KB")

            st.subheader("🔍 Term Lookup")
            term = st.text_input("Enter term", "search")
            if term:
                proc = preprocess(term)
                key  = proc[0] if proc else term
                postings = idx.get(key, {})
                if postings:
                    st.json(postings)
                else:
                    st.info(f"Term `{key}` not found in index.")

            st.subheader("📊 Top 30 Terms by Document Frequency")
            df = pd.DataFrame(
                [(t, len(p), sum(p.values())) for t, p in idx.items()],
                columns=["Term", "Doc Freq", "Total Freq"]
            ).nlargest(30, "Doc Freq")
            st.dataframe(df, use_container_width=True)

    footer()

# ---
# PAGE 4: Text Mining
# ---
elif page == "Text Mining":
    st.markdown("<div class='big-title'>⛏️ Text Mining & Preprocessing</div><br>", unsafe_allow_html=True)
    corpus = st.session_state.corpus

    if not corpus:
        st.warning("⚠️ Crawl some documents first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Preprocessing Comparison", "Keyword Extraction", "Document Classification"])

        with tab1:
            doc_idx = st.selectbox("Select Document", range(len(corpus)),
                                   format_func=lambda i: corpus[i]["title"][:60])
            raw = corpus[doc_idx]["body"][:1000]
            st.text_area("Raw Text (first 1000 chars)", raw, height=110)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Stemming**")
                st.text_area("", " ".join(preprocess(raw, "stem")[:50]), height=90, key="stem_out")
            with c2:
                st.markdown("**Lemmatization**")
                st.text_area("", " ".join(preprocess(raw, "lemmatize")[:50]), height=90, key="lem_out")

            orig    = tokenize(raw)
            no_stop = remove_stopwords(orig)
            comp_df = pd.DataFrame({
                "Stage": ["Raw Tokens", "After Stopword Removal", "After Stemming", "After Lemmatization"],
                "Count": [len(orig), len(no_stop), len(stem_tokens(no_stop)), len(lemmatize_tokens(no_stop))]
            })
            fig = px.bar(comp_df, x="Stage", y="Count",
                         color_discrete_sequence=["#2563EB"],
                         title="Token Reduction across Preprocessing Stages")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if st.session_state.tfidf_vec is None:
                st.info("ℹ️ Build the index first (Index Management page).")
            else:
                vec = st.session_state.tfidf_vec
                mat = st.session_state.tfidf_mat
                doc_idx2 = st.selectbox("Document", range(len(corpus)),
                                        format_func=lambda i: corpus[i]["title"][:60],
                                        key="kw_doc")
                if doc_idx2 >= mat.shape[0]:
                    st.warning("⚠️ This document was added after the last index build. "
                               "Go to **Index Management → Build / Rebuild Index** first.")
                    st.stop()
                row     = mat[doc_idx2].toarray().flatten()
                top_idx = row.argsort()[::-1][:20]
                words   = vec.get_feature_names_out()
                kw_df   = pd.DataFrame({"Keyword": words[top_idx], "TF-IDF": row[top_idx]})
                fig = px.bar(kw_df, x="Keyword", y="TF-IDF",
                             color_discrete_sequence=["#B279A2"],
                             title="Top-20 Keywords by TF-IDF")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Corpus-Wide Most Common Terms")
                all_tok = []
                for d in corpus:
                    all_tok.extend(preprocess(d["body"]))
                freq_df = pd.DataFrame(Counter(all_tok).most_common(30),
                                       columns=["Term", "Frequency"])
                fig2 = px.bar(freq_df, x="Term", y="Frequency",
                              color_discrete_sequence=["#F58518"])
                fig2.update_xaxes(tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            if st.session_state.tfidf_mat is None:
                st.info("ℹ️ Build the index first.")
            else:
                # Use only docs covered by the current TF-IDF matrix (stale-index safety)
                indexed_corpus = corpus[:st.session_state.tfidf_mat.shape[0]]
                labels = [auto_label(d) for d in indexed_corpus]
                lc_df  = pd.DataFrame(Counter(labels).items(), columns=["Category", "Count"])
                fig = px.pie(lc_df, names="Category", values="Count",
                             title="Auto-labeled Category Distribution")
                st.plotly_chart(fig, use_container_width=True)

                mat = st.session_state.tfidf_mat
                le  = LabelEncoder()
                y   = le.fit_transform(labels)
                clf = MultinomialNB()
                clf.fit(mat, y)

                q = st.text_input("Classify custom text", "information retrieval ranking")
                if q:
                    q_vec  = st.session_state.tfidf_vec.transform([preprocess_str(q)])
                    probs  = clf.predict_proba(q_vec)[0]
                    p_df   = pd.DataFrame({"Category": le.classes_, "Probability": probs}).sort_values("Probability", ascending=False)
                    st.dataframe(p_df)

    footer()

# ---
# PAGE 5: Search
# ---
elif page == "Search":
    st.markdown("<div class='big-title'>🔎 Search Interface</div><br>", unsafe_allow_html=True)
    corpus = st.session_state.corpus
    index  = st.session_state.index

    if not corpus:
        st.warning("⚠️ Crawl some documents first.")
    elif not index:
        st.warning("⚠️ Build the index first (Index Management).")
    else:
        # Example queries
        with st.expander("💡 Example queries"):
            st.markdown("`information retrieval` &nbsp; `machine learning algorithm` &nbsp; `web search ranking` &nbsp; `natural language processing` &nbsp; `PageRank link analysis`")

        # Edge case: warn if index is stale (corpus grew since last build)
        if (st.session_state.tfidf_mat is not None
                and st.session_state.tfidf_mat.shape[0] != len(corpus)):
            st.warning(
                f"⚠️ Index is stale: built on "
                f"**{st.session_state.tfidf_mat.shape[0]}** docs but corpus now has "
                f"**{len(corpus)}** docs. Go to **Index Management → Build / Rebuild Index** "
                f"before searching for accurate results.")

        query = st.text_input("Enter your query", "information retrieval")
        c1, c2, c3, c4 = st.columns(4)
        mode    = c1.selectbox("Search Mode", ["TF-IDF Ranked", "Boolean AND", "Boolean OR", "PageRank Combined"])
        top_k   = c2.slider("Top K", 3, 20, 10)
        alpha   = c3.slider("TF-IDF weight", 0.0, 1.0, 0.5, 0.05, help="Only for PageRank Combined")
        min_len = c4.slider("Min doc length (words)", 0, 500, 0)

        sort_by = st.radio("Sort results by", ["Score (desc)", "Title (asc)"], horizontal=True)

        if st.button("🔍 Search", type="primary") and query:
            t0 = time.time()

            if mode == "Boolean AND":
                raw = boolean_search(query, index, corpus, "AND")
                results = [(d, 1.0) for d in raw[:top_k]]
            elif mode == "Boolean OR":
                raw = boolean_search(query, index, corpus, "OR")
                results = [(d, 1.0) for d in raw[:top_k]]
            elif mode == "TF-IDF Ranked":
                if st.session_state.tfidf_vec is None:
                    st.error("Build TF-IDF index first.")
                    st.stop()
                results = tfidf_search(query, corpus, st.session_state.tfidf_vec,
                                       st.session_state.tfidf_mat, top_k)
            else:
                if st.session_state.tfidf_vec is None:
                    st.error("Build index first.")
                    st.stop()
                results = ranked_search(query, corpus, st.session_state.tfidf_vec,
                                        st.session_state.tfidf_mat,
                                        st.session_state.pagerank, top_k, alpha)

            # Filter by min length
            results = [(d, s) for d, s in results if len(d["body"].split()) >= min_len]

            # Sort
            if sort_by == "Title (asc)":
                results.sort(key=lambda x: x[0]["title"])

            elapsed_ms = (time.time() - t0) * 1000

            # Save to history
            hist = st.session_state.search_history
            hist.append((query, mode, len(results), elapsed_ms))
            st.session_state.search_history = hist[-20:]

            # Track mode usage
            s = st.session_state.stats
            s.setdefault("mode_usage", {})
            s["mode_usage"][mode] = s["mode_usage"].get(mode, 0) + 1
            s.setdefault("search_latencies_ms", [])
            s["search_latencies_ms"].append(round(elapsed_ms, 1))
            save_stats(s)
            st.session_state.stats = s

            if not results:
                st.info("🔍 No results found. Try a different query or search mode.")
            else:
                st.caption(f"**{len(results)}** results in **{elapsed_ms:.1f} ms**")
                for rank, (doc, score) in enumerate(results, 1):
                    with st.expander(f"#{rank}  {doc['title'][:70]}  (score: {score:.4f})"):
                        st.markdown(f"**URL:** [{doc['url']}]({doc['url']})")
                        st.markdown(f"**Category:** `{auto_label(doc)}` &nbsp; **Length:** {len(doc['body'].split())} words")
                        snippet = doc["body"][:400]
                        for t in query.lower().split():
                            snippet = re.sub(f"(?i)({re.escape(t)})", r"**\1**", snippet)
                        st.markdown(snippet + "…")

                scores_df = pd.DataFrame({"Title": [r[0]["title"][:40] for r in results],
                                           "Score": [r[1] for r in results]})
                fig = px.bar(scores_df, x="Title", y="Score",
                             title="Result Score Distribution",
                             color_discrete_sequence=["#2563EB"])
                fig.update_xaxes(tickangle=35)
                st.plotly_chart(fig, use_container_width=True)

        if st.session_state.search_history:
            with st.expander("📜 Search History"):
                for q, m, n, ms in st.session_state.search_history[::-1]:
                    st.markdown(f"- `{q}` — *{m}* — {n} results — {ms:.0f} ms")

    footer()

# ---
# PAGE 6: Ranking Visualization
# ---
elif page == "Ranking Visualization":
    st.markdown("<div class='big-title'>📈 Ranking Visualization</div><br>", unsafe_allow_html=True)
    corpus = st.session_state.corpus
    pr     = st.session_state.pagerank
    hubs   = st.session_state.hits_hubs
    auths  = st.session_state.hits_auth
    G      = st.session_state.link_graph

    if not corpus or not pr:
        st.warning("⚠️ Build the index first (Index Management).")
    else:
        tab1, tab2, tab3 = st.tabs(["PageRank", "HITS", "Side-by-Side Comparison"])

        with tab1:
            st.subheader("PageRank Scores")
            pr_df = pd.DataFrame([
                {"Document": d["title"][:50], "PageRank": pr.get(d["id"], 0)}
                for d in corpus
            ]).sort_values("PageRank", ascending=False)
            fig = px.bar(pr_df, x="Document", y="PageRank",
                         color_discrete_sequence=["#E45756"],
                         title="PageRank Score per Document")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pr_df, use_container_width=True)

            if G is not None and G.number_of_edges() > 0:
                st.subheader("Link Graph (similarity-based)")
                node_ids = [d["id"] for d in corpus][:15]
                SG  = G.subgraph(node_ids)
                pos = nx.spring_layout(SG, seed=42)
                ex, ey = [], []
                for u, v in SG.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    ex += [x0, x1, None]; ey += [y0, y1, None]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                          line=dict(width=0.8, color="#aaa")))
                fig2.add_trace(go.Scatter(
                    x=[pos[n][0] for n in SG.nodes()],
                    y=[pos[n][1] for n in SG.nodes()],
                    mode="markers+text", text=list(SG.nodes()),
                    textposition="top center",
                    marker=dict(size=12, color="#2563EB")))
                fig2.update_layout(showlegend=False, height=420,
                                   title="Similarity-Based Link Graph (top 15 nodes)")
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("HITS – Hubs and Authorities")
            hits_df = pd.DataFrame([
                {"Document": d["title"][:50],
                 "Hub Score": hubs.get(d["id"], 0),
                 "Authority Score": auths.get(d["id"], 0)}
                for d in corpus
            ]).sort_values("Authority Score", ascending=False)
            fig = px.scatter(hits_df, x="Hub Score", y="Authority Score",
                             text="Document", title="HITS: Hub vs Authority Score",
                             color_discrete_sequence=["#72B7B2"])
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hits_df, use_container_width=True)

        with tab3:
            st.subheader("Side-by-Side: TF-IDF vs PageRank vs Combined")
            if st.session_state.tfidf_vec is None:
                st.info("Build the index first to compute TF-IDF scores.")
            else:
                query_r = st.text_input("Query for ranking comparison", "information retrieval")
                if query_r:
                    vec = st.session_state.tfidf_vec
                    mat = st.session_state.tfidf_mat
                    # Align to mat rows to prevent shape mismatch when index is stale
                    indexed_docs  = corpus[:mat.shape[0]]
                    tfidf_scores  = cosine_similarity(vec.transform([preprocess_str(query_r)]), mat).flatten()
                    pr_scores_raw = np.array([pr.get(d["id"], 0) for d in indexed_docs])
                    pr_norm       = pr_scores_raw / pr_scores_raw.max() if pr_scores_raw.max() > 0 else pr_scores_raw
                    combined      = 0.5 * tfidf_scores + 0.5 * pr_norm

                    comp_df = pd.DataFrame({
                        "Document":  [d["title"][:40] for d in indexed_docs],
                        "TF-IDF":    tfidf_scores,
                        "PageRank":  pr_norm,
                        "Combined":  combined,
                    }).sort_values("Combined", ascending=False).head(15)

                    fig = px.bar(comp_df, x="Document", y=["TF-IDF", "PageRank", "Combined"],
                                 barmode="group", title="Ranking Score Comparison",
                                 color_discrete_sequence=["#2563EB", "#E45756", "#10B981"])
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(comp_df.round(4), use_container_width=True)

                    st.info("""
**Why combined ranking improves results:**
TF-IDF captures query-term relevance but ignores document authority.
PageRank measures structural importance based on link topology.
Combining both ensures that highly relevant documents which are also well-connected
score higher — reducing the risk of surfacing obscure but keyword-dense pages at the top.
                    """)

    footer()

# ---
# PAGE 7: Recommendations
# ---
elif page == "Recommendations":
    st.markdown("<div class='big-title'>💡 Recommendation Panel</div><br>", unsafe_allow_html=True)
    corpus = st.session_state.corpus
    mat    = st.session_state.tfidf_mat

    if not corpus:
        st.warning("⚠️ Crawl some documents first.")
    elif mat is None:
        st.warning("⚠️ Build the index first (Index Management).")
    else:
        tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])

        with tab1:
            st.subheader("Content-Based Recommendations")
            doc_idx = st.selectbox("Reference document", range(len(corpus)),
                                   format_func=lambda i: corpus[i]["title"][:60])
            top_k   = st.slider("Top K", 3, 10, 5, key="cb_k")
            recs    = content_based_recommend(doc_idx, corpus, mat, top_k)

            if recs:
                for rank, (doc, sim) in enumerate(recs, 1):
                    rec_card(rank, doc, sim, auto_label(doc))

                sim_df = pd.DataFrame({"Title": [r[0]["title"][:40] for r in recs],
                                       "Similarity": [r[1] for r in recs]})
                fig = px.bar(sim_df, x="Title", y="Similarity",
                             color_discrete_sequence=["#10B981"],
                             title="Cosine Similarity of Recommendations")
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No similar documents found for this document.")

        with tab2:
            st.subheader("Collaborative Filtering")
            st.markdown("Rate some documents below, then request recommendations.")
            ratings = st.session_state.ratings
            user_id = st.text_input("Your User ID", "user_1")

            with st.expander("⭐ Rate Documents"):
                for doc in corpus[:10]:
                    r = st.slider(doc["title"][:55], 0, 5, 0, key=f"rate_{doc['id']}")
                    if r > 0:
                        ratings.setdefault(user_id, {})[doc["id"]] = r
                if st.button("💾 Save Ratings"):
                    st.session_state.ratings = ratings
                    save_ratings(ratings)
                    st.success("Ratings saved.")

            if st.button("🤝 Get Collaborative Recommendations"):
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
                    st.info("Rate at least one document first to get personalised recommendations.")

        with tab3:
            st.subheader("Hybrid Recommendation (CB + CF blend)")
            doc_idx_h = st.selectbox("Reference doc", range(len(corpus)),
                                     format_func=lambda i: corpus[i]["title"][:60],
                                     key="hyb_doc")
            user_id_h = st.text_input("User ID for CF", "user_1", key="hyb_user")
            cb_w = st.slider("Content-Based weight", 0.0, 1.0, 0.6)

            if st.button("🔀 Get Hybrid Recommendations"):
                cb_recs = content_based_recommend(doc_idx_h, corpus, mat, top_k=10)
                cf_recs = collab_recommend(user_id_h, st.session_state.ratings,
                                           corpus, mat, top_k=10)
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
                    fig = px.bar(h_df, x="Title", y="Hybrid Score",
                                 color_discrete_sequence=["#B279A2"],
                                 title="Hybrid Recommendation Scores")
                    fig.update_xaxes(tickangle=30)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for hybrid recommendations.")

    footer()

# ---
# PAGE 8: Evaluation
# ---
elif page == "Evaluation":
    st.markdown("<div class='big-title'>📐 Evaluation Dashboard</div><br>", unsafe_allow_html=True)
    corpus = st.session_state.corpus
    index  = st.session_state.index

    if not corpus:
        st.warning("⚠️ Crawl some documents first.")
    elif not index:
        st.warning("⚠️ Build the index first.")
    else:
        st.markdown("Define a **query** and mark **relevant documents** (ground truth), then compare all retrieval methods.")

        query = st.text_input("Evaluation Query", "information retrieval ranking")
        top_k = st.slider("K", 3, min(20, len(corpus)), min(10, len(corpus)))

        st.subheader("✅ Mark Relevant Documents (Ground Truth)")
        relevant_ids = [doc["id"] for doc in corpus
                        if st.checkbox(doc["title"][:70], key=f"rel_{doc['id']}")]

        if st.button("▶️ Run Evaluation", type="primary"):
            if not relevant_ids:
                st.warning("Mark at least one relevant document.")
            elif st.session_state.tfidf_vec is None:
                st.warning("Build the index first.")
            else:
                methods = {
                    "Boolean AND":      [d["id"] for d in boolean_search(query, index, corpus, "AND")[:top_k]],
                    "Boolean OR":       [d["id"] for d in boolean_search(query, index, corpus, "OR")[:top_k]],
                    "TF-IDF":           [d["id"] for d, _ in tfidf_search(query, corpus,
                                         st.session_state.tfidf_vec, st.session_state.tfidf_mat, top_k)],
                    "TF-IDF+PageRank":  [d["id"] for d, _ in ranked_search(query, corpus,
                                         st.session_state.tfidf_vec, st.session_state.tfidf_mat,
                                         st.session_state.pagerank, top_k)],
                }

                rows = []
                for method, retrieved in methods.items():
                    m = compute_metrics(retrieved, relevant_ids, k=top_k)
                    m["Method"] = method
                    rows.append(m)
                st.session_state.eval_results = rows

                metric_cols = ["Precision", "Recall", "F1", "Precision@K",
                               "Recall@K", "AP", "MRR", "NDCG"]
                eval_df = pd.DataFrame(rows).set_index("Method")

                st.subheader("📊 Comparison Table")
                st.dataframe(eval_df[metric_cols].round(4), use_container_width=True)

                # Best method per metric
                st.subheader("🏆 Best Method per Metric")
                best_rows = []
                for col in metric_cols:
                    best_m = eval_df[col].idxmax()
                    best_rows.append({"Metric": col, "Best Method": best_m,
                                      "Score": round(eval_df.loc[best_m, col], 4)})
                st.dataframe(pd.DataFrame(best_rows), use_container_width=True)

                st.subheader("MAP / MRR / NDCG Comparison")
                fig = px.bar(eval_df.reset_index(), x="Method", y=["AP", "MRR", "NDCG"],
                             barmode="group", title="MAP / MRR / NDCG",
                             color_discrete_sequence=["#2563EB", "#E45756", "#10B981"])
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Precision–Recall Curve (TF-IDF)")
                pr_pts = [(compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)["Precision"],
                           compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)["Recall"])
                          for k_ in range(1, top_k + 1)]
                pr_fig = px.line(x=[r for _, r in pr_pts], y=[p for p, _ in pr_pts],
                                 labels={"x": "Recall", "y": "Precision"},
                                 title="Precision–Recall Curve (TF-IDF)", markers=True,
                                 color_discrete_sequence=["#2563EB"])
                st.plotly_chart(pr_fig, use_container_width=True)

                st.subheader("NDCG@K Curve")
                ndcg_fig = px.line(
                    x=list(range(1, top_k + 1)),
                    y=[compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)["NDCG"]
                       for k_ in range(1, top_k + 1)],
                    labels={"x": "K", "y": "NDCG"}, title="NDCG@K Curve",
                    markers=True, color_discrete_sequence=["#F58518"])
                st.plotly_chart(ndcg_fig, use_container_width=True)

                best_overall = eval_df["NDCG"].idxmax()
                st.info(f"""
**Interpretation:** Across all metrics, **{best_overall}** achieves the highest NDCG score,
indicating it returns the most relevant documents in the best rank order.
Boolean methods may achieve higher recall but often rank relevant documents poorly.
TF-IDF combined with PageRank typically improves ranking precision by weighting
both query relevance and document authority.
                """)

    footer()

# ---
# PAGE 9: Performance Analytics
# ---
elif page == "Performance Analytics":
    st.markdown("<div class='big-title'>⚡ Performance Analytics</div><br>", unsafe_allow_html=True)

    corpus = st.session_state.corpus
    index  = st.session_state.index
    stats  = st.session_state.stats
    cs     = st.session_state.crawl_summary
    meta   = load_meta()

    # KPI row
    st.markdown("### 🗂️ Corpus & Index Metrics")
    avg_len = int(np.mean([len(d["body"].split()) for d in corpus])) if corpus else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📄 Documents",       len(corpus))
    k2.metric("📚 Vocabulary Size",  len(index))
    k3.metric("💾 Index Size (KB)",  index_size_kb())
    k4.metric("📖 Avg Doc Length",   avg_len)

    st.markdown("### ⏱️ Timing Metrics")
    t1, t2, t3 = st.columns(3)
    t1.metric("🕷️ Last Crawl Duration",  f"{cs.get('crawl_duration_s', '–')} s")
    t2.metric("⚙️ Last Index Build",     f"{stats.get('index_build_time_s', '–')} s")
    latencies = stats.get("search_latencies_ms", [])
    avg_lat   = round(np.mean(latencies), 1) if latencies else 0
    t3.metric("🔍 Avg Search Latency",   f"{avg_lat} ms")

    st.markdown("### 🌐 Crawl Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌐 URLs Visited",      cs.get("total_visited", "–"))
    c2.metric("✅ Successful Crawls",  cs.get("successful", "–"))
    c3.metric("❌ Failed Requests",    cs.get("failed", "–"))
    c4.metric("🚫 Duplicates Skipped", (cs.get("dup_urls", 0) + cs.get("dup_docs", 0)))

    if corpus:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Document Length Distribution")
            lengths = [len(d["body"].split()) for d in corpus]
            fig1 = px.histogram(x=lengths, nbins=15, labels={"x": "Words"},
                                title="Doc Length Distribution",
                                color_discrete_sequence=["#2563EB"])
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            st.subheader("Crawl Depth Distribution")
            depths = [meta.get(d["id"], {}).get("depth", 0) for d in corpus]
            depth_df = pd.DataFrame(Counter(depths).items(), columns=["Depth", "Count"]).sort_values("Depth")
            fig2 = px.bar(depth_df, x="Depth", y="Count",
                          title="Documents per Crawl Depth",
                          color_discrete_sequence=["#10B981"])
            st.plotly_chart(fig2, use_container_width=True)

        if latencies:
            st.subheader("Search Latency Over Queries")
            fig3 = px.line(x=list(range(1, len(latencies) + 1)), y=latencies,
                           labels={"x": "Query #", "y": "Latency (ms)"},
                           title="Search Latency per Query",
                           color_discrete_sequence=["#F58518"])
            fig3.add_hline(y=avg_lat, line_dash="dash", line_color="red",
                           annotation_text=f"Avg: {avg_lat} ms")
            st.plotly_chart(fig3, use_container_width=True)

        mode_usage = stats.get("mode_usage", {})
        if mode_usage:
            st.subheader("Search Mode Usage")
            mu_df = pd.DataFrame(mode_usage.items(), columns=["Mode", "Count"])
            fig4  = px.pie(mu_df, names="Mode", values="Count",
                           title="Search Mode Usage Distribution")
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("No data yet. Crawl documents and run some searches first.")

    footer()
