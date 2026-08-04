"""
IR Assignment 2 – End-to-End Information Retrieval System
Domain: News Articles
Run: streamlit run app.py
"""

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
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── NLTK downloads (silent) ────────────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
            "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ── Persistence paths ──────────────────────────────────────────────────────────
DATA_DIR = "ir_data"
os.makedirs(DATA_DIR, exist_ok=True)
CORPUS_FILE   = os.path.join(DATA_DIR, "corpus.json")
INDEX_FILE    = os.path.join(DATA_DIR, "index.pkl")
META_FILE     = os.path.join(DATA_DIR, "metadata.json")
RATINGS_FILE  = os.path.join(DATA_DIR, "ratings.json")

# ── Helpers ────────────────────────────────────────────────────────────────────
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

# ── Text preprocessing ─────────────────────────────────────────────────────────
STOP = set(stopwords.words("english"))
stemmer = PorterStemmer()
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
    if method == "stem":
        return stem_tokens(tokens)
    return lemmatize_tokens(tokens)

def preprocess_str(text, method="lemmatize"):
    return " ".join(preprocess(text, method))

# ── Crawling ───────────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0 (IR-Assignment-Bot/1.0)"}

def crawl(seeds, max_depth=1, max_pages=20):
    visited_urls = set()
    seen_hashes  = set()
    docs = []
    meta = {}
    queue = [(url.strip(), 0) for url in seeds if url.strip()]

    progress = st.progress(0)
    status   = st.empty()

    while queue and len(docs) < max_pages:
        url, depth = queue.pop(0)
        if url in visited_urls:
            continue
        visited_urls.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")

            # Extract text
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title else url
            body  = soup.get_text(separator=" ", strip=True)
            body  = re.sub(r"\s+", " ", body)[:5000]

            # Dedup by content hash
            h = hash(body[:500])
            if h in seen_hashes or len(body) < 100:
                continue
            seen_hashes.add(h)

            doc_id = f"doc_{len(docs)}"
            docs.append({"id": doc_id, "url": url, "title": title, "body": body})
            meta[doc_id] = {
                "url": url, "title": title, "length": len(body),
                "depth": depth, "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            status.text(f"Crawled ({len(docs)}/{max_pages}): {title[:60]}")
            progress.progress(len(docs) / max_pages)

            # Enqueue child links
            if depth < max_depth:
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    p = urlparse(link)
                    if p.scheme in ("http", "https") and link not in visited_urls:
                        queue.append((link, depth + 1))

        except Exception:
            continue

    progress.empty()
    status.empty()
    return docs, meta

# ── Indexing ───────────────────────────────────────────────────────────────────
def build_inverted_index(docs):
    index = defaultdict(lambda: defaultdict(int))
    for doc in docs:
        tokens = preprocess(doc["title"] + " " + doc["body"])
        for tok in tokens:
            index[tok][doc["id"]] += 1
    return {k: dict(v) for k, v in index.items()}

def build_tfidf(docs):
    corpus = [preprocess_str(d["title"] + " " + d["body"]) for d in docs]
    vec = TfidfVectorizer(max_features=5000)
    mat = vec.fit_transform(corpus)
    return vec, mat

# ── PageRank ───────────────────────────────────────────────────────────────────
def build_pagerank(docs):
    G = nx.DiGraph()
    for d in docs:
        G.add_node(d["id"])
    # Simulated links: docs that share keywords link to each other
    vec, mat = build_tfidf(docs)
    sim = cosine_similarity(mat)
    for i in range(len(docs)):
        for j in range(len(docs)):
            if i != j and sim[i, j] > 0.1:
                G.add_edge(docs[i]["id"], docs[j]["id"], weight=float(sim[i, j]))
    pr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {d["id"]: 1 / len(docs) for d in docs}
    return pr, G

# ── HITS ──────────────────────────────────────────────────────────────────────
def compute_hits(G):
    if G.number_of_edges() == 0:
        return {}, {}
    hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
    return hubs, authorities

# ── Search ─────────────────────────────────────────────────────────────────────
def boolean_search(query, index, docs, op="AND"):
    tokens = preprocess(query)
    if not tokens:
        return []
    sets = [set(index.get(t, {}).keys()) for t in tokens]
    if op == "AND":
        result = sets[0].intersection(*sets[1:]) if sets else set()
    else:
        result = sets[0].union(*sets[1:]) if sets else set()
    return [d for d in docs if d["id"] in result]

def tfidf_search(query, docs, vec, mat, top_k=10):
    if not docs:
        return []
    q_vec = vec.transform([preprocess_str(query)])
    scores = cosine_similarity(q_vec, mat).flatten()
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], float(scores[i])) for i in ranked if scores[i] > 0]

def ranked_search(query, docs, vec, mat, pr, top_k=10, alpha=0.5):
    if not docs:
        return []
    q_vec = vec.transform([preprocess_str(query)])
    tfidf_scores = cosine_similarity(q_vec, mat).flatten()
    pr_scores = np.array([pr.get(d["id"], 0) for d in docs])
    if pr_scores.max() > 0:
        pr_scores = pr_scores / pr_scores.max()
    combined = alpha * tfidf_scores + (1 - alpha) * pr_scores
    ranked = np.argsort(combined)[::-1][:top_k]
    return [(docs[i], float(combined[i])) for i in ranked if combined[i] > 0]

# ── Recommendations ────────────────────────────────────────────────────────────
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
    # Build user-item matrix from ratings dict {user: {doc_id: score}}
    all_users = list(ratings.keys())
    all_docs  = list({d["id"] for d in docs})
    uid_map = {u: i for i, u in enumerate(all_users)}
    did_map = {d: i for i, d in enumerate(all_docs)}

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

    # Weighted predicted scores
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

    hits = [1 if r in relevant else 0 for r in retrieved]
    prec = sum(hits) / len(retrieved) if retrieved else 0
    rec  = sum(hits) / len(relevant)  if relevant  else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    prec_k = sum(hits) / k if k > 0 else 0
    rec_k  = rec

    # AP
    ap, rel_so_far = 0.0, 0
    for i, h in enumerate(hits):
        if h:
            rel_so_far += 1
            ap += rel_so_far / (i + 1)
    ap = ap / len(relevant) if relevant else 0

    # MRR
    mrr = 0.0
    for i, h in enumerate(hits):
        if h:
            mrr = 1 / (i + 1)
            break

    # NDCG
    dcg  = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg > 0 else 0

    return {"Precision": prec, "Recall": rec, "F1": f1,
            "Precision@K": prec_k, "Recall@K": rec_k,
            "AP": ap, "MRR": mrr, "NDCG": ndcg}

# ── Streamlit App ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IR System – News", layout="wide", page_icon="🔍")

# Session-state init
for key, default in [
    ("corpus", load_corpus()),
    ("index",  load_index()),
    ("tfidf_vec", None),
    ("tfidf_mat", None),
    ("pagerank",  {}),
    ("hits_hubs", {}),
    ("hits_auth", {}),
    ("link_graph", None),
    ("ratings", load_ratings()),
    ("eval_results", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.sidebar.title("🔍 IR System")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Crawling",
    "Index Management",
    "Text Mining",
    "Search",
    "Ranking Visualization",
    "Recommendations",
    "Evaluation",
])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.title("📊 System Dashboard")
    corpus = st.session_state.corpus
    index  = st.session_state.index

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(corpus))
    c2.metric("Index Terms", len(index))
    total_tokens = sum(sum(v.values()) for v in index.values())
    c3.metric("Total Token Occurrences", total_tokens)
    avg_len = np.mean([len(d["body"].split()) for d in corpus]) if corpus else 0
    c4.metric("Avg Doc Length (words)", int(avg_len))

    if corpus:
        st.subheader("Document Length Distribution")
        lengths = [len(d["body"].split()) for d in corpus]
        fig = px.histogram(x=lengths, nbins=20, labels={"x": "Word Count"},
                           title="Document Length Distribution",
                           color_discrete_sequence=["#4C78A8"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top-20 Index Terms by Document Frequency")
        df_terms = pd.DataFrame(
            [(t, len(postings)) for t, postings in index.items()],
            columns=["Term", "Doc Frequency"]
        ).nlargest(20, "Doc Frequency")
        fig2 = px.bar(df_terms, x="Term", y="Doc Frequency",
                      color_discrete_sequence=["#E45756"])
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Crawled Documents")
        df = pd.DataFrame([{"Title": d["title"][:70], "URL": d["url"],
                             "Length": len(d["body"].split())} for d in corpus])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No documents yet. Go to **Crawling** to fetch some pages.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – Crawling
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Crawling":
    st.title("🕷️ Web Crawling")

    with st.expander("Default seed URLs (pre-loaded for demo)", expanded=False):
        st.markdown("""
- https://en.wikipedia.org/wiki/Information_retrieval
- https://en.wikipedia.org/wiki/Natural_language_processing
- https://en.wikipedia.org/wiki/Machine_learning
- https://en.wikipedia.org/wiki/Search_engine
- https://en.wikipedia.org/wiki/PageRank
        """)

    seeds_raw = st.text_area(
        "Seed URLs (one per line)",
        value="\n".join([
            "https://en.wikipedia.org/wiki/Information_retrieval",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Search_engine",
            "https://en.wikipedia.org/wiki/PageRank",
        ]),
        height=140,
    )
    col1, col2 = st.columns(2)
    max_depth = col1.slider("Crawl Depth", 0, 2, 0)
    max_pages = col2.slider("Max Pages", 5, 50, 10)

    if st.button("🚀 Start Crawling", type="primary"):
        seeds = [s.strip() for s in seeds_raw.strip().split("\n") if s.strip()]
        with st.spinner("Crawling…"):
            new_docs, new_meta = crawl(seeds, max_depth=max_depth, max_pages=max_pages)

        # Merge with existing (dedup by URL)
        existing_urls = {d["url"] for d in st.session_state.corpus}
        added = [d for d in new_docs if d["url"] not in existing_urls]

        # Re-ID
        base = len(st.session_state.corpus)
        for i, d in enumerate(added):
            d["id"] = f"doc_{base + i}"

        st.session_state.corpus.extend(added)
        save_corpus(st.session_state.corpus)

        meta = load_meta()
        meta.update({d["id"]: new_meta.get(d.get("id", ""), {}) for d in added})
        save_meta(meta)

        st.success(f"Added {len(added)} new documents. Total: {len(st.session_state.corpus)}")

    if st.session_state.corpus:
        st.subheader("Crawled Documents")
        meta = load_meta()
        rows = []
        for d in st.session_state.corpus:
            m = meta.get(d["id"], {})
            rows.append({"ID": d["id"], "Title": d["title"][:60],
                         "URL": d["url"], "Depth": m.get("depth", "-"),
                         "Length": m.get("length", len(d["body"]))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button("🗑️ Clear Corpus"):
            st.session_state.corpus = []
            st.session_state.index  = {}
            save_corpus([])
            save_index({})
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – Index Management
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Index Management":
    st.title("🗂️ Index Management")
    corpus = st.session_state.corpus

    if not corpus:
        st.warning("Crawl some documents first.")
    else:
        col1, col2 = st.columns(2)
        if col1.button("Build / Rebuild Index", type="primary"):
            with st.spinner("Building inverted index…"):
                idx = build_inverted_index(corpus)
                st.session_state.index = idx
                save_index(idx)

                vec, mat = build_tfidf(corpus)
                st.session_state.tfidf_vec = vec
                st.session_state.tfidf_mat = mat

                pr, G = build_pagerank(corpus)
                hubs, auths = compute_hits(G)
                st.session_state.pagerank   = pr
                st.session_state.hits_hubs  = hubs
                st.session_state.hits_auth  = auths
                st.session_state.link_graph = G
            st.success(f"Index built: {len(idx)} unique terms.")

        index = st.session_state.index
        if index:
            st.metric("Unique Terms", len(index))
            st.metric("Total Postings", sum(len(v) for v in index.values()))

            st.subheader("Lookup a Term")
            term = st.text_input("Term", "search")
            if term:
                proc = preprocess(term)
                key  = proc[0] if proc else term
                postings = index.get(key, {})
                if postings:
                    st.json(postings)
                else:
                    st.info(f"Term '{key}' not in index.")

            st.subheader("Top Terms by Posting-list Length")
            df = pd.DataFrame(
                [(t, len(p), sum(p.values())) for t, p in index.items()],
                columns=["Term", "Doc Freq", "Total Freq"]
            ).nlargest(30, "Doc Freq")
            st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – Text Mining
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Text Mining":
    st.title("⛏️ Text Mining & Preprocessing")
    corpus = st.session_state.corpus
    if not corpus:
        st.warning("Crawl some documents first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Preprocessing Comparison", "Keyword Extraction", "Document Classification"])

        with tab1:
            doc_idx = st.selectbox("Select Document", range(len(corpus)),
                                   format_func=lambda i: corpus[i]["title"][:60])
            doc = corpus[doc_idx]
            raw = doc["body"][:1000]
            st.text_area("Raw Text (first 1000 chars)", raw, height=120)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Stemming**")
                stemmed = " ".join(preprocess(raw, "stem")[:50])
                st.text_area("", stemmed, height=100, key="stem_out")
            with col2:
                st.markdown("**Lemmatization**")
                lemmed = " ".join(preprocess(raw, "lemmatize")[:50])
                st.text_area("", lemmed, height=100, key="lem_out")

            # Token count comparison
            orig_tokens  = tokenize(raw)
            no_stop      = remove_stopwords(orig_tokens)
            stem_tokens_ = stem_tokens(no_stop)
            lem_tokens_  = lemmatize_tokens(no_stop)

            comp = pd.DataFrame({
                "Stage": ["Raw", "No Stopwords", "Stemmed", "Lemmatized"],
                "Token Count": [len(orig_tokens), len(no_stop), len(stem_tokens_), len(lem_tokens_)]
            })
            fig = px.bar(comp, x="Stage", y="Token Count",
                         color_discrete_sequence=["#54A24B"],
                         title="Token Reduction across Preprocessing Stages")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Top Keywords per Document (TF-IDF)")
            if st.session_state.tfidf_vec is None:
                st.info("Build the index first (Index Management page).")
            else:
                vec = st.session_state.tfidf_vec
                mat = st.session_state.tfidf_mat
                doc_idx2 = st.selectbox("Document", range(len(corpus)),
                                        format_func=lambda i: corpus[i]["title"][:60],
                                        key="kw_doc")
                row = mat[doc_idx2].toarray().flatten()
                top_idx = row.argsort()[::-1][:20]
                words = vec.get_feature_names_out()
                kw_df = pd.DataFrame({"Keyword": words[top_idx], "TF-IDF Score": row[top_idx]})
                fig = px.bar(kw_df, x="Keyword", y="TF-IDF Score",
                             color_discrete_sequence=["#B279A2"],
                             title="Top-20 Keywords by TF-IDF")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Corpus-Wide Term Frequency")
                all_tokens = []
                for d in corpus:
                    all_tokens.extend(preprocess(d["body"]))
                freq = Counter(all_tokens).most_common(30)
                freq_df = pd.DataFrame(freq, columns=["Term", "Frequency"])
                fig2 = px.bar(freq_df, x="Term", y="Frequency",
                              color_discrete_sequence=["#F58518"])
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("Document Classification (Naive Bayes, auto-labeled)")
            if st.session_state.tfidf_mat is None:
                st.info("Build the index first.")
            else:
                # Auto-assign categories based on title keywords
                categories = {
                    "ML/AI": ["machine", "learning", "neural", "deep", "model", "algorithm"],
                    "IR/Search": ["retrieval", "search", "index", "query", "ranking"],
                    "NLP": ["language", "text", "nlp", "processing", "corpus"],
                    "Web": ["web", "internet", "crawl", "page", "link", "network"],
                }

                def auto_label(doc):
                    t = (doc["title"] + " " + doc["body"][:200]).lower()
                    scores = {cat: sum(1 for kw in kws if kw in t)
                              for cat, kws in categories.items()}
                    best = max(scores, key=scores.get)
                    return best if scores[best] > 0 else "General"

                labels = [auto_label(d) for d in corpus]
                label_counts = Counter(labels)
                lc_df = pd.DataFrame(label_counts.items(), columns=["Category", "Count"])
                fig = px.pie(lc_df, names="Category", values="Count",
                             title="Auto-labeled Category Distribution")
                st.plotly_chart(fig, use_container_width=True)

                # Train NB and show class probabilities for a query
                mat = st.session_state.tfidf_mat
                le  = LabelEncoder()
                y   = le.fit_transform(labels)
                clf = MultinomialNB()
                # NB needs non-negative; TF-IDF is always >= 0
                clf.fit(mat, y)

                q = st.text_input("Classify a custom text", "information retrieval ranking")
                if q:
                    q_vec = st.session_state.tfidf_vec.transform([preprocess_str(q)])
                    probs = clf.predict_proba(q_vec)[0]
                    prob_df = pd.DataFrame({"Category": le.classes_, "Probability": probs}).sort_values("Probability", ascending=False)
                    st.dataframe(prob_df)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – Search
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Search":
    st.title("🔎 Web Search Interface")
    corpus = st.session_state.corpus
    index  = st.session_state.index

    if not corpus or not index:
        st.warning("Crawl documents and build the index first.")
    else:
        query = st.text_input("Enter your query", "information retrieval")
        col1, col2, col3 = st.columns(3)
        mode  = col1.selectbox("Search Mode", ["TF-IDF Ranked", "Boolean AND", "Boolean OR", "PageRank Combined"])
        top_k = col2.slider("Top K results", 3, 20, 10)
        alpha = col3.slider("PageRank weight (for combined)", 0.0, 1.0, 0.5, 0.05)

        if st.button("Search", type="primary") and query:
            start = time.time()
            if mode == "Boolean AND":
                results = boolean_search(query, index, corpus, "AND")
                results = [(d, 1.0) for d in results[:top_k]]
            elif mode == "Boolean OR":
                results = boolean_search(query, index, corpus, "OR")
                results = [(d, 1.0) for d in results[:top_k]]
            elif mode == "TF-IDF Ranked":
                if st.session_state.tfidf_vec is None:
                    st.error("Build TF-IDF index first.")
                    st.stop()
                results = tfidf_search(query, corpus, st.session_state.tfidf_vec,
                                       st.session_state.tfidf_mat, top_k)
            else:  # PageRank Combined
                if st.session_state.tfidf_vec is None:
                    st.error("Build index first.")
                    st.stop()
                results = ranked_search(query, corpus, st.session_state.tfidf_vec,
                                        st.session_state.tfidf_mat,
                                        st.session_state.pagerank, top_k, alpha)
            elapsed = time.time() - start

            st.caption(f"{len(results)} results in {elapsed*1000:.1f} ms")
            for rank, (doc, score) in enumerate(results, 1):
                with st.expander(f"#{rank} — {doc['title'][:70]}  (score: {score:.4f})"):
                    st.markdown(f"**URL:** [{doc['url']}]({doc['url']})")
                    # Highlight query terms in snippet
                    snippet = doc["body"][:400]
                    for t in query.lower().split():
                        snippet = re.sub(f"(?i)({re.escape(t)})", r"**\1**", snippet)
                    st.markdown(snippet + "…")

            if results:
                scores_df = pd.DataFrame(
                    {"Title": [r[0]["title"][:40] for r in results],
                     "Score": [r[1] for r in results]}
                )
                fig = px.bar(scores_df, x="Title", y="Score",
                             title="Result Score Distribution",
                             color_discrete_sequence=["#4C78A8"])
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 – Ranking Visualization
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Ranking Visualization":
    st.title("📈 Ranking Visualization")
    corpus = st.session_state.corpus
    pr     = st.session_state.pagerank
    hubs   = st.session_state.hits_hubs
    auths  = st.session_state.hits_auth
    G      = st.session_state.link_graph

    if not corpus or not pr:
        st.warning("Build the index first (Index Management).")
    else:
        tab1, tab2 = st.tabs(["PageRank", "HITS"])

        with tab1:
            st.subheader("PageRank Scores")
            pr_df = pd.DataFrame([
                {"Document": d["title"][:50], "PageRank": pr.get(d["id"], 0)}
                for d in corpus
            ]).sort_values("PageRank", ascending=False)
            fig = px.bar(pr_df, x="Document", y="PageRank",
                         color_discrete_sequence=["#E45756"],
                         title="PageRank per Document")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pr_df)

            st.subheader("Link Graph (similarity-based)")
            if G is not None and G.number_of_edges() > 0:
                # Show top 15 nodes for clarity
                top_nodes = pr_df.head(15)["Document"].tolist()
                node_ids  = [d["id"] for d in corpus if d["title"][:50] in top_nodes]
                SG = G.subgraph(node_ids[:15])
                pos = nx.spring_layout(SG, seed=42)
                edge_x, edge_y = [], []
                for u, v in SG.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                node_x = [pos[n][0] for n in SG.nodes()]
                node_y = [pos[n][1] for n in SG.nodes()]
                node_text = [n for n in SG.nodes()]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                          line=dict(width=0.8, color="#aaa")))
                fig2.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                          text=node_text, textposition="top center",
                                          marker=dict(size=12, color="#4C78A8")))
                fig2.update_layout(showlegend=False, height=400,
                                   title="Link Graph (top 15 nodes)")
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
                             text="Document", title="HITS: Hub vs Authority",
                             color_discrete_sequence=["#72B7B2"])
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hits_df)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 – Recommendations
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Recommendations":
    st.title("💡 Recommendation Panel")
    corpus = st.session_state.corpus
    mat    = st.session_state.tfidf_mat

    if not corpus or mat is None:
        st.warning("Build the index first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])

        with tab1:
            st.subheader("Content-Based Recommendations")
            doc_idx = st.selectbox("Reference document", range(len(corpus)),
                                   format_func=lambda i: corpus[i]["title"][:60])
            top_k = st.slider("Top K", 3, 10, 5, key="cb_k")
            recs = content_based_recommend(doc_idx, corpus, mat, top_k)
            if recs:
                for rank, (doc, sim) in enumerate(recs, 1):
                    st.markdown(f"**#{rank}** [{doc['title'][:70]}]({doc['url']}) — Cosine Similarity: `{sim:.4f}`")
                sim_df = pd.DataFrame({"Title": [r[0]["title"][:40] for r in recs],
                                       "Similarity": [r[1] for r in recs]})
                fig = px.bar(sim_df, x="Title", y="Similarity",
                             color_discrete_sequence=["#54A24B"])
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No similar documents found.")

        with tab2:
            st.subheader("Collaborative Filtering")
            st.markdown("Rate some documents to enable collaborative recommendations.")
            ratings = st.session_state.ratings
            user_id = st.text_input("User ID", "user_1")

            with st.expander("Rate Documents"):
                for doc in corpus[:10]:
                    r = st.slider(doc["title"][:50], 0, 5, 0, key=f"rate_{doc['id']}")
                    if r > 0:
                        if user_id not in ratings:
                            ratings[user_id] = {}
                        ratings[user_id][doc["id"]] = r
                if st.button("Save Ratings"):
                    st.session_state.ratings = ratings
                    save_ratings(ratings)
                    st.success("Ratings saved.")

            if st.button("Get Collaborative Recommendations"):
                # Add some synthetic users for meaningful CF
                doc_ids = [d["id"] for d in corpus]
                if "synthetic_user_1" not in ratings:
                    ratings["synthetic_user_1"] = {
                        doc_ids[i]: np.random.randint(1, 6)
                        for i in range(min(len(doc_ids), 8))
                    }
                    ratings["synthetic_user_2"] = {
                        doc_ids[i]: np.random.randint(1, 6)
                        for i in range(min(len(doc_ids), 8))
                    }
                    st.session_state.ratings = ratings

                recs = collab_recommend(user_id, ratings, corpus, mat, top_k=5)
                if recs:
                    st.subheader("Recommended for you:")
                    for rank, (doc, score) in enumerate(recs, 1):
                        st.markdown(f"**#{rank}** {doc['title'][:70]} — Score: `{score:.4f}`")
                else:
                    st.info("Rate at least one document first, or add more users.")

        with tab3:
            st.subheader("Hybrid Recommendation (CB + CF blend)")
            doc_idx_h = st.selectbox("Reference doc", range(len(corpus)),
                                     format_func=lambda i: corpus[i]["title"][:60],
                                     key="hyb_doc")
            user_id_h = st.text_input("User ID for CF component", "user_1", key="hyb_user")
            cb_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.6)
            cf_weight = 1.0 - cb_weight

            if st.button("Get Hybrid Recommendations"):
                cb_recs = content_based_recommend(doc_idx_h, corpus, mat, top_k=10)
                cf_recs = collab_recommend(user_id_h, st.session_state.ratings,
                                           corpus, mat, top_k=10)

                cb_map = {d["id"]: s for d, s in cb_recs}
                cf_map = {d["id"]: s for d, s in cf_recs}
                all_ids = set(cb_map) | set(cf_map)
                doc_map = {d["id"]: d for d in corpus}

                hybrid = []
                for did in all_ids:
                    score = cb_weight * cb_map.get(did, 0) + cf_weight * cf_map.get(did, 0)
                    if did in doc_map:
                        hybrid.append((doc_map[did], score))
                hybrid.sort(key=lambda x: x[1], reverse=True)

                st.subheader("Top Hybrid Recommendations")
                for rank, (doc, score) in enumerate(hybrid[:5], 1):
                    st.markdown(f"**#{rank}** {doc['title'][:70]} — Hybrid Score: `{score:.4f}`")
                if hybrid:
                    h_df = pd.DataFrame({"Title": [h[0]["title"][:40] for h in hybrid[:8]],
                                         "Hybrid Score": [h[1] for h in hybrid[:8]]})
                    fig = px.bar(h_df, x="Title", y="Hybrid Score",
                                 color_discrete_sequence=["#B279A2"])
                    fig.update_xaxes(tickangle=30)
                    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Evaluation":
    st.title("📐 Evaluation Dashboard")
    corpus = st.session_state.corpus
    index  = st.session_state.index

    if not corpus or not index:
        st.warning("Crawl and index documents first.")
    else:
        st.markdown("""
        Define a **query** and mark which documents are **relevant** (ground truth),
        then run different retrieval methods and compare their metrics.
        """)

        query = st.text_input("Evaluation Query", "information retrieval ranking")
        top_k = st.slider("K", 3, min(20, len(corpus)), min(10, len(corpus)))

        st.subheader("Mark Relevant Documents (ground truth)")
        relevant_ids = []
        for doc in corpus:
            if st.checkbox(doc["title"][:70], key=f"rel_{doc['id']}"):
                relevant_ids.append(doc["id"])

        if st.button("Run Evaluation", type="primary"):
            if not relevant_ids:
                st.warning("Mark at least one relevant document.")
            elif st.session_state.tfidf_vec is None:
                st.warning("Build the index first.")
            else:
                methods = {}

                # Boolean AND
                bool_and = boolean_search(query, index, corpus, "AND")
                methods["Boolean AND"] = [d["id"] for d in bool_and[:top_k]]

                # Boolean OR
                bool_or = boolean_search(query, index, corpus, "OR")
                methods["Boolean OR"] = [d["id"] for d in bool_or[:top_k]]

                # TF-IDF
                tfidf_res = tfidf_search(query, corpus, st.session_state.tfidf_vec,
                                         st.session_state.tfidf_mat, top_k)
                methods["TF-IDF"] = [d["id"] for d, _ in tfidf_res]

                # PageRank Combined
                pr_res = ranked_search(query, corpus, st.session_state.tfidf_vec,
                                       st.session_state.tfidf_mat,
                                       st.session_state.pagerank, top_k)
                methods["TF-IDF + PageRank"] = [d["id"] for d, _ in pr_res]

                rows = []
                for method, retrieved in methods.items():
                    m = compute_metrics(retrieved, relevant_ids, k=top_k)
                    m["Method"] = method
                    rows.append(m)

                eval_df = pd.DataFrame(rows).set_index("Method")
                st.session_state.eval_results = rows

                metric_cols = ["Precision", "Recall", "F1", "Precision@K",
                               "Recall@K", "AP", "MRR", "NDCG"]
                st.subheader("Comparison Table")
                st.dataframe(eval_df[metric_cols].round(4), use_container_width=True)

                st.subheader("MAP & MRR Comparison")
                fig = px.bar(eval_df.reset_index(), x="Method",
                             y=["AP", "MRR", "NDCG"],
                             barmode="group", title="MAP / MRR / NDCG",
                             color_discrete_sequence=["#4C78A8", "#E45756", "#54A24B"])
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Precision–Recall Curve (TF-IDF)")
                prec_points, rec_points = [], []
                for k_ in range(1, top_k + 1):
                    m_ = compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)
                    prec_points.append(m_["Precision"])
                    rec_points.append(m_["Recall"])
                pr_fig = px.line(x=rec_points, y=prec_points,
                                 labels={"x": "Recall", "y": "Precision"},
                                 title="Precision–Recall Curve (TF-IDF)",
                                 markers=True)
                st.plotly_chart(pr_fig, use_container_width=True)

                st.subheader("NDCG@K Curve")
                ndcg_vals = []
                for k_ in range(1, top_k + 1):
                    m_ = compute_metrics(methods["TF-IDF"][:k_], relevant_ids, k=k_)
                    ndcg_vals.append(m_["NDCG"])
                ndcg_fig = px.line(x=list(range(1, top_k + 1)), y=ndcg_vals,
                                   labels={"x": "K", "y": "NDCG"},
                                   title="NDCG@K Curve", markers=True,
                                   color_discrete_sequence=["#F58518"])
                st.plotly_chart(ndcg_fig, use_container_width=True)
