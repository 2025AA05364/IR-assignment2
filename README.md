# IR Assignment 2 - Information Retrieval System
**Group 52** | BITS Pilani MTech | Semester 2 2025-26

A complete end-to-end Information Retrieval system built with Python and Streamlit. Covers web crawling, text preprocessing, inverted index, TF-IDF, PageRank, HITS, recommendations, and evaluation metrics.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### 3. Basic workflow
1. **Crawling** → click *Start Crawling* (Wikipedia seeds are pre-filled)
2. **Index Management** → click *Build / Rebuild Index*
3. Use Search, Ranking, Recommendations, Evaluation pages

> **No internet?** Go to **Crawling → Upload CSV Dataset** and upload `sample_dataset.csv` (included)

---

## Pages and Features

### Dashboard
- Corpus statistics: document count, vocabulary size, total tokens, average length
- Document length distribution histogram
- Top-20 index terms bar chart
- Recent search history
- Full crawled documents table

### Crawling
Three data source tabs:
- **Web Crawling** — configurable seed URLs (default: 5 Wikipedia IR articles), crawl depth (0–2), max pages (5–50). Handles duplicate URLs and duplicate documents automatically. Stores metadata separately from document content.
- **Upload CSV** — supports any CSV with text/title/url columns, auto-detects column names, skips NaN/empty rows
- **API** — fetch from any REST JSON endpoint with configurable field mapping

### Index Management
- Builds inverted index (posting lists with term frequencies)
- Builds TF-IDF matrix (max 5000 features)
- Computes PageRank scores (α=0.85, similarity-based graph)
- Computes HITS hub and authority scores
- Term lookup: enter any term to see its posting list
- Top-30 terms by document frequency table

### Text Mining
- **Preprocessing Comparison** — stemming (Porter) vs lemmatization (WordNet), token reduction chart across stages
- **Keyword Extraction** — top-20 TF-IDF keywords per document, corpus-wide term frequency chart
- **Document Classification** — auto-labels documents into ML/AI, IR/Search, NLP, Web categories; Naive Bayes classifier; classify any custom text

### Search
Four retrieval modes:
- **Boolean AND** — returns documents containing all query terms
- **Boolean OR** — returns documents containing any query term
- **TF-IDF Ranked** — cosine similarity between query and document TF-IDF vectors
- **PageRank Combined** — weighted blend of TF-IDF score and PageRank score (α slider)

Additional controls: Top K slider, minimum document length filter, sort by score or title. Query term highlighting in snippets. Search history tracking.

### Ranking Visualization
- **PageRank** — bar chart of all documents by PageRank score + similarity-based link graph
- **HITS** — hub vs authority scatter plot + sortable table
- **Side-by-Side Comparison** — grouped bar chart comparing TF-IDF, PageRank, and Combined scores for any query

### Recommendations
- **Content-Based** — cosine similarity between TF-IDF document vectors; Top-K with similarity scores
- **Collaborative Filtering** — user-user CF using rating matrix; rate documents and get personalised recommendations
- **Hybrid** — weighted blend of content-based and collaborative scores (weight slider)

### Evaluation
- Enter a query and manually mark relevant documents (ground truth)
- Runs all 4 retrieval methods automatically
- Reports: Precision, Recall, F1, Precision@K, Recall@K, MAP (AP), MRR, NDCG
- Comparison table, best method per metric table
- Precision-Recall curve, NDCG@K curve, MAP/MRR/NDCG grouped bar chart

### Performance Analytics
- Corpus and index metrics (document count, vocabulary, index size, avg length)
- Timing metrics (crawl duration, index build time, average search latency)
- Crawl statistics (URLs visited, successful, failed, duplicates skipped)
- Document length distribution, crawl depth distribution
- Search latency trend chart, search mode usage pie chart

---

## Project Structure

```
app.py                    - main Streamlit application (~1400 lines)
requirements.txt          - Python dependencies
README.md                 - this file
sample_dataset.csv        - 16 IR/ML articles for offline use
Group52_Report.docx       - full project report with analysis and inferences
Group52_Contribution.xlsx - member contribution percentages
ir_data/                  - auto-created at runtime
    corpus.json           - crawled documents
    index.pkl             - inverted index
    metadata.json         - document metadata (separate from content)
    ratings.json          - user ratings for collaborative filtering
    stats.json            - performance statistics
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| scikit-learn | TF-IDF, cosine similarity, Naive Bayes, LabelEncoder |
| networkx | PageRank, HITS, graph construction |
| nltk | Tokenization, stopwords, stemming, lemmatization |
| plotly | Interactive charts |
| beautifulsoup4 + lxml | HTML parsing for web crawling |
| requests | HTTP requests for crawling and API |
| numpy, pandas | Numerical and data operations |

---

## Algorithms Implemented

- **Inverted Index** — posting lists with term frequency counts
- **TF-IDF** — term frequency × inverse document frequency with cosine similarity ranking
- **Boolean Retrieval** — set intersection (AND) and union (OR) on posting lists
- **PageRank** — iterative link analysis on cosine-similarity graph (damping α=0.85)
- **HITS** — hub and authority score computation via power iteration
- **Content-Based Filtering** — cosine similarity between document TF-IDF vectors
- **Collaborative Filtering** — user-user similarity with rating matrix
- **Naive Bayes** — multinomial NB for document category classification
- **Evaluation** — Precision, Recall, F1, P@K, R@K, AP/MAP, MRR, DCG/NDCG