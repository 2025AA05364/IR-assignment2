# IR Assignment 2 – End-to-End Information Retrieval System

## Domain
News / Encyclopedia articles (Wikipedia + configurable seed URLs)

## Features
| Section | What it covers |
|---------|---------------|
| Dashboard | Corpus stats, term frequency charts, document table |
| Crawling | Configurable seed URLs, crawl depth, max pages, duplicate dedup |
| Index Management | Inverted index build, TF-IDF matrix, PageRank & HITS computation |
| Text Mining | Stemming vs lemmatization comparison, TF-IDF keyword extraction, Naive Bayes classification |
| Search | Boolean AND/OR, TF-IDF ranked, TF-IDF + PageRank combined |
| Ranking Visualization | PageRank bar chart, link graph, HITS hub-vs-authority scatter |
| Recommendations | Content-based (cosine), Collaborative (user-user CF), Hybrid |
| Evaluation | Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG with charts |

## Install dependencies

```bash
pip install streamlit scikit-learn scipy networkx plotly requests \
            beautifulsoup4 nltk numpy pandas lxml
```

Download NLTK data (first run only):
```python
import nltk
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    nltk.download(pkg)
```

## Run the app

```bash
streamlit run app.py
```

## Workflow

1. **Crawling** – Enter seed URLs, set depth (0 = seed pages only), click *Start Crawling*
2. **Index Management** – Click *Build / Rebuild Index* to build inverted index, TF-IDF, PageRank, HITS
3. **Text Mining** – Explore preprocessing, keywords, and classification
4. **Search** – Query the corpus with different retrieval modes
5. **Ranking Visualization** – Compare PageRank and HITS scores
6. **Recommendations** – Content-based, collaborative, and hybrid recommendations
7. **Evaluation** – Mark relevant docs for a query and compare all retrieval methods

## Files
```
app.py          - Main Streamlit application
README.md       - This file
ir_data/        - Auto-created at runtime (corpus, index, metadata, ratings)
```
