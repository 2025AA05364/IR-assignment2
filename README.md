# IR Assignment 2 - Group 52

Information Retrieval System built using Python and Streamlit.

## How to run

Install dependencies:
```
pip install -r requirements.txt
```

Start the app:
```
streamlit run app.py
```

## Pages

- **Dashboard** - overview of corpus stats and term frequencies
- **Crawling** - crawl Wikipedia/web pages, upload CSV, or fetch from API
- **Index Management** - build inverted index, TF-IDF, PageRank and HITS
- **Text Mining** - preprocessing comparison, keyword extraction, document classification
- **Search** - Boolean AND/OR, TF-IDF ranked, combined TF-IDF+PageRank search
- **Ranking Visualization** - PageRank scores, HITS hubs/authorities, side-by-side comparison
- **Recommendations** - content-based, collaborative filtering, hybrid
- **Evaluation** - Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG
- **Performance Analytics** - crawl stats, index size, search latencies

## Workflow

1. Go to **Crawling** and crawl some pages (default seeds are Wikipedia IR articles)
2. Go to **Index Management** and click *Build / Rebuild Index*
3. Use **Search**, **Ranking Visualization**, **Recommendations**, **Evaluation** etc.

## Files

```
app.py               - main application
requirements.txt     - python packages needed
sample_dataset.csv   - offline dataset (16 IR articles) for CSV upload if no internet
ir_data/             - created automatically at runtime (corpus, index, metadata)