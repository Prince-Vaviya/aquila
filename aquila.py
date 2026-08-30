from beir import util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
import numpy as np

# Load
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

data_path = util.download_and_unzip(url, "datasets")

corpus, queries, qrels = GenericDataLoader(
    data_path
).load(split="test")

# Build corpus
doc_ids = list(corpus.keys())

doc_texts = [
    corpus[d]["title"] + " " + corpus[d]["text"]
    for d in doc_ids
]

tokenized_corpus = [
    text.split()
    for text in doc_texts
]

# Build BM25
bm25 = BM25Okapi(tokenized_corpus)

# Search one query
query_id = next(iter(queries))
query = queries[query_id]

scores = bm25.get_scores(query.split())

# Top 10
top_indices = np.argsort(scores)[::-1][:10]

results = [
    (doc_ids[i], scores[i])
    for i in top_indices
]

print("QUERY:")
print(query)

print("\nTOP RESULTS:")
for rank, (doc_id, score) in enumerate(results, 1):
    print(rank, doc_id, score)

(doc_id, score) = results[0]
print(corpus[doc_id])