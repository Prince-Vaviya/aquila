# ============================================================
# 🦅 AQUILA — COMPLETE CURRENT IMPLEMENTATION
# ============================================================
#
# 1. Load SciFact
# 2. BM25 retrieval
# 3. BM25 evaluation
# 4. Dense retrieval
# 5. Dense evaluation
# 6. RRF Hybrid retrieval
# 7. Hybrid evaluation
# 8. Rule-based AQUILA router
# 9. AQUILA evaluation
#
# NOT IMPLEMENTED YET:
# - Prepare Dataset
# - AQUILA ML Router
# - Final AQUILA ML evaluation
# ============================================================


# ============================================================
# 0. INSTALLATION
# ============================================================
#
# pip install -r requirements.txt
#
# ============================================================
# 1. IMPORTS
# ============================================================

import time
import numpy as np
import pandas as pd

from collections import Counter, defaultdict

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

import pytrec_eval

import spacy


# ============================================================
# 2. LOAD SCIFACT
# ============================================================

url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

data_path = util.download_and_unzip(url, "datasets")

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


print("=" * 60)
print("📚 SCIFACT")
print("=" * 60)

print("Documents :", len(corpus))
print("Queries   :", len(queries))
print("Qrels     :", len(qrels))


# ============================================================
# 3. PREPARE CORPUS
# ============================================================

doc_ids = list(corpus.keys())

doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in doc_ids]


# ============================================================
# 4. BM25 INDEX
# ============================================================

tokenized_corpus = [text.split() for text in doc_texts]

bm25 = BM25Okapi(tokenized_corpus)


# ============================================================
# 5. BM25 SEARCH
# ============================================================

def bm25_search(query, top_k=10):

    scores = bm25.get_scores(query.split())

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(doc_ids[i], scores[i]) for i in top_indices]


# ============================================================
# 6. BM25 EVALUATION
# ============================================================

bm25_results = {}
bm25_latencies = []

for query_id, query in queries.items():

    start = time.perf_counter()

    results = bm25_search(query, top_k=10)

    end = time.perf_counter()

    bm25_latencies.append(end - start)

    bm25_results[query_id] = results


# Convert to TREC format

bm25_run = {}

for query_id, results in bm25_results.items():

    bm25_run[query_id] = {
        doc_id: 1.0 / (rank + 1)
        for rank, (doc_id, score)
        in enumerate(results)
    }


# Evaluate

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {
        "ndcg_cut.10",
        "recall.10"
    }
)

bm25_evaluation = evaluator.evaluate(
    bm25_run
)


mean_bm25_ndcg = np.mean([
    result["ndcg_cut_10"]
    for result in bm25_evaluation.values()
])

mean_bm25_recall = np.mean([
    result["recall_10"]
    for result in bm25_evaluation.values()
])

avg_bm25_latency = np.mean(
    bm25_latencies
)


print()
print("=" * 60)
print("🔎 BM25 EVALUATION")
print("=" * 60)

print(
    f"nDCG@10     : {mean_bm25_ndcg:.4f}"
)

print(
    f"Recall@10   : {mean_bm25_recall:.4f}"
)

print(
    f"Avg Latency : {avg_bm25_latency * 1000:.2f} ms"
)


# ============================================================
# 7. DENSE MODEL
# ============================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 8. DOCUMENT EMBEDDINGS
# ============================================================

doc_embeddings = model.encode(
    doc_texts,
    show_progress_bar=True,
    batch_size=64
)


# ============================================================
# 9. DENSE SEARCH
# ============================================================

def dense_search(query, top_k=10):

    query_embedding = model.encode(
        [query]
    )

    similarities = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    return [
        (doc_ids[i], similarities[i])
        for i in top_indices
    ]


# ============================================================
# 10. DENSE EVALUATION
# ============================================================

dense_results = {}
dense_latencies = []

for query_id, query in queries.items():

    start = time.perf_counter()

    results = dense_search(
        query,
        top_k=10
    )

    end = time.perf_counter()

    dense_latencies.append(
        end - start
    )

    dense_results[query_id] = results


# Convert to TREC format

dense_run = {}

for query_id, results in dense_results.items():

    dense_run[query_id] = {
        doc_id: 1.0 / (rank + 1)
        for rank, (doc_id, score)
        in enumerate(results)
    }


# Evaluate

dense_evaluation = evaluator.evaluate(
    dense_run
)


mean_dense_ndcg = np.mean([
    result["ndcg_cut_10"]
    for result in dense_evaluation.values()
])

mean_dense_recall = np.mean([
    result["recall_10"]
    for result in dense_evaluation.values()
])

avg_dense_latency = np.mean(
    dense_latencies
)


print()
print("=" * 60)
print("🧠 DENSE EVALUATION")
print("=" * 60)

print(
    f"nDCG@10     : {mean_dense_ndcg:.4f}"
)

print(
    f"Recall@10   : {mean_dense_recall:.4f}"
)

print(
    f"Avg Latency : {avg_dense_latency * 1000:.2f} ms"
)


# ============================================================
# 11. RRF HYBRID
# ============================================================

def rrf_fuse(
    bm25_results,
    dense_results,
    k=60,
    top_k=10
):

    scores = {}

    # BM25 contribution

    for rank, (doc_id, _) in enumerate(
        bm25_results
    ):

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank + 1)
        )


    # Dense contribution

    for rank, (doc_id, _) in enumerate(
        dense_results
    ):

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank + 1)
        )


    # Sort

    ranked = sorted(
        scores.items(),
        key=lambda x: -x[1]
    )[:top_k]

    return ranked


# ============================================================
# 12. HYBRID EVALUATION
# ============================================================

hybrid_results = {}
hybrid_latencies = []

for query_id, query in queries.items():

    start = time.perf_counter()

    bm25_result = bm25_search(
        query,
        top_k=10
    )

    dense_result = dense_search(
        query,
        top_k=10
    )

    results = rrf_fuse(
        bm25_result,
        dense_result,
        k=60,
        top_k=10
    )

    end = time.perf_counter()

    hybrid_latencies.append(
        end - start
    )

    hybrid_results[query_id] = results


# TREC format

hybrid_run = {}

for query_id, results in hybrid_results.items():

    hybrid_run[query_id] = {
        doc_id: 1.0 / (rank + 1)
        for rank, (doc_id, score)
        in enumerate(results)
    }


# Evaluate

hybrid_evaluation = evaluator.evaluate(
    hybrid_run
)


mean_hybrid_ndcg = np.mean([
    result["ndcg_cut_10"]
    for result in hybrid_evaluation.values()
])

mean_hybrid_recall = np.mean([
    result["recall_10"]
    for result in hybrid_evaluation.values()
])

avg_hybrid_latency = np.mean(
    hybrid_latencies
)


print()
print("=" * 60)
print("🔀 HYBRID / RRF EVALUATION")
print("=" * 60)

print(
    f"nDCG@10     : {mean_hybrid_ndcg:.4f}"
)

print(
    f"Recall@10   : {mean_hybrid_recall:.4f}"
)

print(
    f"Avg Latency : {avg_hybrid_latency * 1000:.2f} ms"
)


# ============================================================
# 13. LOAD SPACY
# ============================================================

try:

    nlp = spacy.load(
        "en_core_web_sm"
    )

except:

    import subprocess

    subprocess.run(
        [
            "python",
            "-m",
            "spacy",
            "download",
            "en_core_web_sm"
        ],
        check=True
    )

    nlp = spacy.load(
        "en_core_web_sm"
    )


# ============================================================
# 14. QUERY FEATURE EXTRACTION
# ============================================================

def extract_query_features(query):

    doc = nlp(query)

    return {

        "word_count":
            len(query.split()),

        "has_entities":
            len(doc.ents) > 0,

        "has_numbers":
            any(
                token.like_num
                for token in doc
            ),

        "has_quotes":
            '"' in query
    }


# ============================================================
# 15. INITIAL RULE-BASED AQUILA ROUTER
# ============================================================

def route_query(query):

    features = extract_query_features(
        query
    )

    word_count = features[
        "word_count"
    ]

    has_entities = features[
        "has_entities"
    ]

    has_numbers = features[
        "has_numbers"
    ]

    has_quotes = features[
        "has_quotes"
    ]


    # Lexical

    if (
        has_entities
        or has_numbers
        or has_quotes
        or word_count <= 3
    ):

        return "lexical"


    # Semantic

    elif word_count >= 8:

        return "semantic"


    # Hybrid

    else:

        return "hybrid"


# ============================================================
# 16. AQUILA SEARCH
# ============================================================

def aquila_search(
    query,
    strategy,
    top_k=10
):

    if strategy == "lexical":

        return bm25_search(
            query,
            top_k
        )


    elif strategy == "semantic":

        return dense_search(
            query,
            top_k
        )


    elif strategy == "hybrid":

        bm25_result = bm25_search(
            query,
            top_k
        )

        dense_result = dense_search(
            query,
            top_k
        )

        return rrf_fuse(
            bm25_result,
            dense_result,
            k=60,
            top_k=top_k
        )


# ============================================================
# 17. AQUILA EVALUATION
# ============================================================

aquila_results = {}
aquila_run = {}

aquila_routes = {}
aquila_latencies = []


for query_id, query in queries.items():

    start = time.perf_counter()

    # Route exactly once

    strategy = route_query(
        query
    )

    # Retrieve

    results = aquila_search(
        query,
        strategy,
        top_k=10
    )

    end = time.perf_counter()

    aquila_latencies.append(
        end - start
    )

    aquila_routes[
        query_id
    ] = strategy

    aquila_results[
        query_id
    ] = results


# TREC format

for query_id, results in aquila_results.items():

    aquila_run[query_id] = {

        doc_id:
            1.0 / (rank + 1)

        for rank, (doc_id, score)
        in enumerate(results)
    }


# Evaluate

aquila_evaluation = evaluator.evaluate(
    aquila_run
)


mean_aquila_ndcg = np.mean([
    result["ndcg_cut_10"]
    for result in aquila_evaluation.values()
])

mean_aquila_recall = np.mean([
    result["recall_10"]
    for result in aquila_evaluation.values()
])

avg_aquila_latency = np.mean(
    aquila_latencies
)

median_aquila_latency = np.median(
    aquila_latencies
)


# Routing distribution

route_counts = Counter(
    aquila_routes.values()
)


print()
print("=" * 60)
print("🦅 AQUILA — RULE-BASED EVALUATION")
print("=" * 60)

print(
    f"nDCG@10        : {mean_aquila_ndcg:.4f}"
)

print(
    f"Recall@10      : {mean_aquila_recall:.4f}"
)

print(
    f"Avg Latency    : "
    f"{avg_aquila_latency * 1000:.2f} ms"
)

print(
    f"Median Latency : "
    f"{median_aquila_latency * 1000:.2f} ms"
)

print()

for strategy in [
    "lexical",
    "semantic",
    "hybrid"
]:

    count = route_counts.get(
        strategy,
        0
    )

    percentage = (
        count / len(queries)
    ) * 100

    print(
        f"{strategy:8s}: "
        f"{count:3d} "
        f"({percentage:.2f}%)"
    )