from src.tokenizer import tokenize

def retrieve_candidates(query, index):
    query_terms = tokenize(query)
    candidates = set()

    for term in query_terms:
        postings = index.search(term)

        candidates.update(postings.keys())

    return candidates

def search_bm25(query, index, ranker, top_k=5):
    candidates = retrieve_candidates(query, index)

    ranked_results = ranker.rank(query, candidates)

    return ranked_results[:top_k]