from src.tokenizer import tokenize

def retrieve_candidates(query, index):
    query_terms = tokenize(query)
    candidates = set()

    for term in query_terms:
        postings = index.search(term)

        candidates.update(postings.keys())

    return candidates