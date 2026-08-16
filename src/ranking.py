import math
from src.tokenizer import tokenize

class BM25Ranker:
    def __init__(self, index, k1=1.2, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term):
        df = len(self.index.search(term))
        N = len(self.index.document_lengths)

        if df == 0:
            return 0

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, document_id, query):
        query_terms = set(tokenize(query))
        score = 0

        dl = self.index.document_lengths[document_id]
        avgdl = self.index.average_document_length()

        for term in query_terms:
            postings = self.index.search(term)

            if document_id not in postings:
                continue

            tf = postings[document_id]

            length_normalization = ((1 - self.b) + self.b * (dl / avgdl))

            tf_component = (tf * (self.k1+ 1)) / (tf + self.k1 * length_normalization)
            score += self.idf(term) * tf_component

        return score

    
    def rank(self, query, candidates):
        results = []

        for document_id in candidates:
            score = self.score(document_id, query)
            results.append((document_id, score))

        results.sort(key=lambda x : x[1], reverse=True)

        return results