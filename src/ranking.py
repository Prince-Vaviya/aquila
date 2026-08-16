import math
from src.tokenizer import tokenize

class TFIDFRanker:
    def __init__(self, index, total_documents):
        self.index = index
        self.total_documents = total_documents

    def idf(self, term):
        df = len(self.index.search(term))

        if df == 0:
            return 0

        return math.log(self.total_documents / df)

    def score(self, document_id, query):
        query_terms = set(tokenize(query))
        total_score = 0

        for term in query_terms:
            postings = self.index.search(term)

            if document_id not in postings:
                continue

            tf = postings[document_id]
            idf = self.idf(term)
            total_score += tf * idf

        return total_score

    
    def rank(self, query, candidates):
        scores = []

        for document_id in candidates:
            score = self.score(document_id, query)

            scores.append((document_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores