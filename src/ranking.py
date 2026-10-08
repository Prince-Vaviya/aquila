import math
from src.tokenizer import tokenize
from collections import defaultdict

class BM25Ranker:
    def __init__(self, index, k1=1.2, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term):
        df = len(self.index.search(term))
        N = len(self.index.document_lengths)

        if df == 0 or N == 0:
            return 0.0

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, document_id, query):
        query_terms = set(tokenize(query))

        document_length = (self.index.document_lengths[document_id])

        average_length = (self.index.average_document_length())

        score = 0.0

        for term in query_terms:
            postings = self.index.search(term)

            if document_id not in postings:
                continue

            tf = postings[document_id]["frequency"]

            length_normalization = ((1 - self.b) + self.b * (document_length / average_length))

            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_normalization)
            score += (self.idf(term) * tf_component)

        return score

    
    def rank(self, query, candidates):
        results = []

        for document_id in candidates:
            score = self.score(document_id, query)
            results.append((document_id, score))

        results.sort(key=lambda x : x[1], reverse=True)

        return results

    def term_proximity(self, document_id, term1, term2):

        postings1 = self.index.search(term1)
        postings2 = self.index.search(term2)

        if document_id not in postings1 or document_id not in postings2:
            return float("inf")

        positions1 = postings1[document_id]["positions"]
        positions2 = postings2[document_id]["positions"]

        min_distance = float("inf")

        for p1 in positions1:
            for p2 in positions2:
                distance = abs(p1 - p2)

                if distance < min_distance:
                    min_distance = distance

        return min_distance

    def proximity_score(self, document_id, term1, term2):

        distance = self.term_proximity(
            document_id,
            term1,
            term2
        )

        if distance == float("inf"):
            return 0.0

        return 1 / (1 + distance)

    def combined_score(self, document_id, query, proximity_weight=0.2):

        bm25_score = self.score(document_id, query)

        terms = list(dict.fromkeys(tokenize(query())))

        if len(terms) < 2:
            return bm25_score

        proximity = self.proximity_score(
            document_id,
            terms[0],
            terms[1]
        )

        return bm25_score + proximity_weight * proximity

    def query_proximity(self, document_id, query):

        terms = list(dict.fromkeys(tokenize(query)))

        if len(terms) < 2:
            return 0.0

        positions = []

        for term in terms:

            postings = self.index.search(term)

            if document_id not in postings:
                return 0.0

            for position in postings[document_id]["positions"]:
                positions.append((position, term))

        positions.sort()

        required_terms = set(terms)
        window_counts = defaultdict(int)

        left = 0
        best_window = float("inf")

        for right in range(len(positions)):

            right_term = positions[right][1]

            window_counts[right_term] += 1
            while all(window_counts[term] > 0 for term in required_terms):

                window_size = (
                    positions[right][0]
                    - positions[left][0]
                    + 1
                )

                best_window = min(
                    best_window,
                    window_size
                )

                left_term = positions[left][1]

                window_counts[left_term] -= 1

                left += 1

        if best_window == float("inf"):
            return 0.0

        return 1 / best_window