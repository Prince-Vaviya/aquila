from collections import defaultdict, Counter

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)

    def add_documents(self, document_id, tokens):
        term_frequency = Counter(tokens)

        for token, frequency in term_frequency.items():
            self.index[token][document_id] = frequency

    def search(self, token):
        return self.index.get(token, {})