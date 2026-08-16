from collections import defaultdict, Counter

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)
        self.document_lengths = {}

    def add_documents(self, document_id, tokens):
        term_frequency = Counter(tokens)
        self.document_lengths[document_id] = len(tokens)

        for token, frequency in term_frequency.items():
            self.index[token][document_id] = frequency

    def search(self, token):
        return self.index.get(token, {})

    def document_length(self, document_id):
        return self.document_lengths[document_id]

    def average_document_length(self):
        if not self.document_lengths:
            return 0

        return sum(self.document_lengths.values())/ len(self.document_lengths)