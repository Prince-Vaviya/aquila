from collections import defaultdict
from src.tokenizer import tokenize


class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(dict)
        self.document_lengths = {}

    def add_document(self, document_id, tokens):

        self.document_lengths[document_id] = len(tokens)

        term_data = defaultdict(lambda: {
            "frequency": 0,
            "positions": []
        })

        for position, token in enumerate(tokens):

            term_data[token]["frequency"] += 1
            term_data[token]["positions"].append(position)

        for token, data in term_data.items():

            self.index[token][document_id] = data

    def search(self, token):
        return self.index.get(token, {})

    def average_document_length(self):

        if not self.document_lengths:
            return 0

        return (
            sum(self.document_lengths.values())
            / len(self.document_lengths)
        )


def build_index(documents):

    index = InvertedIndex()

    for document_id, text in documents.items():

        tokens = tokenize(text)

        index.add_document(
            document_id,
            tokens
        )

    return index