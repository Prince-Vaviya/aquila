from src.index import InvertedIndex
from src.ranking import BM25Ranker


def test_query_proximity():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["solar", "energy", "is", "important"]
    )

    ranker = BM25Ranker(index)

    score = ranker.query_proximity(
        "1.txt",
        "solar energy"
    )

    assert score > 0


def test_query_proximity_missing_term():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["solar", "energy"]
    )

    ranker = BM25Ranker(index)

    score = ranker.query_proximity(
        "1.txt",
        "solar moon"
    )

    assert score == 0
