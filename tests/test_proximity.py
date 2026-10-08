from src.index import InvertedIndex
from src.ranking import BM25Ranker


def test_term_proximity():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["sun", "is", "very", "hot"]
    )

    ranker = BM25Ranker(index)

    distance = ranker.term_proximity(
        "1.txt",
        "sun",
        "hot"
    )

    assert distance == 3


def test_proximity_score():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["sun", "hot"]
    )

    ranker = BM25Ranker(index)

    score = ranker.proximity_score(
        "1.txt",
        "sun",
        "hot"
    )

    assert score > 0
