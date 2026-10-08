from src.index import InvertedIndex
from src.ranking import BM25Ranker


def create_index():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["sun", "is", "hot"]
    )

    index.add_document(
        "2.txt",
        ["sun", "sun", "hot"]
    )

    index.add_document(
        "3.txt",
        ["moon", "is", "cold"]
    )

    return index


def test_idf():
    index = create_index()
    ranker = BM25Ranker(index)

    score = ranker.idf("sun")

    assert score > 0


def test_bm25_score():
    index = create_index()
    ranker = BM25Ranker(index)

    score = ranker.score("1.txt", "sun")

    assert score > 0


def test_ranking():
    index = create_index()
    ranker = BM25Ranker(index)

    results = ranker.rank(
        "sun",
        {"1.txt", "2.txt"}
    )

    assert len(results) == 2
    assert results[0][1] >= results[1][1]
