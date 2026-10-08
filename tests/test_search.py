from src.index import InvertedIndex
from src.ranking import BM25Ranker
from src.search import retrieve_candidates, search_bm25


def create_index():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["sun", "is", "hot"]
    )

    index.add_document(
        "2.txt",
        ["moon", "is", "cold"]
    )

    index.add_document(
        "3.txt",
        ["sun", "provides", "energy"]
    )

    return index


def test_retrieve_candidates():
    index = create_index()

    candidates = retrieve_candidates("sun", index)

    assert "1.txt" in candidates
    assert "3.txt" in candidates
    assert "2.txt" not in candidates


def test_search_bm25():
    index = create_index()
    ranker = BM25Ranker(index)

    results = search_bm25(
        "sun",
        index,
        ranker,
        top_k=5
    )

    assert len(results) == 2
    assert results[0][1] >= results[1][1]
