from src.index import InvertedIndex


def test_add_document():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["sun", "is", "hot", "sun"]
    )

    postings = index.search("sun")

    assert "1.txt" in postings
    assert postings["1.txt"]["frequency"] == 2
    assert postings["1.txt"]["positions"] == [0, 3]


def test_document_length():
    index = InvertedIndex()

    index.add_document(
        "1.txt",
        ["sun", "is", "hot"]
    )

    assert index.document_lengths["1.txt"] == 3


def test_average_document_length():
    index = InvertedIndex()

    index.add_document("1.txt", ["a", "b"])
    index.add_document("2.txt", ["a", "b", "c", "d"])

    assert index.average_document_length() == 3
