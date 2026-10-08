from src.dense import DenseRetriever


def test_dense_retrieval():
    documents = {
        "1.txt": (
            "The Sun is the star at the center "
            "of the Solar System."
        ),
        "2.txt": (
            "The Moon is Earth's natural satellite."
        ),
        "3.txt": (
            "The Sun provides energy and light "
            "that supports life on Earth."
        )
    }

    retriever = DenseRetriever(documents)

    results = retriever.search(
        "energy from the sun",
        top_k=2
    )

    assert len(results) == 2
    assert results[0][0] in documents
    assert results[0][1] >= results[1][1]
