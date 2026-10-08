from src.engine import RetrievalEngine


def test_full_retrieval_engine(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

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
            "for life on Earth."
        ),
        "4.txt": (
            "Mars is a rocky planet in the Solar System."
        ),
        "5.txt": (
            "Plants use sunlight to produce energy "
            "through photosynthesis."
        ),
        "6.txt": (
            "Earth supports many forms of life."
        )
    }

    for filename, text in documents.items():
        (data_dir / filename).write_text(
            text,
            encoding="utf-8"
        )

    engine = RetrievalEngine(
        data_dir=str(data_dir)
    )

    results = engine.search(
        "energy from the sun",
        top_k=5
    )

    assert "bm25" in results
    assert "dense" in results
    assert "hybrid" in results

    assert len(results["bm25"]) <= 5
    assert len(results["dense"]) == 5
    assert len(results["hybrid"]) <= 5
