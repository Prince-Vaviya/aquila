from src.hybrid import reciprocal_rank_fusion


def test_rrf():
    bm25_results = [
        ("1.txt", 10.0),
        ("2.txt", 8.0),
        ("3.txt", 5.0)
    ]

    dense_results = [
        ("2.txt", 0.9),
        ("1.txt", 0.8),
        ("4.txt", 0.7)
    ]

    results = reciprocal_rank_fusion(
        [bm25_results, dense_results],
        top_k=4
    )

    documents = [
        document_id
        for document_id, score in results
    ]

    assert "1.txt" in documents
    assert "2.txt" in documents
    assert "3.txt" in documents
    assert "4.txt" in documents


def test_rrf_top_k():
    results = reciprocal_rank_fusion(
        [[
            ("1.txt", 10),
            ("2.txt", 9),
            ("3.txt", 8)
        ]],
        top_k=2
    )

    assert len(results) == 2
