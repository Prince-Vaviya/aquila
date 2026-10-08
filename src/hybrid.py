def reciprocal_rank_fusion(
    ranked_lists,
    top_k=5,
    k=60
):
    """
    Combine multiple ranked lists using
    Reciprocal Rank Fusion.

    ranked_lists:

        [
            bm25_results,
            dense_results
        ]

    Each result:

        (document_id, score)
    """

    fused_scores = {}

    for ranked_list in ranked_lists:

        for rank, (
            document_id,
            _
        ) in enumerate(
            ranked_list,
            start=1
        ):

            if document_id not in fused_scores:

                fused_scores[
                    document_id
                ] = 0.0

            fused_scores[
                document_id
            ] += 1 / (
                k + rank
            )

    results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return results[:top_k]


def hybrid_search(
    query,
    bm25_search_fn,
    dense_search_fn,
    top_k=5
):

    bm25_results = bm25_search_fn(
        query,
        top_k
    )

    dense_results = dense_search_fn(
        query,
        top_k
    )

    hybrid_results = reciprocal_rank_fusion(
        [
            bm25_results,
            dense_results
        ],
        top_k=top_k
    )

    return hybrid_results