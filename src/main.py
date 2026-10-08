from src.engine import RetrievalEngine


def print_results(
    title,
    results
):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    if not results:

        print("No results found.")

        return

    for rank, (
        document_id,
        score
    ) in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{document_id:<10} "
            f"score = {score:.6f}"
        )


def main():

    print()
    print("=" * 60)
    print("INFORMATION RETRIEVAL ENGINE")
    print("=" * 60)

    query = input(
        "\nEnter query: "
    ).strip()

    if not query:

        print("Query cannot be empty.")

        return

    print(
        "\nLoading retrieval engine..."
    )

    engine = RetrievalEngine(
        data_dir="data"
    )

    results = engine.search(
        query,
        top_k=5
    )

    print_results(
        "BM25 RESULTS",
        results["bm25"]
    )

    print_results(
        "DENSE RESULTS",
        results["dense"]
    )

    print_results(
        "HYBRID RESULTS",
        results["hybrid"]
    )

    print()
    print("=" * 60)
    print("SEARCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()