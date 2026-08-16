# AQUILA
A search engine built from scratch, designed to explore the fundamentals of information retrieval.


## Tokenizer Unit

**Tokenizer** is a foundation of the Aquila search engine pipeline. before everything else, raw text documents must be broken down into tokens.

## Indexing Unit

**Inverted Index** maps each unique token to the documents containing it along with term frequencies, making it a fast document lookup without scanning the full text of documents.

## Ranking Unit ( Okapi BM25 )

**BM25 Ranker** (`BM25Ranker` in `src/ranking.py`) scores and ranks candidate documents for a given query using the Okapi BM25 algorithm. It incorporates probabilistic IDF weighting, term frequency saturation ($k_1$), and document length normalization ($b$) relative to average document length.


## Candidate Retrieval Unit

**Search Retrieval** processes multi-term search queries by tokenizing the query and performing posting list union operations across the inverted index to efficiently identify candidate documents.

---

> [!TIP]  
> **Running Tests:** The `tests/` directory contains unit test scripts for each component (tokenizer, index, ranking, search). You can run individual tests using Python's module flag:
> ```bash
> python3 -m tests.test_tokenizer
> python3 -m tests.test_index
> python3 -m tests.test_ranking
> python3 -m tests.test_search
> ```