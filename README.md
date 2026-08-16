# AQUILA
A search engine built from scratch, designed to explore the fundamentals of information retrieval.


## Tokenizer Unit

**Tokenizer** is a foundation of the Aquila search engine pipeline. before everything else, raw text documents must be broken down into tokens.

## Indexing Unit

**Inverted Index** maps each unique token to the documents containing it along with term frequencies, making it a fast document lookup without scanning the full text of documents.

## Ranking Unit ( TF-IDF )

**TF-IDF Ranker** computes similarity scores between a search query and retrieved candidate documents using Term Frequency (TF) and Inverse Document Frequency (IDF), ranking results by relevance.

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