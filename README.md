# AQUILA
A search engine built from scratch, designed to explore the fundamentals of information retrieval.


## Tokenizer Unit

**Tokenizer** is a foundation of the Aquila search engine pipeline. before everything else, raw text documents must be broken down into tokens.

## Indexing Unit

**Inverted Index** maps each unique token to the documents containing it along with term frequencies, making it a fast document lookup without scanning the full text of documents.

---

> [!TIP]  
> **Running Tests:** The `tests/` directory contains unit test scripts for each component (tokenizer, index). You can run individual tests using Python's module flag:
> ```bash
> python3 -m tests.test_tokenizer
> python3 -m tests.test_index
> ```