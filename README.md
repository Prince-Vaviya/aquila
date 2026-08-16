# AQUILA
A search engine built from scratch, designed to explore the fundamentals of information retrieval.


## Tokenizer Unit

### Significance
The **Tokenizer** is a foundation of the Aquila search engine pipeline. before everything else, raw text documents must be broken down into tokens.

### Functionality
The `tokenize()` function in `src/tokenizer.py` processes input text by:
- Converting all characters to lowercase for case-insensitive matching.
- Extracting alphanumeric tokens (`[a-z0-9]+`) while removing punctuation and whitespace.

---

> [!TIP]  
> **Running Tests:** The `tests/` directory contains unit test scripts for each component (tokenizer, ). You can run individual tests using Python's module flag:
> ```bash
> python3 -m tests.test_tokenizer
> ```