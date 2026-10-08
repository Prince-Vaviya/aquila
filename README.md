# Aquila 🦅

Aquila is a lightweight **Hybrid Search Engine** built in Python. It combines traditional keyword search (**BM25**) and AI-powered semantic search (**Dense Embeddings**) into a single hybrid search system using **Reciprocal Rank Fusion (RRF)**.

---

## 💡 How It Works

Aquila gives you three search modes depending on your needs:

1. **Keyword Search (BM25 + Proximity)**
   - Matches exact words in documents.
   - Uses term frequency and document length normalization.
   - Boosts documents where search terms appear close together.

2. **Semantic Search (Dense Embeddings)**
   - Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to convert text into vector embeddings.
   - Understands context, synonyms, and meaning (even if exact words don't match).

3. **Hybrid Search (Best of Both Worlds)**
   - Combines results from both Keyword and Semantic search using **Reciprocal Rank Fusion (RRF)**.
   - Delivers accurate results for both exact keywords and broad conceptual queries.

---

## 📁 Project Structure

```text
aquila/
├── data/              # Text documents to index (.txt files)
├── src/
│   ├── tokenizer.py   # Cleans and splits text into tokens
│   ├── index.py       # Builds inverted index with term positions
│   ├── ranking.py     # BM25 and proximity scoring algorithms
│   ├── search.py      # Keyword candidate retrieval
│   ├── dense.py       # Vector embeddings and semantic search
│   ├── hybrid.py      # Reciprocal Rank Fusion (RRF) logic
│   ├── engine.py      # Main RetrievalEngine interface
│   └── main.py        # Interactive CLI search demo
├── tests/             # Automated test suite (pytest)
└── requirements.txt   # Project dependencies
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repo
git clone https://github.com/Prince-Vaviya/aquila.git
cd aquila

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Try Interactive Search

Run the interactive search terminal:

```bash
python -m src.main
```

Type any search query (for example: `energy from the sun`) to see BM25, Dense, and Hybrid results side by side.

---

## 💻 Python Usage

You can easily import and use Aquila in your Python scripts:

```python
from src.engine import RetrievalEngine

# Initialize the engine with your documents folder
engine = RetrievalEngine(data_dir="data")

query = "energy from the sun"

# 1. Run Hybrid Search (Recommended)
results = engine.hybrid(query, top_k=5)
for doc_id, score in results:
    print(f"Doc: {doc_id} | Score: {score:.4f}")

# 2. Or run individual search modes
bm25_results = engine.bm25(query, top_k=5)
dense_results = engine.dense(query, top_k=5)

# 3. Get all results in one dictionary
all_results = engine.search(query, top_k=5)
```

---

## 🧪 Running Tests

Run all unit and integration tests with:

```bash
pytest
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **[Sentence-Transformers](https://sbert.net/)**: Semantic embeddings
- **[NumPy](https://numpy.org/)**: Fast vector operations
- **[Pytest](https://pytest.org/)**: Automated testing
