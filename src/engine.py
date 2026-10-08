from src.document_store import DocumentStore
from src.index import build_index
from src.ranking import BM25Ranker
from src.search import search_bm25
from src.dense import DenseRetriever
from src.hybrid import hybrid_search


class RetrievalEngine:

    def __init__(
        self,
        data_dir="data",
        dense_model="all-MiniLM-L6-v2"
    ):

        # -------------------------
        # Load documents
        # -------------------------

        self.store = DocumentStore(
            data_dir
        )

        self.documents = (
            self.store.load()
        )

        # -------------------------
        # BM25
        # -------------------------

        self.index = build_index(
            self.documents
        )

        self.bm25_ranker = BM25Ranker(
            self.index
        )

        # -------------------------
        # Dense Retrieval
        # -------------------------

        self.dense_retriever = (
            DenseRetriever(
                self.documents,
                model_name=dense_model
            )
        )

    def bm25(
        self,
        query,
        top_k=5
    ):

        return search_bm25(
            query,
            self.index,
            self.bm25_ranker,
            top_k
        )

    def dense(
        self,
        query,
        top_k=5
    ):

        return self.dense_retriever.search(
            query,
            top_k
        )

    def hybrid(
        self,
        query,
        top_k=5
    ):

        bm25_results = self.bm25(
            query,
            top_k
        )

        dense_results = self.dense(
            query,
            top_k
        )

        return hybrid_search(
            query,
            lambda q, k: bm25_results,
            lambda q, k: dense_results,
            top_k
        )

    def search(
        self,
        query,
        top_k=5
    ):

        return {
            "query": query,

            "bm25": self.bm25(
                query,
                top_k
            ),

            "dense": self.dense(
                query,
                top_k
            ),

            "hybrid": self.hybrid(
                query,
                top_k
            )
        }