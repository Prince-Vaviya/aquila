import numpy as np

from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(
        self,
        documents,
        model_name="all-MiniLM-L6-v2"
    ):

        self.documents = documents

        self.document_ids = list(
            documents.keys()
        )

        self.model = SentenceTransformer(
            model_name
        )

        texts = [
            documents[doc_id]
            for doc_id in self.document_ids
        ]

        self.embeddings = (
            self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )

    def search(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
        )

        scores = (
            self.embeddings
            @ query_embedding
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[
            :top_k
        ]:

            document_id = (
                self.document_ids[index]
            )

            score = float(
                scores[index]
            )

            results.append(
                (
                    document_id,
                    score
                )
            )

        return results