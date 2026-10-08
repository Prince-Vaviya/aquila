from pathlib import Path


class DocumentStore:

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

        self.documents = {}

    def load(self):

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

        for path in sorted(
            self.data_dir.glob("*.txt")
        ):

            self.documents[
                path.name
            ] = path.read_text(
                encoding="utf-8"
            )

        return self.documents

    def get(self, document_id):

        return self.documents.get(
            document_id
        )

    def all_documents(self):

        return self.documents