from __future__ import annotations

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from ragnar.embeddings.interfaces.base_embedding import BaseEmbedding
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever
load_dotenv()


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        strategy: str,
        embedder: BaseEmbedding,
    ):
        self.url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.client = QdrantClient(url=self.url)
        self.embedder = embedder
        self.collection = collection_name
        self.strategy = strategy

    def retrieve(self, query: str, top_k: int = 5) -> list:
        return []
