from __future__ import annotations

from index.interfaces.base_indexer import BaseIndexer
from qdrant_client import QdrantClient

from ragnar.models.chunk import Chunk


class QdrantIndexer(BaseIndexer):
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)

    async def index(self, chunks: list[Chunk]) -> None:
        # Implement the indexing logic for Qdrant here
        pass
