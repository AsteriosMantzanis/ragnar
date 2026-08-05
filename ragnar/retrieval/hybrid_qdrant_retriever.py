from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch
from qdrant_client.models import Rrf
from qdrant_client.models import RrfQuery

from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever


class HybridQdrantRetriever(BaseRetriever):
    def __init__(self, url: str, collection_name: str):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.sparse_embedding = SparseFastEmbedding()
        self.dense_embedding = DenseFastEmbedding()

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:

        query_sparse_vector = await self.sparse_embedding.embed_query(query)
        query_dense_vector = await self.dense_embedding.embed_query(query)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_sparse_vector,
                    using='text-sparse',
                    limit=top_k,
                ),
                Prefetch(
                    query=query_dense_vector,
                    using='text-dense',
                    limit=top_k,
                ),
            ],
            query=RrfQuery(rrf=Rrf()),
            with_payload=True,
        )

        return search_results
