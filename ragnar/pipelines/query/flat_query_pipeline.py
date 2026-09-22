from __future__ import annotations

from ragnar.pipelines.interfaces.query_pipeline import BaseQueryPipeline


class FlatQueryPipeline(BaseQueryPipeline):
    """Query pipeline with flat retrieval (single collection)"""

    def __init__(self, *args, collection_name: str = 'subsections', **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_name = collection_name

    async def _retrieve(self, query: str) -> list[dict]:
        """Flat retrieval from single collection"""
        return await self.retriever.retrieve(
            query=query,
            collection_name=self.collection_name,
            top_k=10,
        )
