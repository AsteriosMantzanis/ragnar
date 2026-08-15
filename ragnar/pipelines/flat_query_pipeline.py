from __future__ import annotations

from ragnar.pipelines.query_pipeline import BaseQueryPipeline


class FlatQueryPipeline(BaseQueryPipeline):
    """Query pipeline with flat retrieval (single collection)"""

    def __init__(self, *args, collection_name: str = 'subsections', **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_name = collection_name

    async def _retrieve(self, expanded_queries: list[str]) -> list[dict]:
        """Flat retrieval from single collection"""
        all_results = []
        for q in expanded_queries:
            results = await self.retriever.retrieve(
                query=q,
                collection_name=self.collection_name,
                top_k=10,
            )
            all_results.extend(results)
        return all_results
