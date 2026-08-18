from __future__ import annotations

from ragnar.pipelines.interfaces.query_pipeline import BaseQueryPipeline


class HierarchicalQueryPipeline(BaseQueryPipeline):
    """Query pipeline with hierarchical retrieval
    (parent + child collections)"""

    def __init__(
        self,
        *args,
        parent_collection_name: str = 'sections',
        child_collection_name: str = 'subsections',
        linkage_id: str = 'parent_section_id',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parent_collection_name = parent_collection_name
        self.child_collection_name = child_collection_name
        self.linkage_id = linkage_id

    async def _retrieve(self, expanded_queries: list[str]) -> list[dict]:
        """Hierarchical retrieval (child + parent)"""
        all_results = []
        for q in expanded_queries:
            results = await self.retriever.retrieve_hierarchical(
                query=q,
                parent_collection_name=self.parent_collection_name,
                child_collection_name=self.child_collection_name,
                linkage_id=self.linkage_id,
                top_k=10,
            )
            all_results.extend(results[self.child_collection_name])
        return all_results
