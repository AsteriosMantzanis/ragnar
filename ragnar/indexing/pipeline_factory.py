from __future__ import annotations

from ragnar.chunkers.section_chunker import SectionChunker
from ragnar.chunkers.subsection_chunker import SubsectionChunker
from ragnar.embeddings.interfaces.base_embedding import BaseEmbedding
from ragnar.index.Qdrant_Indexer import QdrantIndexer
from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.pipelines.index.flat_index_pipeline import FlatIndexPipeline
from ragnar.pipelines.index.hier_index_pipeline import (
    HierarchicalIndexPipeline,
)
from ragnar.pipelines.interfaces.indexing_pipeline import BaseIndexPipeline


def build_index_pipeline(
    strategy: str,
    *,
    loader: BaseLoader,
    dense_embedder: BaseEmbedding,
    sparse_embedder: BaseEmbedding | None,
    qdrant_url: str,
    parent_collection_name: str,
    child_collection_name: str,
) -> BaseIndexPipeline:
    if strategy == 'flat':
        indexer = QdrantIndexer(
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            collection_name=child_collection_name,
            url=qdrant_url,
        )
        return FlatIndexPipeline(
            loader=loader,
            chunker=SubsectionChunker(),
            indexer=indexer,
            collection_name=child_collection_name,
        )

    if strategy == 'hierarchical':
        parent_indexer = QdrantIndexer(
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            collection_name=parent_collection_name,
            url=qdrant_url,
        )
        child_indexer = QdrantIndexer(
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            collection_name=child_collection_name,
            url=qdrant_url,
        )
        return HierarchicalIndexPipeline(
            loader=loader,
            parent_chunker=SectionChunker(),
            child_chunker=SubsectionChunker(),
            parent_indexer=parent_indexer,
            child_indexer=child_indexer,
            parent_collection_name=parent_collection_name,
            child_collection_name=child_collection_name,
        )

    raise ValueError(f"Unknown index pipeline strategy: {strategy}")
