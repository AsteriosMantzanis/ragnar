from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.index.Qdrant_Indexer import QdrantIndexer
from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.pipelines.interfaces.indexing_pipeline import BaseIndexPipeline


class HierarchicalIndexPipeline(BaseIndexPipeline):
    """Index pipeline with hierarchical chunking
    (parent + child collections)"""

    def __init__(
        self,
        loader: BaseLoader,
        parent_chunker,
        child_chunker,
        parent_collection_name: str,
        child_collection_name: str,
        store_url: str = 'http://localhost:6333',
    ):
        super().__init__(loader)
        self.parent_chunker = parent_chunker
        self.child_chunker = child_chunker
        self.parent_collection_name = parent_collection_name
        self.child_collection_name = child_collection_name
        self.store_url = os.getenv('QDRANT_URL')

        self.parent_indexer = QdrantIndexer(
            dense_embedder=DenseFastEmbedding(),
            sparse_embedder=SparseFastEmbedding(),
            collection_name=parent_collection_name,
            url=self.store_url if self.store_url else store_url,
        )

        self.child_indexer = QdrantIndexer(
            dense_embedder=DenseFastEmbedding(),
            sparse_embedder=SparseFastEmbedding(),
            collection_name=child_collection_name,
            url=self.store_url if self.store_url else store_url,
        )

        logger.info(
            f"Hierarchical indexing: {child_collection_name} / "
            f"{parent_collection_name}",
        )

    async def index(self, path: Path) -> dict:
        """Index file(s) with hierarchical chunking"""
        path = Path(path)
        logger.info(f"Starting hierarchical index pipeline: {path}")

        # Load all files
        logger.info('Step 1/5: Loading documents')
        documents = self.loader.load(path)
        logger.info(f"Loaded {len(documents)} documents")

        if not documents:
            logger.warning('No documents loaded')
            return {'status': 'no_documents', 'documents_indexed': 0}

        total_parent_chunks = 0
        total_child_chunks = 0

        # Process each document
        for i, doc in enumerate(documents, 1):
            logger.info(
                f"Processing document {i}/{len(documents)}: "
                f"{doc.metadata['filename']}",
            )

            # 2. Chunk parent
            logger.info('  Chunking into parent chunks')
            parent_chunks = self.parent_chunker.chunk(doc)
            logger.info(f"  Created {len(parent_chunks)} parent chunks")

            # 3. Chunk child
            logger.info('  Chunking into child chunks')
            child_chunks = self.child_chunker.chunk(doc)
            logger.info(f"  Created {len(child_chunks)} child chunks")

            total_parent_chunks += len(parent_chunks)
            total_child_chunks += len(child_chunks)

            # 4. Index parent
            logger.info('  Indexing parent chunks')
            await self.parent_indexer.index(parent_chunks)

            # 5. Index child
            logger.info('  Indexing child chunks')
            await self.child_indexer.index(child_chunks)

        logger.info('Hierarchical indexing complete')

        return {
            'status': 'indexed',
            'documents_indexed': len(documents),
            'parent_chunks': total_parent_chunks,
            'child_chunks': total_child_chunks,
            'parent_collection': self.parent_collection_name,
            'child_collection': self.child_collection_name,
        }
