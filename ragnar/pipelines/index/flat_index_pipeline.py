from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from ragnar.chunkers.interfaces.base_chunker import BaseChunker
from ragnar.index.interfaces.base_indexer import BaseIndexer
from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.pipelines.interfaces.indexing_pipeline import BaseIndexPipeline


class FlatIndexPipeline(BaseIndexPipeline):
    """Index pipeline with flat chunking (single collection)"""

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        collection_name: str,
        indexer: BaseIndexer,
        store_url: str = 'http://localhost:6333',
    ):
        super().__init__(loader)
        self.chunker = chunker
        self.collection_name = collection_name
        self.store_url = os.getenv('QDRANT_URL')
        self.indexer = indexer
        logger.info(f"Flat indexing into collection: {collection_name}")

    async def index(self, path: Path) -> dict:
        """Index file(s) with flat chunking"""
        path = Path(path)
        logger.info(f"Starting flat index pipeline: {path}")

        # Load all files
        logger.info('Step 1/4: Loading documents')
        documents = self.loader.load(path)
        logger.info(f"Loaded {len(documents)} documents")

        if not documents:
            logger.warning('No documents loaded')
            return {'status': 'no_documents', 'documents_indexed': 0}

        total_chunks = 0

        # Process each document
        for i, doc in enumerate(documents, 1):
            logger.info(
                f"Processing document {i}/{len(documents)}: "
                f"{doc.metadata['filename']}",
            )

            # 2. Chunk
            logger.info('  Chunking document')
            chunks = self.chunker.chunk(doc)
            logger.info(f"  Created {len(chunks)} chunks")

            total_chunks += len(chunks)

            # 3. Index
            logger.info('  Indexing chunks')
            await self.indexer.index(chunks)

        logger.info('Flat indexing complete')

        return {
            'status': 'indexed',
            'documents_indexed': len(documents),
            'chunks': total_chunks,
            'collection': self.collection_name,
        }
