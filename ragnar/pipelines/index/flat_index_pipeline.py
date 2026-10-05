from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from ragnar.chunkers.interfaces.base_chunker import BaseChunker
from ragnar.index.interfaces.base_indexer import BaseIndexer
from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.pipelines.interfaces.indexing_pipeline import BaseIndexPipeline
from ragnar.utils.memory import release_memory


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

        # Stream files so Docling does not retain the whole corpus.
        logger.info('Step 1/4: Streaming documents')
        progress = self.loader.resume_progress()
        documents_indexed = progress.get('documents_indexed', 0)
        total_chunks = progress.get('chunks', 0)
        loaded_any = documents_indexed > 0

        # Process one document at a time.
        for document_number, doc in self.loader.iter_load_indexed(path):
            loaded_any = True
            documents_indexed = document_number
            logger.info(
                f"Processing document {document_number}: "
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

            del chunks
            del doc
            release_memory()
            self.loader.mark_completed(
                document_number,
                {
                    'documents_indexed': documents_indexed,
                    'chunks': total_chunks,
                },
            )

        if not loaded_any:
            logger.warning('No documents loaded')
            return {'status': 'no_documents', 'documents_indexed': 0}

        logger.info('Flat indexing complete')

        return {
            'status': 'indexed',
            'documents_indexed': documents_indexed,
            'chunks': total_chunks,
            'collection': self.collection_name,
        }
