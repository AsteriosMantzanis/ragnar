from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from ragnar.chunkers.interfaces.base_chunker import BaseChunker
from ragnar.index.interfaces.base_indexer import BaseIndexer
from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.pipelines.interfaces.indexing_pipeline import BaseIndexPipeline
from ragnar.utils.memory import release_memory


class HierarchicalIndexPipeline(BaseIndexPipeline):
    """Index pipeline with hierarchical chunking
    (parent + child collections)"""

    def __init__(
        self,
        loader: BaseLoader,
        parent_chunker: BaseChunker,
        child_chunker: BaseChunker,
        parent_indexer: BaseIndexer,
        child_indexer: BaseIndexer,
        parent_collection_name: str,
        child_collection_name: str,
        store_url: str = 'http://localhost:6333',
    ):
        super().__init__(loader)
        self.parent_chunker = parent_chunker
        self.child_chunker = child_chunker
        self.parent_collection_name = parent_collection_name
        self.child_collection_name = child_collection_name
        self.parent_indexer = parent_indexer
        self.child_indexer = child_indexer
        self.store_url = os.getenv('QDRANT_URL')

        logger.info(
            f"Hierarchical indexing: {child_collection_name} / "
            f"{parent_collection_name}",
        )

    async def index(self, path: Path) -> dict:
        """Index file(s) with hierarchical chunking"""
        path = Path(path)
        logger.info(f"Starting hierarchical index pipeline: {path}")

        # Stream files so Docling does not retain the whole corpus.
        logger.info('Step 1/5: Streaming documents')
        progress = self.loader.resume_progress()
        documents_indexed = progress.get('documents_indexed', 0)
        total_parent_chunks = progress.get('parent_chunks', 0)
        total_child_chunks = progress.get('child_chunks', 0)
        loaded_any = documents_indexed > 0

        # Process one document at a time.
        for document_number, doc in self.loader.iter_load_indexed(path):
            loaded_any = True
            documents_indexed = document_number
            logger.info(
                f"Processing document {document_number}: "
                f"{doc.metadata['filename']}",
            )

            # 2. Chunk parent
            logger.info('  Chunking into parent chunks')
            parent_chunks = self.parent_chunker.chunk(doc)
            logger.info(f"  Created {len(parent_chunks)} parent chunks")

            total_parent_chunks += len(parent_chunks)

            # 4. Index parent before creating the child chunk list.
            logger.info('  Indexing parent chunks')
            await self.parent_indexer.index(parent_chunks)
            del parent_chunks

            # 5. Create + index child chunks after parent memory is released.
            logger.info('  Chunking into child chunks')
            child_chunks = self.child_chunker.chunk(doc)
            logger.info(f"  Created {len(child_chunks)} child chunks")
            total_child_chunks += len(child_chunks)

            logger.info('  Indexing child chunks')
            await self.child_indexer.index(child_chunks)

            del child_chunks
            del doc
            release_memory()
            self.loader.mark_completed(
                document_number,
                {
                    'documents_indexed': documents_indexed,
                    'parent_chunks': total_parent_chunks,
                    'child_chunks': total_child_chunks,
                },
            )

        if not loaded_any:
            logger.warning('No documents loaded')
            return {'status': 'no_documents', 'documents_indexed': 0}

        logger.info('Hierarchical indexing complete')

        return {
            'status': 'indexed',
            'documents_indexed': documents_indexed,
            'parent_chunks': total_parent_chunks,
            'child_chunks': total_child_chunks,
            'parent_collection': self.parent_collection_name,
            'child_collection': self.child_collection_name,
        }
