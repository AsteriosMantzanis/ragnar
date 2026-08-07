from __future__ import annotations

import asyncio
from pathlib import Path

from ragnar.chunkers.page_chunker import PageChunker
from ragnar.chunkers.section_chunker import SectionChunker
from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.index.Qdrant_Indexer import QdrantIndexer
from ragnar.loaders.docling_loader import DoclingLoader
from ragnar.retrieval.hybrid_qdrant_retriever import HybridQdrantRetriever


async def main():

    loader = DoclingLoader()
    documents = loader.load(
        Path('C:/Users/aster/Desktop/manuals/Hammer_Driver_Drill_HP0300.pdf'),
    )
    print(f"Loaded {len(documents)} documents.")
    print(documents[0])

    page = PageChunker().chunk(documents[0])
    sections = SectionChunker().chunk(documents[0])

    indexer_pages = QdrantIndexer(
        dense_embedder=DenseFastEmbedding(),
        sparse_embedder=SparseFastEmbedding(),
        collection_name='ragnar_pages',
    )

    await indexer_pages.index(page)

    indexer_sections = QdrantIndexer(
        dense_embedder=DenseFastEmbedding(),
        sparse_embedder=SparseFastEmbedding(),
        collection_name='ragnar_sections',
    )
    await indexer_sections.index(sections)

    retriever = HybridQdrantRetriever()
    search_results = await retriever.retrieve_hierarchical(
        'What is the maximum \
        torque of the Hammer Driver Drill HP0300?', top_k=5,
        parent_collection_name='ragnar_pages',
        child_collection_name='ragnar_sections',
    )
    print('Sections:')
    for section in search_results['sections']:
        print(section['text'])
        print('=' * 80)

    print('\nPages:')
    for page in search_results['pages']:
        print(page['text'])
        print('=' * 80)


if __name__ == '__main__':
    asyncio.run(main())
