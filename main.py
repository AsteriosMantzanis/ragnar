from __future__ import annotations

import asyncio
from pathlib import Path

from ragnar.chunkers.fixed_chunker import FixedChunker
from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.index.Qdrant_Indexer import QdrantIndexer
from ragnar.loaders.docling_loader import DoclingLoader
from ragnar.retrieval.dense_qdrant_retriever import DenseQdrantRetriever


async def main():

    loader = DoclingLoader()
    documents = loader.load(
        Path('C:/Users/aster/Desktop/manuals/Hammer_Driver_Drill_HP0300.pdf'),
    )
    print(f"Loaded {len(documents)} documents.")
    print(documents[0])

    chunks = FixedChunker().chunk(documents[0])
    print(f"Created {len(chunks)} chunks.")
    print(chunks[0])

    indexer = QdrantIndexer(
        dense_embedder=DenseFastEmbedding(),
        sparse_embedder=SparseFastEmbedding(),
        collection_name='testttdbsrthsrth',
    )

    await indexer.index(chunks)

    retriever = DenseQdrantRetriever(collection_name='testttdbsrthsrth')
    search_results = await retriever.retrieve(
        'What is the maximum \
        torque of the Hammer Driver Drill HP0300?', top_k=5,
    )
    print(search_results)


if __name__ == '__main__':
    asyncio.run(main())
