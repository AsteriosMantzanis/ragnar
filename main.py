from __future__ import annotations

import asyncio
from pathlib import Path

from ragnar.chunkers.fixed_chunker import FixedChunker
from ragnar.embeddings.ollama_embedding import OllamaEmbedding
from ragnar.index.Qdrant_Indexer import QdrantIndexer
from ragnar.loaders.docling_loader import DoclingLoader
# from docling.document_converter import DocumentConverter


async def main():

    # converter = DocumentConverter()
    # result = converter.convert(
    #     Path('C:/Users/aster/Desktop/manuals/Hammer_Driver_Drill_HP0300.pdf'),
    # )
    # doc = result.document
    # table_md = doc.tables[0].export_to_markdown(doc)
    # return doc, table_md

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
        embedder=OllamaEmbedding(),
        collection_name='manuals',
    )

    await indexer.index(chunks)

    # Query to see first item
    points, _ = indexer.client.scroll(
        collection_name=indexer.collection_name,
        limit=1,
    )

    if points:
        point = points[0]
        print('First indexed point:')
        print(f"ID: {point.id}")
        print(f"Text: {point.payload['text'][:100]}...")
        print(f"Metadata: {point.payload}")
    else:
        print('No points in collection')


if __name__ == '__main__':
    asyncio.run(main())
