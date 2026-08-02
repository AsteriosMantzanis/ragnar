from __future__ import annotations

import asyncio
from pathlib import Path

from ragnar.chunkers.fixed_chunker import FixedChunker
from ragnar.embeddings.ollama_embedding import OllamaEmbedding
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

    embedding = OllamaEmbedding()
    embeddings = await embedding.embed([i.text for i in chunks])
    print(f"Created {len(embeddings)} embeddings.")
    print(embeddings[0])


if __name__ == '__main__':
    asyncio.run(main())
