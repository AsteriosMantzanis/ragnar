from __future__ import annotations

import asyncio
from pathlib import Path

from ragnar.chunkers.page_chunker import PageChunker
from ragnar.chunkers.section_chunker import SectionChunker
from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.generation.simple_generator import Generator
from ragnar.index.Qdrant_Indexer import QdrantIndexer
from ragnar.llm.ollama_llm import OllamaLLMAdapter
from ragnar.loaders.docling_loader import DoclingLoader
from ragnar.prompts.prompt_loader import PromptLoader
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

    query = 'What is the maximum \
        torque of the Hammer Driver Drill HP0300?'

    retriever = HybridQdrantRetriever()
    search_results = await retriever.retrieve_hierarchical(
        query=query, top_k=5,
        parent_collection_name='ragnar_pages',
        child_collection_name='ragnar_sections',
    )

    # Generation
    llm_adapter = OllamaLLMAdapter()
    prompt_loader = PromptLoader()
    generator = Generator(llm_adapter, prompt_loader)

    answer = await generator.generate(
        query=query,
        context=search_results['sections'],
        prompt_template='simple_qa',
    )

    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{answer}")


if __name__ == '__main__':
    asyncio.run(main())
