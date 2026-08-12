from __future__ import annotations

import asyncio
from pathlib import Path

from ragnar.chunkers.section_chunker import SectionChunker
from ragnar.chunkers.subsection_chunker import SubsectionChunker
from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.generation.simple_generator import Generator
from ragnar.grounding.hf_grounding import HF_Grounding
from ragnar.index.Qdrant_Indexer import QdrantIndexer
from ragnar.llm.ollama_llm import OllamaLLMAdapter
from ragnar.loaders.docling_loader import DoclingLoader
from ragnar.prompts.prompt_loader import PromptLoader
from ragnar.rerank.fast_embed_reranker import FastEmbedReranker
from ragnar.retrieval.hybrid_qdrant_retriever import HybridQdrantRetriever


async def main():

    loader = DoclingLoader()
    documents = loader.load(
        Path('C:/Users/aster/Desktop/manuals/Hammer_Driver_Drill_HP0300.pdf'),
    )
    print(f"Loaded {len(documents)} documents.")
    print(documents[0])

    subsections = SubsectionChunker().chunk(documents[0])
    sections = SectionChunker().chunk(documents[0])

    indexer_subsections = QdrantIndexer(
        dense_embedder=DenseFastEmbedding(),
        sparse_embedder=SparseFastEmbedding(),
        collection_name='subsections',
    )

    await indexer_subsections.index(subsections)

    indexer_sections = QdrantIndexer(
        dense_embedder=DenseFastEmbedding(),
        sparse_embedder=SparseFastEmbedding(),
        collection_name='sections',
    )

    await indexer_sections.index(sections)

    query = 'What are the drilling capacities?'

    retriever = HybridQdrantRetriever()
    search_results = await retriever.retrieve_hierarchical(
        query=query, top_k=20,
        parent_collection_name='sections',
        child_collection_name='subsections',
        linkage_id='parent_section_id',
    )

    search_results = await retriever.retrieve_hierarchical(
        query=query, top_k=20,
        parent_collection_name='sections',
        child_collection_name='subsections',
        linkage_id='parent_section_id',
    )

    # reranking
    reranker = FastEmbedReranker()
    scored_results = await reranker.score_hierarchical(
        query=query,
        results=search_results,
        child_entity='subsections',
    )

    # Generation
    llm_adapter = OllamaLLMAdapter()
    prompt_loader = PromptLoader()

    generator = Generator(llm_adapter, prompt_loader)

    answer = await generator.generate_hierarchical(
        query=query,
        context=scored_results,
        prompt_template='simple_qa',
        generation_entity='subsections',
    )

    # grounding
    grounding = HF_Grounding()
    await grounding.ground_hierarchical(answer, scored_results, 'subsections')

    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{answer}")


if __name__ == '__main__':
    asyncio.run(main())
