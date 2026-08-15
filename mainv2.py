from __future__ import annotations

import asyncio
import os

from ragnar.generation.simple_generator import Generator
from ragnar.grounding.hf_grounding import HF_Grounding
from ragnar.llm.ollama_llm import OllamaLLMAdapter
from ragnar.pipelines.flat_query_pipeline import FlatQueryPipeline
from ragnar.prompts.prompt_loader import PromptLoader
from ragnar.rerank.fast_embed_reranker import FastEmbedReranker
from ragnar.retrieval.hybrid_qdrant_retriever import HybridQdrantRetriever
from ragnar.session.redis_session_store import RedisSessionStore


async def main():
    # Initialize components
    llm_adapter = OllamaLLMAdapter()
    prompt_loader = PromptLoader()
    retriever = HybridQdrantRetriever()
    reranker = FastEmbedReranker()
    generator = Generator(llm_adapter, prompt_loader)
    grounder = HF_Grounding()

    # Initialize session store
    session_store_type = os.getenv('SESSION_STORE')
    if session_store_type == 'redis':
        store = RedisSessionStore(
            os.getenv('REDIS_URL', 'redis://localhost:6379'),
        )
        await store.connect()
    else:
        from ragnar.session.in_memory_session_store import InMemorySessionStore
        store = InMemorySessionStore()

    # Initialize pipeline
    pipeline = FlatQueryPipeline(
        llm_adapter=llm_adapter,
        prompt_loader=prompt_loader,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        grounder=grounder,
        session_store=store,
        collection_name='subsections',
    )

    # Run queries
    session_id = 'user_1232'

    # Query 1
    result1 = await pipeline.query(
        user_query='What are the drilling capacities?',
        session_id=session_id,
    )
    print(f"\nQuery 1 Answer:\n{result1['answer']}")
    print(f"Grounding: {result1['grounded_percentage']:.0f}%")
    print(f"Sources: {len(result1['sources'])}")

    # Query 2 (same session)
    result2 = await pipeline.query(
        user_query='What about wood?',
        session_id=session_id,
    )
    print(f"\nQuery 2 Answer:\n{result2['answer']}")
    print(f"Conversation length: {result2['conversation_length']} messages")

    # Cleanup
    if session_store_type == 'redis':
        await store.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
