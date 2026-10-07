from __future__ import annotations

import os
from functools import lru_cache

from loguru import logger

from ragnar.cache.semantic_qdrant_cache import QdrantSemanticCache
from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
from ragnar.generation.simple_generator import Generator
from ragnar.indexing.pipeline_factory import build_index_pipeline
from ragnar.llm.ollama_llm import OllamaLLMAdapter
from ragnar.loaders.docling_loader import DoclingLoader
from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.pipelines.interfaces.indexing_pipeline import BaseIndexPipeline
from ragnar.pipelines.interfaces.query_pipeline import BaseQueryPipeline
from ragnar.pipelines.query.flat_query_pipeline import FlatQueryPipeline
from ragnar.pipelines.query.hierarchical_query_pipeline import (
    HierarchicalQueryPipeline,
)
from ragnar.prompts.prompt_loader import PromptLoader
from ragnar.rerank.fast_embed_reranker import FastEmbedReranker
from ragnar.retrieval.hybrid_qdrant_retriever import HybridQdrantRetriever
from ragnar.session.interfaces.session_store import BaseSessionStore
from ragnar.session.redis_session_store import RedisSessionStore

# Configuration from environment
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
SESSION_STORE = os.getenv('SESSION_STORE', 'redis')
PARENT_COLLECTION = os.getenv('PARENT_COLLECTION', 'sections')
CHILD_COLLECTION = os.getenv('CHILD_COLLECTION', 'subsections')
LINKAGE_ID = os.getenv('LINKAGE_ID', 'parent_section_id')

logger.info(
    f"Configuration loaded | Qdrant: {QDRANT_URL} | Ollama: {OLLAMA_URL}",
)

# Core


def get_loader() -> BaseLoader:
    """Factory for document loader.

    Deliberately NOT cached: a DoclingLoader holds the layout/table models
    (torch, GBs of RAM). A fresh one per index request lets the memory be
    released when the request ends, instead of sitting next to the query
    models forever.
    """
    logger.info('Creating DoclingLoader')
    return DoclingLoader()


@lru_cache
def get_session_store() -> BaseSessionStore | None:
    """Factory for session store"""
    if SESSION_STORE == 'redis':
        logger.info(f"Using Redis session store: {REDIS_URL}")
        return RedisSessionStore(REDIS_URL)
    else:
        logger.info('No session store configured.')
        return None


@lru_cache
def get_prompt_loader() -> PromptLoader:
    """Factory for prompt loader"""
    logger.info('Creating PromptLoader')
    return PromptLoader()


@lru_cache
def get_llm_adapter():
    """Factory for LLM adapter"""
    logger.info(f"Creating OllamaLLMAdapter: {OLLAMA_URL}")
    return OllamaLLMAdapter(base_url=OLLAMA_URL)


@lru_cache
def get_retriever():
    """Factory for retriever"""
    logger.info(f"Creating HybridQdrantRetriever: {QDRANT_URL}")
    return HybridQdrantRetriever(
        url=QDRANT_URL,
        dense_embedding=get_dense_embedding(),
        sparse_embedding=get_sparse_embedding(),
    )


@lru_cache
def get_reranker():
    """Factory for reranker"""
    logger.info('Creating FastEmbedReranker')
    return FastEmbedReranker()


@lru_cache
def get_generator():
    """Factory for generator"""
    logger.info('Creating Generator')
    llm = get_llm_adapter()
    prompt_loader = get_prompt_loader()
    return Generator(llm, prompt_loader)


@lru_cache
def get_dense_embedding() -> DenseFastEmbedding:
    logger.info('Creating DenseFastEmbedding')
    return DenseFastEmbedding()


@lru_cache
def get_sparse_embedding() -> SparseFastEmbedding:
    logger.info('Creating SparseFastEmbedding')
    return SparseFastEmbedding()


# Index

def get_flat_index_pipeline(
    loader: BaseLoader | None = None,
) -> BaseIndexPipeline:
    """Factory for flat index pipeline"""
    logger.info('Creating FlatIndexPipeline')

    return build_index_pipeline(
        'flat',
        loader=loader if loader is not None else get_loader(),
        dense_embedder=get_dense_embedding(),
        sparse_embedder=get_sparse_embedding(),
        qdrant_url=QDRANT_URL,
        parent_collection_name=PARENT_COLLECTION,
        child_collection_name=CHILD_COLLECTION,
    )


def get_hierarchical_index_pipeline(
    loader: BaseLoader | None = None,
) -> BaseIndexPipeline:
    """Factory for hierarchical index pipeline"""
    logger.info('Creating HierarchicalIndexPipeline')

    return build_index_pipeline(
        'hierarchical',
        loader=loader if loader is not None else get_loader(),
        dense_embedder=get_dense_embedding(),
        sparse_embedder=get_sparse_embedding(),
        qdrant_url=QDRANT_URL,
        parent_collection_name=PARENT_COLLECTION,
        child_collection_name=CHILD_COLLECTION,
    )


def get_index_pipeline(
    strategy: str = 'hierarchical',
    loader: BaseLoader | None = None,
) -> BaseIndexPipeline:
    """Factory for index pipeline by strategy"""
    if strategy == 'hierarchical':
        return get_hierarchical_index_pipeline(loader)
    elif strategy == 'flat':
        return get_flat_index_pipeline(loader)
    else:
        raise ValueError(f"Unknown index pipeline strategy: {strategy}")


# Query

def get_flat_query_pipeline() -> BaseQueryPipeline:
    """Factory for flat query pipeline"""
    logger.info('Creating FlatQueryPipeline')

    llm_adapter = get_llm_adapter()
    prompt_loader = get_prompt_loader()
    retriever = get_retriever()
    reranker = get_reranker()
    generator = get_generator()
    session_store = get_session_store()
    semantic_cache = get_semantic_cache()

    return FlatQueryPipeline(
        llm_adapter=llm_adapter,
        prompt_loader=prompt_loader,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        session_store=session_store,
        semantic_cache=semantic_cache,
        collection_name=CHILD_COLLECTION,
    )


def get_hierarchical_query_pipeline() -> BaseQueryPipeline:
    """Factory for hierarchical query pipeline"""
    logger.info('Creating HierarchicalQueryPipeline')

    llm_adapter = get_llm_adapter()
    prompt_loader = get_prompt_loader()
    retriever = get_retriever()
    reranker = get_reranker()
    generator = get_generator()
    session_store = get_session_store()
    semantic_cache = get_semantic_cache()

    return HierarchicalQueryPipeline(
        llm_adapter=llm_adapter,
        prompt_loader=prompt_loader,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        session_store=session_store,
        semantic_cache=semantic_cache,
        parent_collection_name=PARENT_COLLECTION,
        child_collection_name=CHILD_COLLECTION,
        linkage_id=LINKAGE_ID,
    )


def get_query_pipeline(strategy: str = 'hierarchical') -> BaseQueryPipeline:
    """Factory for query pipeline by strategy"""
    if strategy == 'hierarchical':
        return get_hierarchical_query_pipeline()
    elif strategy == 'flat':
        return get_flat_query_pipeline()
    else:
        raise ValueError(f"Unknown query pipeline strategy: {strategy}")

# cache


@lru_cache
def get_semantic_cache() -> QdrantSemanticCache:
    return QdrantSemanticCache(
        url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
        collection_name='ragnar_semantic_cache',
        threshold=0.90,
        ttl_seconds=2592000,  # 30 days
        embedding=get_dense_embedding(),
    )


# Startup


def preload_models() -> None:
    """Load query-time models only.

    Document ingestion models are owned exclusively by the
    dedicated indexer worker.
    """
    logger.info('Preloading query models...')

    get_dense_embedding()
    get_sparse_embedding()
    get_reranker()
    get_semantic_cache()
    get_retriever()

    logger.info('All query models loaded')
