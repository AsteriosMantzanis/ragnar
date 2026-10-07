from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import asdict
from typing import Any

from loguru import logger

from ragnar.cache.interfaces.base_sem_cache import BaseSemanticCache
from ragnar.generation.interfaces.base_generator import BaseGenerator
from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter
from ragnar.observability.metrics import MetricsCollector
from ragnar.observability.metrics import QueryMetrics
from ragnar.observability.tracker import QueryTracker
from ragnar.prompts.prompt_loader import PromptLoader
from ragnar.rerank.interfaces.base_cross_encoder import BaseCrossEncoder
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever
from ragnar.session.interfaces.session_store import BaseSessionStore
from ragnar.session.session import Session


class BaseQueryPipeline(ABC):
    """Base query pipeline with common logic"""

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        prompt_loader: PromptLoader,
        retriever: BaseRetriever,
        reranker: BaseCrossEncoder,
        generator: BaseGenerator,
        session_store: BaseSessionStore | None = None,
        semantic_cache: BaseSemanticCache | None = None,
    ):
        # Accept interfaces, not implementations
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.session_store = session_store
        self.semantic_cache = semantic_cache
        self.metrics_collector = MetricsCollector()

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    async def _retrieve(self, query: str) -> list[dict]:
        """Retrieve results based on strategy"""

    @staticmethod
    def _normalize_answer(generated_answer: str | dict[str, Any]) -> str:
        """Normalize plain or structured generator output to answer text."""
        if isinstance(generated_answer, str):
            return generated_answer

        if isinstance(generated_answer, dict):
            answer = generated_answer.get('answer')
            if isinstance(answer, str):
                return answer

        raise TypeError(
            'Generator must return a string or a dict containing a string '
            "'answer' field",
        )

    async def query(self, user_query: str, session_id: str) -> dict:
        """Execute full query pipeline"""

        # Start Tracker
        tracker = QueryTracker()
        tracker.start()

        session_store = self.session_store
        if session_store is None:
            raise RuntimeError('Query pipeline requires a session store')

        # Get or create session FIRST
        session = await session_store.get(session_id)
        if not session:
            session = Session(session_id=session_id)
            logger.info(f"Created new session: {session_id}")

        # Only cache if first message (no history)
        should_cache = len(session.messages) == 0

        if should_cache and self.semantic_cache:
            strategy_key = self.__class__.__name__.replace(
                'QueryPipeline', '',
            ).lower()
            cached = await self.semantic_cache.get(
                query=user_query,
                strategy=strategy_key,
            )
            if cached:
                cache_score = cached.pop('cache_score')
                metrics = QueryMetrics(
                    session_id=session_id,
                    query=user_query,
                    strategy=strategy_key,
                    total_duration_s=tracker.end(),
                    steps=[],
                    answer_length=len(cached.get('answer', '')),
                    num_sources=len(cached.get('sources', [])),
                    cache_hit=True,
                    cache_score=cache_score,
                )
                self.metrics_collector.record_query(metrics)

                logger.info(
                    f"Semantic cache HIT | query: {user_query[:50]}... | "
                    f"score: {cache_score:.3f}",
                )

                cached['session_id'] = session_id
                cached['conversation_length'] = 0
                cached['metrics'] = asdict(metrics)
                return cached

            logger.info(f"Semantic cache MISS | query: {user_query[:50]}...")

            logger.info(
                f"Pipeline start | session: {session_id} | "
                f"query: {user_query[:50]}...",
            )

        try:
            session.add_message('user', user_query)

            # 1. Retrieve (strategy-specific)
            async with tracker.step('retrieval'):
                logger.info('Step 1/4: Retrieving')
                all_results = await self._retrieve(user_query)
                logger.info(f"Retrieved {len(all_results)} results")

            # 2. Rerank
            async with tracker.step('reranking'):
                logger.info('Step 2/4: Reranking')
                reranked = await self.reranker.score(
                    query=user_query,
                    results=all_results,
                    top_k=5,
                )
                logger.info(f"Top-{len(reranked)} reranked")

            # 3. Generate
            async with tracker.step('generation'):
                logger.info('Step 3/4: Generating answer')
                generated_answer = await self.generator.generate(
                    query=user_query,
                    context=reranked,
                    prompt_template='simple_qa',
                )
                answer = self._normalize_answer(generated_answer)
                logger.info(f"Generated ({len(answer)} chars)")

            # Store in session
            session.add_message(
                'assistant',
                answer,
                sources=reranked,
            )
            await session_store.save(session)

            # Record metrics (full pipeline)
            total_s = tracker.end()
            strategy_key = self.__class__.__name__.replace(
                'QueryPipeline', '',
            ).lower()
            metrics = QueryMetrics(
                session_id=session_id,
                query=user_query,
                strategy=strategy_key,
                total_duration_s=total_s,
                steps=tracker.steps,
                answer_length=len(answer),
                num_sources=len(reranked),
                cache_hit=False,
                cache_score=None,
            )
            self.metrics_collector.record_query(metrics)

            result = {
                'answer': answer,
                'sources': reranked,
                'session_id': session_id,
                'conversation_length': len(session.messages),
                'metrics': asdict(metrics),
            }

            # Only cache initial questions
            if should_cache and self.semantic_cache:
                await self.semantic_cache.set(
                    query=user_query,
                    strategy=strategy_key,
                    response=result,
                )

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
