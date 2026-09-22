from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import asdict

from loguru import logger

from ragnar.cache.interfaces.base_sem_cache import BaseSemanticCache
from ragnar.generation.interfaces.base_generator import BaseGenerator
from ragnar.grounding.interfaces.base_grounder import BaseGrounding
from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter
from ragnar.observability.metrics import MetricsCollector
from ragnar.observability.metrics import QueryMetrics
from ragnar.observability.tracker import QueryTracker
from ragnar.prompts.prompt_loader import PromptLoader
from ragnar.rerank.interfaces.base_cross_encoder import BaseCrossEncoder
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever
from ragnar.session.in_memory_session_store import InMemorySessionStore
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
        grounder: BaseGrounding,
        session_store: BaseSessionStore | None = None,
        semantic_cache: BaseSemanticCache | None = None,
    ):
        # Accept interfaces, not implementations
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.grounder = grounder
        self.session_store = session_store or InMemorySessionStore()
        self.semantic_cache = semantic_cache
        self.metrics_collector = MetricsCollector()

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    async def _retrieve(self, query: str) -> list[dict]:
        """Retrieve results based on strategy"""

    async def query(self, user_query: str, session_id: str) -> dict:
        """Execute full query pipeline"""

        # Start Tracker
        tracker = QueryTracker()
        tracker.start()

        # Get or create session FIRST
        session = await self.session_store.get(session_id)
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
                    grounding_percentage=cached.get('grounded_percentage', 0),
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
                logger.info('Step 1/5: Retrieving')
                all_results = await self._retrieve(user_query)
                logger.info(f"Retrieved {len(all_results)} results")

            # 2. Deduplicate
            async with tracker.step('deduplication'):
                logger.info('Step 2/5: Deduplicating')
                unique_results = {}
                for r in all_results:
                    chunk_id = r.get('chunk_id')
                    if chunk_id not in unique_results:
                        unique_results[chunk_id] = r
                logger.info(f"Deduplicated to {len(unique_results)}")

            # 3. Rerank
            async with tracker.step('reranking'):
                logger.info('Step 3/5: Reranking')
                reranked = await self.reranker.score(
                    query=user_query,
                    results=list(unique_results.values()),
                    top_k=5,
                )
                logger.info(f"Top-{len(reranked)} reranked")

            # 4. Generate
            async with tracker.step('generation'):
                logger.info('Step 4/5: Generating answer')
                generated = await self.generator.generate(
                    query=user_query,
                    context=reranked,
                    prompt_template='simple_qa',
                )

                if isinstance(generated, dict):
                    answer = str(
                        generated.get('answer')
                        or generated.get('response')
                        or '',
                    )
                else:
                    answer = str(generated)

                logger.info(f"Generated ({len(answer)} chars)")

            # 5. Ground
            async with tracker.step('grounding'):
                logger.info('Step 5/5: Grounding answer')
                grounding = await self.grounder.ground(answer, reranked)
                grounded_count = sum(1 for g in grounding if g['grounded'])
                grounded_pct = (
                    grounded_count / len(grounding)
                    * 100
                ) if grounding else 0
                logger.info(f"Grounding: {grounded_count}/{len(grounding)}")

            # Store in session
            session.add_message(
                'assistant', answer,
                sources=reranked, grounding=grounding,
            )
            await self.session_store.save(session)

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
                grounding_percentage=grounded_pct,
                cache_hit=False,
                cache_score=None,
            )
            self.metrics_collector.record_query(metrics)

            result = {
                'answer': answer,
                'grounding': grounding,
                'grounded_percentage': grounded_pct,
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
