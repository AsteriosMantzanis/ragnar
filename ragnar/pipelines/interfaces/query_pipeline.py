from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import asdict

from loguru import logger

from ragnar.expand.interfaces.base_expander import BaseQueryExpander
from ragnar.generation.interfaces.base_generator import BaseGenerator
from ragnar.grounding.interfaces.base_grounder import BaseGrounding
from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter
from ragnar.observability.metrics import MetricsCollector
from ragnar.observability.metrics import QueryMetrics
from ragnar.observability.tracker import QueryTracker
from ragnar.prompts.prompt_loader import PromptLoader
from ragnar.rerank.interfaces.base_cross_encoder import BaseCrossEncoder
from ragnar.retrieval.interfaces.base_retriever import BaseRetriever
from ragnar.rewrite.interfaces.base_rewritter import BaseQueryRewriter
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
        rewriter: BaseQueryRewriter,
        expander: BaseQueryExpander,
        session_store: BaseSessionStore | None = None,
    ):
        # Accept interfaces, not implementations
        self.retriever = retriever
        self.rewriter = rewriter
        self.expander = expander
        self.reranker = reranker
        self.generator = generator
        self.grounder = grounder
        self.session_store = session_store or InMemorySessionStore()
        self.metrics_collector = MetricsCollector()

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    async def _retrieve(self, expanded_queries: list[str]) -> list[dict]:
        """Retrieve results based on strategy"""

    async def query(self, user_query: str, session_id: str) -> dict:
        """Execute full query pipeline"""
        # Start Tracker
        tracker = QueryTracker()
        tracker.start()

        logger.info(
            f"Pipeline start | session: {session_id} | "
            f"query: {user_query[:50]}...",
        )

        try:
            # Get or create session
            session = await self.session_store.get(session_id)
            if not session:
                session = Session(session_id=session_id)
                logger.info(f"Created new session: {session_id}")

            # Add conversation context to query
            history = session.get_history(max_messages=4)  # Last 2 exchanges
            enriched_query = f"{history}\n\nNew question: {user_query}"

            session.add_message('user', user_query)

            # 1. Rewrite
            async with tracker.step('rewrite'):
                logger.info('Step 1/7: Rewriting query')
                rewritten = await self.rewriter.rewrite(enriched_query)
                logger.debug(f"Rewritten: {rewritten[:60]}...")

            # 2. Expand
            async with tracker.step('expand'):
                logger.info('Step 2/7: Expanding query')
                expanded_queries = await self.expander.expand(rewritten)
                logger.info(f"Expanded to {len(expanded_queries)} queries")

            # 3. Retrieve (strategy-specific)
            async with tracker.step('retrieval'):
                logger.info('Step 3/7: Retrieving')
                all_results = await self._retrieve(expanded_queries)
                logger.info(f"Retrieved {len(all_results)} results")

            # 4. Deduplicate
            async with tracker.step('deduplication'):
                logger.info('Step 4/7: Deduplicating')
                unique_results = {}
                for r in all_results:
                    chunk_id = r.get('chunk_id')
                    if chunk_id not in unique_results:
                        unique_results[chunk_id] = r
                logger.info(f"Deduplicated to {len(unique_results)}")

            # 5. Rerank
            async with tracker.step('reranking'):
                logger.info('Step 5/7: Reranking')
                reranked = await self.reranker.score(
                    query=user_query,
                    results=list(unique_results.values()),
                    top_k=5,
                )
                logger.info(f"Top-{len(reranked)} reranked")

            # 6. Generate
            async with tracker.step('generation'):
                logger.info('Step 6/7: Generating answer')
                answer = await self.generator.generate(
                    query=user_query,
                    context=reranked,
                    prompt_template='simple_qa',
                )
                logger.info(f"Generated ({len(answer)} chars)")

            # 7. Ground
            async with tracker.step('grounding'):
                logger.info('Step 7/7: Grounding answer')
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

            # Record metrics
            total_s = tracker.end()
            metrics = QueryMetrics(
                session_id=session_id,
                query=user_query,
                strategy=self.__class__.__name__,
                total_duration_s=total_s,
                steps=tracker.steps,
                answer_length=len(answer),
                num_sources=len(reranked),
                grounding_percentage=grounded_pct,
            )
            self.metrics_collector.record_query(metrics)

            logger.info('Pipeline complete')

            return {
                'answer': answer,
                'grounding': grounding,
                'grounded_percentage': grounded_pct,
                'sources': reranked,
                'session_id': session_id,
                'conversation_length': len(session.messages),
                'metrics': asdict(metrics),
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
