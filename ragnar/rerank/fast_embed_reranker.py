from __future__ import annotations

import os

from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder
from loguru import logger

from ragnar.rerank.interfaces.base_cross_encoder import BaseCrossEncoder

load_dotenv()


class FastEmbedReranker(BaseCrossEncoder):
    def __init__(self, top_k: int = 5):
        self.model_name = os.getenv(
            'rerank_model', 'jinaai/jina-reranker-v2-base-multilingual',
        )
        self.model = TextCrossEncoder(self.model_name)
        logger.info(
            f"Initialized FastEmbedReranker with model: {self.model_name}",
        )
        self.top_k = top_k

    async def score(
        self, query: str, results: list[dict],
    ) -> list[dict]:
        logger.info(f"Starting reranking for {len(results)} results")
        logger.debug(f"Query: {query[:50]}...")

        texts = [r['text'] for r in results]
        scores = self.model.rerank(query, texts)

        for i, (result, score) in enumerate(zip(results, scores)):
            result['rerank_score'] = float(score)
            logger.debug(f"Result {i+1}: score={result['rerank_score']:.3f}")

        ranked = sorted(
            results, key=lambda x: x['rerank_score'], reverse=True,
        )
        top_results = ranked[:self.top_k]

        logger.info(
            f"Reranking complete: returning top"
            f" {len(top_results)}/{len(results)} results",
        )

        return top_results

    async def score_hierarchical(
        self, query: str, results: dict,
        child_entity: str = 'sections',
    ) -> dict:
        logger.info(
            f"Starting hierarchical reranking for entity: {child_entity}",
        )
        logger.debug(f"Query: {query[:50]}...")

        child_results = results[child_entity]
        logger.debug(f"Reranking {len(child_results)} {child_entity}")

        texts = [item['text'] for item in child_results]
        scores = self.model.rerank(query, texts)

        for i, (item, score) in enumerate(zip(child_results, scores)):
            item['rerank_score'] = float(score)
            logger.debug(
                f"{child_entity} {i+1}: score={item['rerank_score']:.3f}",
            )

        ranked = sorted(
            child_results,
            key=lambda x: x['rerank_score'],
            reverse=True,
        )
        results[child_entity] = ranked[:self.top_k]

        logger.info(
            f"Hierarchical reranking complete: "
            f"returning top {len(results[child_entity])}/{len(child_results)}",
        )

        return results
