from __future__ import annotations

import os

from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder
from loguru import logger

from ragnar.rerank.interfaces.base_cross_encoder import BaseCrossEncoder

load_dotenv()


class FastEmbedReranker(BaseCrossEncoder):
    def __init__(self, batch_size: int = 5):
        self.model_name = os.getenv(
            'rerank_model', 'jinaai/jina-reranker-v1-turbo-en',
        )
        self.model = TextCrossEncoder(self.model_name)
        self.batch_size = batch_size
        logger.info(
            f"Initialized FastEmbedReranker with model: {self.model_name} | "
            f"batch_size: {batch_size}",
        )

    async def score(
        self, query: str, results: list[dict], top_k: int = 5,
    ) -> list[dict]:
        logger.info(f"Starting reranking for {len(results)} results")
        logger.debug(f"Query: {query[:50]}...")

        texts = [r['text'] for r in results]

        # Batch scoring to avoid OOM
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_scores = self.model.rerank(query, batch_texts)
            all_scores.extend(batch_scores)
            logger.debug(f"Scored batch {i // self.batch_size + 1}")

        for i, (result, score) in enumerate(zip(results, all_scores)):
            result['rerank_score'] = float(score)
            logger.debug(f"Result {i+1}: score={result['rerank_score']:.3f}")

        ranked = sorted(
            results, key=lambda x: x['rerank_score'], reverse=True,
        )
        top_results = ranked[:top_k]

        logger.info(
            f"Reranking complete: returning top"
            f" {len(top_results)}/{len(results)} results",
        )

        return top_results

    async def score_hierarchical(
        self, query: str, results: dict,
        child_entity: str,
        top_k: int = 5,
    ) -> dict:
        logger.info(
            f"Starting hierarchical reranking for entity: {child_entity}",
        )
        logger.debug(f"Query: {query[:50]}...")

        child_results = results[child_entity]
        logger.debug(f"Reranking {len(child_results)} {child_entity}")

        texts = [item['text'] for item in child_results]

        # Batch scoring to avoid OOM
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_scores = self.model.rerank(query, batch_texts)
            all_scores.extend(batch_scores)
            logger.debug(f"Scored batch {i // self.batch_size + 1}")

        for i, (item, score) in enumerate(zip(child_results, all_scores)):
            item['rerank_score'] = float(score)
            logger.debug(
                f"{child_entity} {i+1}: score={item['rerank_score']:.3f}",
            )

        ranked = sorted(
            child_results,
            key=lambda x: x['rerank_score'],
            reverse=True,
        )
        results[child_entity] = ranked[:top_k]

        logger.info(
            f"Hierarchical reranking complete: "
            f"returning top {len(results[child_entity])}/{len(child_results)}",
        )

        return results
