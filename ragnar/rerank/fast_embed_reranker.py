from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragnar.rerank.interfaces.base_cross_encoder import BaseCrossEncoder
load_dotenv()


class FastEmbedReranker(BaseCrossEncoder):
    def __init__(self):
        self.model_name = os.getenv(
            'rerank_model', 'jinaai/jina-reranker-v2-base-multilingual',
        )
        self.model = TextCrossEncoder(self.model_name)

    async def score(self, query: str, results: list[dict]) -> list[dict]:
        texts = [(query, i['text']) for i in results]
        scores = self.model.rerank(query=query, documents=texts)
        for result, score in zip(results, scores):
            result['rerank_score'] = score
        return sorted(results, key=lambda x: x['rerank_score'], reverse=True)

    async def score_hierarchical(
        self, query: str,
        results: dict[str, Any],
        child_entity: str,
    ) -> dict[str, Any]:

        texts = [i['text'] for i in results[child_entity]]
        scores = self.model.rerank(query=query, documents=texts)

        for result, score in zip(results[child_entity], scores):
            result['rerank_score'] = score

        results[child_entity] = sorted(
            results[child_entity],
            key=lambda x: x['rerank_score'],
            reverse=True,
        )

        return results
