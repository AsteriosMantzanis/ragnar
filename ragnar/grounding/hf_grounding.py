from __future__ import annotations

import os
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from ragnar.grounding.interfaces.base_grounder import BaseGrounding


class HF_Grounding(BaseGrounding):
    def __init__(self):
        self.model_name = os.getenv(
            'grounding_model', 'cross-encoder/nli-deberta-v3-xsmall',
        )
        self.model = CrossEncoder(self.model_name)
        logger.info(f"Initialized HF_Grounding with model: {self.model_name}")

    async def ground(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        logger.info(
            f"Starting grounding for answer with {len(sources)} sources",
        )

        # Handle empty sources
        if not sources:
            logger.warning('No sources available for grounding')
            return [{
                'claim': answer,
                'status': 'unsupported',
                'grounded': False,
                'entailment_score': 0.0,
                'contradiction_score': 0.0,
                'evidence': None,
            }]

        claims = [s.strip() for s in answer.split('.') if s.strip()]
        logger.debug(f"Extracted {len(claims)} claims from answer")

        results = []

        for i, claim in enumerate(claims):
            logger.debug(
                f"Grounding claim {i+1}/{len(claims)}: {claim[:50]}...",
            )

            pairs = [(source['text'], claim) for source in sources]
            scores = self.model.predict(pairs)
            probabilities = self._softmax(scores)

            best_entailment_index = int(np.argmax(probabilities[:, 1]))
            best_entailment_probability = float(
                probabilities[best_entailment_index, 1],
            )
            best_source = sources[best_entailment_index]

            best_contradiction_index = int(np.argmax(probabilities[:, 0]))
            best_contradiction_probability = float(
                probabilities[best_contradiction_index, 0],
            )

            grounded = best_entailment_probability >= 0.80
            status = 'grounded' if grounded else (
                'contradicted' if
                best_contradiction_probability >= 0.80 else 'unsupported'
            )

            result = {
                'claim': claim,
                'status': status,
                'grounded': grounded,
                'entailment_score': best_entailment_probability,
                'contradiction_score': best_contradiction_probability,
                'evidence': best_source,
            }
            results.append(result)

            logger.debug(
                f"Claim {i+1} | Status: {status} | "
                f"Entailment: {best_entailment_probability:.3f} | "
                f"Contradiction: {best_contradiction_probability:.3f}",
            )

        grounded_count = sum(1 for r in results if r['grounded'])
        logger.info(
            f"Grounding complete: {grounded_count}/{len(results)} "
            f"claims grounded",
        )

        return results

    async def ground_hierarchical(
        self,
        answer: str,
        sources: dict[str, list[dict[str, Any]]],
        grounding_entity: str,
    ) -> list[dict[str, Any]]:
        logger.info(
            f"Starting hierarchical grounding for entity: {grounding_entity}",
        )

        # Handle empty sources
        source_dicts = sources[grounding_entity]

        if not source_dicts:
            logger.warning(
                f"No {grounding_entity} sources available for grounding",
            )
            return [{
                'claim': answer,
                'status': 'unsupported',
                'grounded': False,
                'entailment_score': 0.0,
                'contradiction_score': 0.0,
                'evidence': None,
            }]

        claims = [s.strip() for s in answer.split('.') if s.strip()]
        source_dicts = sources[grounding_entity]

        logger.debug(
            f"Extracted {len(claims)} claims, "
            f"{len(source_dicts)} {grounding_entity} sources",
        )

        results = []

        for i, claim in enumerate(claims):
            logger.debug(
                f"Grounding claim {i+1}/{len(claims)}: {claim[:50]}...",
            )

            pairs = [(source['text'], claim) for source in source_dicts]
            scores = self.model.predict(pairs)
            probabilities = self._softmax(scores)

            best_entailment_index = int(np.argmax(probabilities[:, 1]))
            best_source = source_dicts[best_entailment_index]
            best_entailment_probability = float(
                probabilities[best_entailment_index, 1],
            )

            best_contradiction_index = int(np.argmax(probabilities[:, 0]))
            best_contradiction_probability = float(
                probabilities[best_contradiction_index, 0],
            )

            grounded = best_entailment_probability >= 0.80
            status = (
                'grounded'
                if grounded
                else (
                    'contradicted'
                    if best_contradiction_probability >= 0.80
                    else 'unsupported'
                )
            )

            result = {
                'claim': claim,
                'status': status,
                'grounded': grounded,
                'entailment_score': best_entailment_probability,
                'contradiction_score': best_contradiction_probability,
                'evidence': best_source,
            }
            results.append(result)

            logger.debug(
                f"Claim {i+1} | Status: {status} | "
                f"Entailment: {best_entailment_probability:.3f} | "
                f"Contradiction: {best_contradiction_probability:.3f}",
            )

        grounded_count = sum(1 for r in results if r['grounded'])
        logger.info(
            f"Hierarchical grounding complete: "
            f"{grounded_count}/{len(results)} claims grounded",
        )

        return results

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        exp_scores = np.exp(
            scores - np.max(scores, axis=1, keepdims=True),
        )
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
