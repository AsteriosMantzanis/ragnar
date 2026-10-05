from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from ragnar.grounding.interfaces.base_grounder import BaseGrounding


class HF_Grounding(BaseGrounding):
    """Ground generated claims against retrieved context with an HF NLI model.

    """

    DEFAULT_MODEL = 'cross-encoder/nli-deberta-v3-xsmall'
    DEFAULT_ENTAILMENT_THRESHOLD = 0.80
    DEFAULT_CONTRADICTION_THRESHOLD = 0.80
    DEFAULT_BATCH_SIZE = 32

    _SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')
    _REQUIRED_LABELS = ('contradiction', 'entailment', 'neutral')

    def __init__(self) -> None:
        self.model_name = os.getenv('grounding_model', self.DEFAULT_MODEL)
        self.entailment_threshold = float(
            os.getenv(
                'grounding_entailment_threshold',
                str(self.DEFAULT_ENTAILMENT_THRESHOLD),
            ),
        )
        self.contradiction_threshold = float(
            os.getenv(
                'grounding_contradiction_threshold',
                str(self.DEFAULT_CONTRADICTION_THRESHOLD),
            ),
        )
        self.batch_size = int(
            os.getenv(
                'grounding_batch_size',
                str(self.DEFAULT_BATCH_SIZE),
            ),
        )

        if not 0.0 <= self.entailment_threshold <= 1.0:
            raise ValueError(
                'grounding_entailment_threshold must be between 0 and 1',
            )
        if not 0.0 <= self.contradiction_threshold <= 1.0:
            raise ValueError(
                'grounding_contradiction_threshold must be between 0 and 1',
            )
        if self.batch_size <= 0:
            raise ValueError('grounding_batch_size must be positive')

        self.model = CrossEncoder(self.model_name)
        self.label_indices = self._resolve_label_indices()

        logger.info(
            f'Initialized HF_Grounding with model: {self.model_name} | '
            f'labels: {self.label_indices} | '
            f'entailment_threshold: {self.entailment_threshold:.2f} | '
            f'contradiction_threshold: {self.contradiction_threshold:.2f}',
        )

    async def ground(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        logger.info(
            f'Starting grounding for answer with {len(sources)} sources',
        )
        return self._ground_sources(answer, sources)

    async def ground_hierarchical(
        self,
        answer: str,
        sources: dict[str, list[dict[str, Any]]],
        grounding_entity: str,
    ) -> list[dict[str, Any]]:
        logger.info(
            f'Starting hierarchical grounding for entity: '
            f'{grounding_entity}',
        )

        source_dicts = sources.get(grounding_entity, [])
        logger.debug(
            f'Using {len(source_dicts)} {grounding_entity} sources',
        )

        return self._ground_sources(answer, source_dicts)

    def _ground_sources(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        claims = self._extract_claims(answer)
        logger.debug(f'Extracted {len(claims)} claims from answer')

        if not claims:
            return []

        if not sources:
            logger.warning('No sources available for grounding')
            return [self._unsupported_result(claim) for claim in claims]

        # Evaluate every claim/source pair in one batched inference call.
        pairs = [
            (source['text'], claim)
            for claim in claims
            for source in sources
        ]

        probabilities = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            apply_softmax=True,
        )
        probabilities = np.asarray(probabilities, dtype=np.float32)

        expected_shape = (len(claims), len(sources), 3)
        if probabilities.size != np.prod(expected_shape):
            raise ValueError(
                f'Expected 3-class NLI output for {len(pairs)} pairs, '
                f'but received shape {probabilities.shape} from '
                f'{self.model_name!r}',
            )

        probabilities = probabilities.reshape(expected_shape)

        contradiction_idx = self.label_indices['contradiction']
        entailment_idx = self.label_indices['entailment']

        results: list[dict[str, Any]] = []

        for claim_index, claim in enumerate(claims):
            claim_probabilities = probabilities[claim_index]

            entailment_scores = claim_probabilities[:, entailment_idx]
            contradiction_scores = claim_probabilities[:, contradiction_idx]

            best_entailment_index = int(np.argmax(entailment_scores))
            best_contradiction_index = int(np.argmax(contradiction_scores))

            support_scores = self._scores_to_dict(
                claim_probabilities[best_entailment_index],
            )
            contradiction_evidence_scores = self._scores_to_dict(
                claim_probabilities[best_contradiction_index],
            )

            best_entailment_score = support_scores['entailment']
            best_contradiction_score = contradiction_evidence_scores[
                'contradiction'
            ]

            grounded = (
                best_entailment_score >= self.entailment_threshold
            )
            status = (
                'grounded'
                if grounded
                else (
                    'contradicted'
                    if best_contradiction_score
                    >= self.contradiction_threshold
                    else 'unsupported'
                )
            )

            supporting_evidence = {
                'source': sources[best_entailment_index],
                'scores': support_scores,
            }
            contradicting_evidence = {
                'source': sources[best_contradiction_index],
                'scores': contradiction_evidence_scores,
            }

            # Keep `evidence` backward-compatible as the raw source dict.
            evidence = (
                supporting_evidence['source']
                if status == 'grounded'
                else contradicting_evidence['source']
                if status == 'contradicted'
                else None
            )

            result = {
                'claim': claim,
                'status': status,
                'grounded': grounded,
                # These scores are one coherent NLI distribution from the
                # strongest supporting source and therefore sum to 1.0.
                'scores': support_scores,
                'entailment_score': best_entailment_score,
                'contradiction_score': best_contradiction_score,
                'evidence': evidence,
                'supporting_evidence': supporting_evidence,
                'contradicting_evidence': contradicting_evidence,
            }
            results.append(result)

            logger.debug(
                f'Claim {claim_index + 1}/{len(claims)} | '
                f'Status: {status} | '
                f'Entailment: {best_entailment_score:.3f} | '
                f'Neutral: {support_scores["neutral"]:.3f} | '
                f'Contradiction: {best_contradiction_score:.3f}',
            )

            if (
                grounded
                and best_contradiction_score
                >= self.contradiction_threshold
            ):
                logger.warning(
                    f'Claim has both strong supporting and contradicting '
                    f'evidence: {claim[:100]}...',
                )

        grounded_count = sum(1 for result in results if result['grounded'])
        logger.info(
            f'Grounding complete: {grounded_count}/{len(results)} '
            f'claims grounded',
        )

        return results

    def _resolve_label_indices(self) -> dict[str, int]:
        """Resolve NLI label indices from the loaded HF model config."""
        if self.model.num_labels != 3:
            raise ValueError(
                f'Grounding model {self.model_name!r} must expose exactly '
                f'3 labels (contradiction, entailment, neutral), but '
                f'num_labels={self.model.num_labels}',
            )

        config = self.model.model.config
        label2id = getattr(config, 'label2id', None) or {}

        normalized = {
            str(label).strip().lower(): int(index)
            for label, index in label2id.items()
        }

        missing = [
            label
            for label in self._REQUIRED_LABELS
            if label not in normalized
        ]
        if missing:
            raise ValueError(
                f'Grounding model {self.model_name!r} does not expose the '
                f'required NLI labels in config.label2id. Missing: '
                f'{missing}. Found: {sorted(normalized)}',
            )

        indices = {
            label: normalized[label]
            for label in self._REQUIRED_LABELS
        }

        if sorted(indices.values()) != [0, 1, 2]:
            raise ValueError(
                f'Invalid NLI label mapping for {self.model_name!r}: '
                f'{indices}',
            )

        return indices

    @staticmethod
    def _extract_claims(answer: str) -> list[str]:
        """Split an answer into sentence-level grounding units."""
        return [
            claim.strip()
            for claim in HF_Grounding._SENTENCE_BOUNDARY.split(answer.strip())
            if claim.strip()
        ]

    def _scores_to_dict(
        self,
        row: np.ndarray,
    ) -> dict[str, float]:
        return {
            'contradiction': round(
                float(row[self.label_indices['contradiction']]),
                6,
            ),
            'entailment': round(
                float(row[self.label_indices['entailment']]),
                6,
            ),
            'neutral': round(float(row[self.label_indices['neutral']]), 6),
        }

    @staticmethod
    def _unsupported_result(claim: str) -> dict[str, Any]:
        return {
            'claim': claim,
            'status': 'unsupported',
            'grounded': False,
            'scores': {
                'contradiction': 0.0,
                'entailment': 0.0,
                'neutral': 0.0,
            },
            'entailment_score': 0.0,
            'contradiction_score': 0.0,
            'evidence': None,
            'supporting_evidence': None,
            'contradicting_evidence': None,
        }
