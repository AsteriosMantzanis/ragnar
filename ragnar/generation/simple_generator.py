from __future__ import annotations

import json

from loguru import logger

from ragnar.api.models import QueryResponse
from ragnar.generation.interfaces.base_generator import BaseGenerator
from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter
from ragnar.prompts.prompt_loader import PromptLoader


class Generator(BaseGenerator):
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        prompt_loader: PromptLoader,
    ):
        self.llm = llm_adapter
        self.prompt_loader = prompt_loader

    async def generate(
        self,
        query: str,
        context: list[dict],
        prompt_template: str = 'simple_qa',
        response_schema: dict | None = None,
    ) -> str | dict:
        logger.info(f"Generating answer from {len(context)} context items")

        if response_schema is None:
            response_schema = QueryResponse.model_json_schema()

        context_text = '\n\n'.join([c['text'] for c in context])
        prompt = self.prompt_loader.load(
            prompt_template, context=context_text, query=query,
        )
        result = await self.llm.generate(prompt, format=response_schema)

        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                logger.warning(
                    'Structured generation returned non-JSON content',
                )
                return {'answer': result}

        return result

    async def generate_hierarchical(
        self,
        query: str,
        context: dict[str, list[dict]],
        generation_entity: str,
        prompt_template: str = 'simple_qa',
        response_schema: dict | None = None,
    ) -> str | dict:
        logger.info(
            f"Generating hierarchical answer, "
            f"primary entity: {generation_entity}",
        )

        if response_schema is None:
            response_schema = QueryResponse.model_json_schema()

        # Extract text from all entities
        all_texts = []
        for entity_name, items in context.items():
            logger.debug(f"Adding {len(items)} items from {entity_name}")
            all_texts.extend([item['text'] for item in items])

        context_text = '\n\n'.join(all_texts)
        logger.debug(
            f"Combined context: {len(all_texts)} items, "
            f"{len(context_text)} chars",
        )

        prompt = self.prompt_loader.load(
            prompt_template, context=context_text, query=query,
        )
        result = await self.llm.generate(prompt, format=response_schema)

        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                logger.warning(
                    'Structured generation returned non-JSON content',
                )
                return {'answer': result}

        return result
