from __future__ import annotations

from loguru import logger

from ragnar.expand.interfaces.base_expander import BaseQueryExpander
from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter
from ragnar.prompts.prompt_loader import PromptLoader


class LLMQueryExpander(BaseQueryExpander):
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        prompt_loader: PromptLoader,
    ):
        self.llm = llm_adapter
        self.prompt_loader = prompt_loader
        logger.info('Initialized LLMQueryExpander')

    async def expand(self, query: str) -> list[str]:
        logger.info(f"Expanding query: {query[:50]}...")

        prompt = self.prompt_loader.load('expand_multiquery', query=query)
        response = await self.llm.generate(prompt)
        queries = self._parse_queries(response)
        logger.debug(f"Generated {len(queries)} alternative queries")

        step_back_prompt = self.prompt_loader.load(
            'expand_step_back', query=query,
        )
        step_back = await self.llm.generate(step_back_prompt)
        logger.debug(f"Generated step-back query: {step_back[:50]}...")

        all_queries = [query] + queries + [step_back]
        logger.info(f"Total expanded queries: {len(all_queries)}")
        return all_queries

    def _parse_queries(self, response: str) -> list[str]:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        logger.debug(f"Parsed {len(lines)} queries from LLM response")
        return lines
