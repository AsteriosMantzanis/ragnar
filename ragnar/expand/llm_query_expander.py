from __future__ import annotations

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

    async def expand(self, query: str) -> list[str]:
        # Get multi-query variations
        prompt = self.prompt_loader.load('expand_multiquery', query=query)
        response = await self.llm.generate(prompt)
        queries = self._parse_queries(response)

        # Get step-back query
        step_back_prompt = self.prompt_loader.load(
            'expand_step_back', query=query,
        )
        step_back = await self.llm.generate(step_back_prompt)

        return [query] + queries + [step_back]

    def _parse_queries(self, response: str) -> list[str]:
        # Parse LLM output into list of queries
        return [line.strip() for line in response.split('\n') if line.strip()]
