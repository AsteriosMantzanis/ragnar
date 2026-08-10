from __future__ import annotations

from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter
from ragnar.prompts.prompt_loader import PromptLoader
from ragnar.rewrite.interfaces.base_rewritter import BaseQueryRewriter


class LLMQueryRewriter(BaseQueryRewriter):
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        prompt_loader: PromptLoader,
    ):
        self.llm = llm_adapter
        self.prompt_loader = prompt_loader

    async def rewrite(self, query: str) -> str:
        prompt = self.prompt_loader.load('rewrite_query', query=query)
        return await self.llm.generate(prompt)
