from __future__ import annotations

from loguru import logger

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
        logger.info('Initialized LLMQueryRewriter')

    async def rewrite(self, query: str) -> str:
        logger.info(f"Rewriting query: {query[:50]}...")

        prompt = self.prompt_loader.load('rewrite_query', query=query)
        rewritten = await self.llm.generate(prompt)

        logger.debug(
            f"Rewrite complete\n"
            f"  Original: {query[:60]}...\n"
            f"  Rewritten: {rewritten[:60]}...",
        )

        return rewritten
