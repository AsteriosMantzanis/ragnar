from __future__ import annotations

from ragnar.llm.interfaces.llm_adapter import BaseLLMAdapter
from ragnar.prompts.prompt_loader import PromptLoader


class Generator:
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
    ) -> str:
        context_text = '\n\n'.join([c['text'] for c in context])
        prompt = self.prompt_loader.load(
            prompt_template, context=context_text, query=query,
        )
        return await self.llm.generate(prompt)

    async def generate_hierarchical(
        self,
        query: str,
        context: dict[str, list[dict]],
        generation_entity: str,
        prompt_template: str = 'simple_qa',
    ) -> str:
        context_text = '\n\n'.join([
            c['text']
            for c in context[generation_entity]
        ])
        prompt = self.prompt_loader.load(
            prompt_template, context=context_text, query=query,
        )
        return await self.llm.generate(prompt)
