from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader


class PromptLoader:
    def __init__(self, prompts_dir: str = 'ragnar/prompts'):
        self.prompts_dir = Path(prompts_dir)
        self.env = Environment(loader=FileSystemLoader(self.prompts_dir))

    def load(self, prompt_name: str, **kwargs) -> str:
        template = self.env.get_template(f"{prompt_name}.j2")
        return template.render(**kwargs)
