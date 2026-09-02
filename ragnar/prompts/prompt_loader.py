from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader

_DEFAULT_PROMPTS_DIR = Path(__file__).parent


class PromptLoader:
    def __init__(self, prompts_dir: str | Path = _DEFAULT_PROMPTS_DIR):
        self.prompts_dir = Path(prompts_dir)
        self.env = Environment(loader=FileSystemLoader(self.prompts_dir))

    def load(self, prompt_name: str, **kwargs) -> str:
        template = self.env.get_template(f"{prompt_name}.j2")
        return template.render(**kwargs)
