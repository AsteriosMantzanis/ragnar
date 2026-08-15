from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from loguru import logger

from ragnar.loaders.interfaces.base_loader import BaseLoader


class BaseIndexPipeline(ABC):
    """Base indexing pipeline"""

    def __init__(self, loader: BaseLoader):
        self.loader = loader
        logger.info(
            f"Initialized {self.__class__.__name__}"
            f" with {type(loader).__name__}",
        )

    @abstractmethod
    async def index(self, path: Path) -> dict:
        """Index file or directory"""
