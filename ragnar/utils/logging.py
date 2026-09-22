from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = 'INFO', *, color: bool = True) -> None:
    """Configure a clean console logger for the app.

    This keeps the output readable in development while still allowing
    debug-level tracing when needed.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=color,
        format=(
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{module}</cyan>:<cyan>{function}</cyan>:'
            '<cyan>{line}</cyan> - <level>{message}</level>'
        ),
        enqueue=True,
    )
