from __future__ import annotations

import ctypes
import gc
import sys

from loguru import logger


def release_memory() -> None:
    """Hand freed memory back to the OS.

    Python and torch free tensors, but glibc keeps the pages in its arenas,
    so the container's RSS never shrinks and the OOM killer (exit 137)
    eventually fires. gc + malloc_trim returns them.
    """
    gc.collect()

    if not sys.platform.startswith('linux'):
        return

    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        logger.debug('malloc_trim unavailable')
