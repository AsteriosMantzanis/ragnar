"""Small in-memory Loguru sink used by the Streamlit UI.

The API and Streamlit app run in separate processes, so the UI cannot attach
directly to Loguru. The API exposes this buffer through /logs.
"""
from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

from loguru import logger

_MAX_LOGS = 2000

_lock = Lock()
_logs: deque[dict[str, Any]] = deque(maxlen=_MAX_LOGS)
_next_id = 0
_LOG_BUFFER_CONFIGURED = False


def _sink(message: Any) -> None:
    global _next_id

    record = message.record
    entry = {
        'id': _next_id,
        'time': record['time'].strftime('%H:%M:%S.%f')[:-3],
        'level': record['level'].name,
        'message': record['message'],
        'module': record['name'].split('.')[-1],
    }

    with _lock:
        _logs.append(entry)
        _next_id += 1


def configure_log_buffer(level: str = 'INFO') -> None:
    """Install the UI log sink once.

    Keep the UI stream focused on production-relevant messages and avoid
    flooding the frontend with noisy debug output from retrievers/rerankers.
    """
    global _LOG_BUFFER_CONFIGURED

    if not _LOG_BUFFER_CONFIGURED:
        logger.add(_sink, level=level, enqueue=False)
        _LOG_BUFFER_CONFIGURED = True


def latest_cursor() -> int:
    with _lock:
        return _next_id


def get_logs(after: int = -1, limit: int = 200) -> list[dict[str, Any]]:
    with _lock:
        return [
            entry
            for entry in _logs
            if entry['id'] > after
        ][-limit:]
