from __future__ import annotations


class CollectionNotFoundError(Exception):
    """Raised when a Qdrant collection doesn't exist.

    In practice this almost always means one thing: no documents have
    been indexed yet for this strategy. Kept as its own exception type
    rather than a generic one so callers (the API layer) can tell this
    apart from a genuine Qdrant/network failure and respond accordingly,
    instead of surfacing a raw 500 with an unhelpful qdrant-client error
    string.
    """
