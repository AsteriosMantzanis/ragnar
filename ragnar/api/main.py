from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from loguru import logger

from ragnar.api.dependencies import get_index_pipeline
from ragnar.api.dependencies import get_query_pipeline
from ragnar.api.dependencies import get_session_store
from ragnar.api.models import IndexRequest
from ragnar.api.models import IndexResponse
from ragnar.api.models import MessageModel
from ragnar.api.models import QueryRequest
from ragnar.api.models import QueryResponse
from ragnar.api.models import SessionResponse

# Initialize FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown"""
    # Startup
    logger.info('Ragnar API starting up...')
    session_store = get_session_store()

    if hasattr(session_store, 'connect'):
        await session_store.connect()
        logger.info('Session store connected')

    yield

    # Shutdown
    logger.info('Ragnar API shutting down...')
    if hasattr(session_store, 'disconnect'):
        await session_store.disconnect()
        logger.info('Session store disconnected')

app = FastAPI(
    title='Ragnar RAG API',
    description='Production RAG framework with query & indexing pipelines',
    version='0.1.0',
    docs_url='/docs',
    redoc_url='/redoc',
    lifespan=lifespan,
)

logger.info('Ragnar API starting up...')


# Endpoints
@app.post('/query', response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Execute query against RAG system"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(
            f"Query request | session: {session_id} | "
            f"strategy: {request.strategy}",
        )

        # Get pipeline based on strategy
        query_pipeline = get_query_pipeline(request.strategy)

        # Execute query
        result = await query_pipeline.query(
            user_query=request.query,
            session_id=session_id,
        )

        logger.info(
            f"Query completed | session: {session_id} | "
            f"grounded: {result['grounded_percentage']:.0f}%",
        )

        return QueryResponse(
            answer=result['answer'],
            grounding=result['grounding'],
            grounded_percentage=result['grounded_percentage'],
            sources=result['sources'],
            session_id=result['session_id'],
            conversation_length=result['conversation_length'],
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/index', response_model=IndexResponse)
async def index_endpoint(request: IndexRequest):
    """Index documents"""
    try:
        path = Path(request.path)
        logger.info(
            f"Index request | path: {path} | strategy: {request.strategy}",
        )

        # Validate path exists
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Get pipeline based on strategy
        index_pipeline = get_index_pipeline(request.strategy)

        # Execute indexing
        result = await index_pipeline.index(path)

        logger.info(
            f"Indexing completed | documents: {result['documents_indexed']} | "
            f"strategy: {request.strategy}",
        )

        return IndexResponse(
            status=result['status'],
            documents_indexed=result.get('documents_indexed', 0),
            chunks=result.get('chunks'),
            parent_chunks=result.get('parent_chunks'),
            child_chunks=result.get('child_chunks'),
        )

    except FileNotFoundError as e:
        logger.error(f"Index failed - file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(f"Index failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/sessions/{session_id}', response_model=SessionResponse)
async def get_session(session_id: str):
    """Retrieve session history"""
    try:
        logger.info(f"Session retrieval | session: {session_id}")

        session_store = get_session_store()
        session = await session_store.get(session_id)

        if not session:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(status_code=404, detail='Session not found')

        messages = [
            MessageModel(
                role=m.role,
                content=m.content,
                timestamp=m.timestamp.isoformat(),
            )
            for m in session.messages
        ]

        logger.info(
            f"Session retrieved | session: "
            f"{session_id} | messages: {len(messages)}",
        )

        return SessionResponse(session_id=session_id, messages=messages)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Session retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/sessions/{session_id}')
async def delete_session(session_id: str):
    """Delete session"""
    try:
        logger.info(f"Session deletion | session: {session_id}")

        session_store = get_session_store()
        await session_store.delete(session_id)

        logger.info(f"Session deleted | session: {session_id}")

        return {'status': 'deleted', 'session_id': session_id}

    except Exception as e:
        logger.error(f"Session deletion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Run

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level='info',
    )
