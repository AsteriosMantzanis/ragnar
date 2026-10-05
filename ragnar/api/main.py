from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from loguru import logger

from ragnar.api.dependencies import get_query_pipeline
from ragnar.api.dependencies import get_session_store
from ragnar.api.dependencies import preload_models
from ragnar.api.models import IndexRequest
from ragnar.api.models import IndexResponse
from ragnar.api.models import MessageModel
from ragnar.api.models import QueryRequest
from ragnar.api.models import QueryResponse
from ragnar.api.models import SessionResponse
from ragnar.indexing.job_queue import IndexJobQueue
from ragnar.observability.log_buffer import configure_log_buffer
from ragnar.observability.log_buffer import get_logs
from ragnar.observability.log_buffer import latest_cursor
from ragnar.utils.logging import configure_logging


configure_logging(level='INFO')

# Keep Loguru output in stderr as before, and additionally retain a small
# in-memory tail that the Streamlit UI can poll while a request is running.
# Keep the UI stream at INFO+ so debug-level rerank/grading noise doesn't flood
# the frontend polling endpoint.
configure_log_buffer(level='INFO')

# Initialize FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Ragnar API starting up...')

    preload_models()

    session_store = get_session_store()

    if session_store is not None and hasattr(session_store, 'connect'):
        await session_store.connect()
        logger.info('Session store connected')

    app.state.index_queue = IndexJobQueue()
    await app.state.index_queue.connect()

    yield

    logger.info('Ragnar API shutting down...')

    await app.state.index_queue.close()

    if session_store is not None and hasattr(session_store, 'disconnect'):
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


@app.get('/logs')
async def logs_endpoint(after: int = -1, limit: int = 200):
    """Return Loguru records newer than ``after``."""
    limit = max(1, min(limit, 500))
    return {
        'logs': get_logs(after=after, limit=limit),
        'cursor': latest_cursor(),
    }


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
            metrics=result['metrics'],
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def wait_for_index_job(
    queue: IndexJobQueue,
    job_id: str,
) -> IndexResponse:
    while True:
        job = await queue.get(job_id)
        if job is None:
            raise RuntimeError(f"Index job disappeared: {job_id}")

        status = job.get('status')
        if status == 'completed':
            result = job.get('result') or {}
            logger.info(
                f"Indexing completed | job_id={job_id} | "
                f"documents={result.get('documents_indexed', 0)}",
            )
            return IndexResponse(
                job_id=job_id,
                status=result.get('status', 'indexed'),
                documents_indexed=result.get('documents_indexed', 0),
                chunks=result.get('chunks'),
                parent_chunks=result.get('parent_chunks'),
                child_chunks=result.get('child_chunks'),
            )

        if status == 'failed':
            raise HTTPException(
                status_code=500,
                detail={
                    'job_id': job_id,
                    'message': job.get('error') or 'Indexer failed',
                },
            )

        await asyncio.sleep(0.5)


@app.post('/index', response_model=IndexResponse)
async def index_endpoint(request: IndexRequest):
    """Submit documents to the dedicated indexing worker."""
    try:
        path = Path(request.path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        queue: IndexJobQueue = app.state.index_queue
        job_id = await queue.enqueue(
            path=str(path),
            strategy=request.strategy,
        )
        logger.info(
            f"Index job submitted | job_id={job_id} | "
            f"path={path} | strategy={request.strategy}",
        )
        return await wait_for_index_job(queue, job_id)

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        logger.error(f"Index failed - file not found: {exc}")
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Index failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/index/{job_id}/retry', response_model=IndexResponse)
async def retry_index_endpoint(job_id: str):
    """Retry a failed job from its last completed JSONL document."""
    queue: IndexJobQueue = app.state.index_queue
    try:
        await queue.retry(job_id)
        logger.info(f"Index job retry queued | job_id={job_id}")
        return await wait_for_index_job(queue, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"Index retry failed | job_id={job_id}: {exc}", exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/sessions/{session_id}', response_model=SessionResponse)
async def get_session(session_id: str):
    """Retrieve session history"""
    try:
        logger.info(f"Session retrieval | session: {session_id}")

        session_store = get_session_store()
        if session_store is None:
            raise HTTPException(
                status_code=503,
                detail='Session storage is not configured',
            )
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
        if session_store is None:
            raise HTTPException(
                status_code=503,
                detail='Session storage is not configured',
            )
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
