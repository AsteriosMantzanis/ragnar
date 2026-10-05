from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from loguru import logger

from ragnar.indexing.job_queue import IndexJobQueue
from ragnar.indexing.job_queue import QUEUE_NAME
from ragnar.utils.logging import configure_logging


class IndexStageError(RuntimeError):
    def __init__(self, stage: str, returncode: int):
        self.stage = stage
        self.returncode = returncode
        signal_number = (
            -returncode
            if returncode < 0
            else returncode - 128
            if returncode >= 128
            else None
        )

        if signal_number is None:
            message = f'Index stage {stage} exited with code {returncode}.'
        else:
            if signal_number == 9:
                signal_name = 'SIGKILL'
            else:
                try:
                    signal_name = signal.Signals(signal_number).name
                except ValueError:
                    signal_name = f'signal {signal_number}'

            message = (
                f'Index stage {stage} was terminated by {signal_name} '
                f'(signal {signal_number}).'
            )
            if signal_number == 9:
                message += (
                    ' This commonly indicates an out-of-memory kill; '
                    'check Docker memory usage and reduce the indexing '
                    'stage peak if needed.'
                )

        super().__init__(message)


def run_stage(*arguments: str) -> None:
    command = [
        sys.executable,
        '-m',
        'ragnar.indexing.job_stage',
        *arguments,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise IndexStageError(
            stage=arguments[0],
            returncode=completed.returncode,
        )


async def process_job(
    queue: IndexJobQueue,
    job: dict,
) -> None:
    job_id = job['job_id']
    path = Path(job['path'])
    strategy = job['strategy']

    logger.info(
        f"Starting index job | job_id={job_id} | "
        f"path={path} | strategy={strategy}",
    )

    await queue.update(
        job_id,
        status='running',
    )

    spool_dir: Path | None = None
    succeeded = False

    try:
        if not path.exists():
            raise FileNotFoundError(
                f"Path not found: {path}",
            )

        spool_root = Path(
            os.getenv(
                'INDEX_SPOOL_DIR',
                '/var/lib/ragnar/index-spool',
            ),
        )
        spool_root.mkdir(parents=True, exist_ok=True)
        spool_dir = spool_root / job_id
        spool_dir.mkdir(parents=True, exist_ok=True)
        spool_path = spool_dir / 'documents.jsonl'
        checkpoint_path = spool_dir / 'progress.json'
        result_path = spool_dir / 'result.json'

        if spool_path.exists():
            logger.info(
                f'Reusing converted JSONL spool | job_id={job_id} '
                f'| spool={spool_path}',
            )
        else:
            checkpoint_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            logger.info(f'Running conversion stage | job_id={job_id}')
            run_stage('convert', str(path), str(spool_path))

        logger.info(f'Running embedding/index stage | job_id={job_id}')
        run_stage(
            'index',
            strategy,
            str(spool_path),
            str(checkpoint_path),
            str(result_path),
        )

        with result_path.open(encoding='utf-8') as result_file:
            result = json.load(result_file)

        await queue.update(
            job_id,
            status='completed',
            result=result,
        )

        logger.info(
            f"Index job completed | job_id={job_id} | "
            f"documents={result.get('documents_indexed', 0)}",
        )
        succeeded = True

    except Exception as exc:
        logger.exception(
            f"Index job failed | job_id={job_id}: {exc}",
        )

        await queue.update(
            job_id,
            status='failed',
            error=str(exc),
        )

        if isinstance(exc, IndexStageError) and exc.returncode < 0:
            raise SystemExit(1) from exc

    finally:
        if spool_dir is not None:
            if succeeded:
                shutil.rmtree(spool_dir, ignore_errors=True)
                logger.info(f'Removed successful job spool | job_id={job_id}')
            else:
                logger.warning(
                    f'Retaining failed job spool for diagnosis | '
                    f'job_id={job_id} | path={spool_dir}',
                )


async def worker_loop() -> None:
    configure_logging(level='INFO')

    logger.info('Ragnar indexer starting...')

    logger.info('Resource-isolated indexing worker ready')

    queue = IndexJobQueue()
    await queue.connect()

    if queue.client is None:
        raise RuntimeError('Could not connect to Redis')

    logger.info(
        f"Waiting for indexing jobs on {QUEUE_NAME}",
    )

    while True:
        item = await queue.client.blpop(
            QUEUE_NAME,
            timeout=0,
        )

        if item is None:
            continue

        _, raw_job = item

        try:
            job = json.loads(raw_job)
        except json.JSONDecodeError:
            logger.error(
                f"Invalid indexing job payload: {raw_job}",
            )
            continue

        await process_job(queue, job)


def main() -> None:
    asyncio.run(worker_loop())


if __name__ == '__main__':
    main()
