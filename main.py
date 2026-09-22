"""Run Ragnar's indexing and query pipelines directly, outside Docker and
outside FastAPI.

This calls the same pipeline factories the API uses
(`ragnar.api.dependencies`), so behavior matches the API exactly — it just
skips uvicorn and hits the pipeline objects in-process. Qdrant, Redis and
Ollama still need to be reachable (e.g. `docker compose up qdrant redis
ollama`, then run this script on the host against .env's URLs).

Logs stream to stderr as the pipeline runs — every step (retrieve,
retrieve, dedupe, rerank, generate, ground) logs through loguru already, so
there's nothing extra to wire up to see them; they just print.

Usage:
    uv run python main.py index --path ./docs --strategy hierarchical
    uv run python main.py query --strategy hierarchical
    uv run python main.py query --strategy hierarchical --question "What is X?"
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from ragnar.api.dependencies import get_index_pipeline
from ragnar.api.dependencies import get_query_pipeline
from ragnar.utils.logging import configure_logging

load_dotenv()
configure_logging(level='INFO')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument(
        '--path', required=True,
        help='Directory of documents to index',
    )
    index_parser.add_argument(
        '--strategy', default='hierarchical',
        choices=['flat', 'hierarchical'],
    )

    query_parser = subparsers.add_parser(
        'query',
        help='Query the index — interactive loop if --question is omitted',
    )
    query_parser.add_argument(
        '--strategy', default='hierarchical',
        choices=['flat', 'hierarchical'],
    )
    query_parser.add_argument('--session-id', default='cli-session')
    query_parser.add_argument(
        '--question', default=None,
        help='Ask a single question and exit, instead of looping',
    )

    return parser


def print_query_result(result: dict) -> None:
    print(f"\nAnswer: {result.get('answer')}\n")
    print(f"Grounded: {result.get('grounded_percentage', 0):.0f}%")

    sources = result.get('sources', [])
    print(f"Sources ({len(sources)}):")
    for i, source in enumerate(sources[:5], start=1):
        filename = source.get('filename', source.get('source', 'unknown'))
        print(f"  {i}. {filename}")

    metrics = result.get('metrics', {})
    if metrics.get('steps'):
        print('\nStep timings:')
        for step in metrics['steps']:
            print(f"  {step['name']:<12} {step['duration_s']:.2f}s")

    print()


async def run_index(path: str, strategy: str) -> None:
    index_pipeline = get_index_pipeline(strategy)
    result = await index_pipeline.index(Path(path))
    print(
        '\nDone — '
        f'{result.get("documents_indexed", 0)} document(s) indexed.\n',
    )


async def run_single_query(
    question: str,
    strategy: str,
    session_id: str,
) -> None:
    query_pipeline = get_query_pipeline(strategy)
    result = await query_pipeline.query(
        user_query=question,
        session_id=session_id,
    )
    print_query_result(result)


async def run_interactive_query(strategy: str, session_id: str) -> None:
    query_pipeline = get_query_pipeline(strategy)
    print(
        'Ragnar interactive query — '
        f'strategy: {strategy}, session: {session_id}',
    )
    print("Type a question, or 'exit' / Ctrl+D to quit.\n")

    while True:
        try:
            question = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in ('exit', 'quit'):
            break

        result = await query_pipeline.query(
            user_query=question,
            session_id=session_id,
        )
        print_query_result(result)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == 'index':
        asyncio.run(run_index(args.path, args.strategy))
    elif args.command == 'query':
        if args.question:
            asyncio.run(
                run_single_query(
                    args.question,
                    args.strategy,
                    args.session_id,
                ),
            )
        else:
            asyncio.run(
                run_interactive_query(args.strategy, args.session_id),
            )


if __name__ == '__main__':
    main()
