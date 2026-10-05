from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from loguru import logger


def convert_documents(input_path: Path, spool_path: Path) -> None:
    from ragnar.indexing.document_spool import write_document
    from ragnar.loaders.docling_loader import (
        DoclingLoader,
        ensure_docling_models,
    )

    ensure_docling_models()
    loader = DoclingLoader()
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = spool_path.with_suffix(spool_path.suffix + '.partial')
    documents_written = 0

    try:
        with partial_path.open('w', encoding='utf-8') as output:
            for document in loader.iter_load(input_path):
                write_document(document, output)
                documents_written += 1

        os.replace(partial_path, spool_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    logger.info(
        f'Docling conversion stage complete | documents={documents_written} '
        f'| spool={spool_path}',
    )


async def index_documents(
    strategy: str,
    spool_path: Path,
    checkpoint_path: Path,
    result_path: Path,
) -> None:
    from ragnar.embeddings.dense.fast_dense import DenseFastEmbedding
    from ragnar.embeddings.sparse.fast_sparse import SparseFastEmbedding
    from ragnar.indexing.document_spool import DocumentSpoolLoader
    from ragnar.indexing.pipeline_factory import build_index_pipeline

    pipeline = build_index_pipeline(
        strategy,
        loader=DocumentSpoolLoader(checkpoint_path),
        dense_embedder=DenseFastEmbedding(),
        sparse_embedder=SparseFastEmbedding(),
        qdrant_url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
        parent_collection_name=os.getenv('PARENT_COLLECTION', 'sections'),
        child_collection_name=os.getenv('CHILD_COLLECTION', 'subsections'),
    )
    result = await pipeline.index(spool_path)

    partial_path = result_path.with_suffix(result_path.suffix + '.partial')
    try:
        with partial_path.open('w', encoding='utf-8') as output:
            json.dump(result, output)
        os.replace(partial_path, result_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='stage', required=True)

    convert_parser = commands.add_parser('convert')
    convert_parser.add_argument('input_path', type=Path)
    convert_parser.add_argument('spool_path', type=Path)

    index_parser = commands.add_parser('index')
    index_parser.add_argument('strategy', choices=('flat', 'hierarchical'))
    index_parser.add_argument('spool_path', type=Path)
    index_parser.add_argument('checkpoint_path', type=Path)
    index_parser.add_argument('result_path', type=Path)

    args = parser.parse_args()
    if args.stage == 'convert':
        convert_documents(args.input_path, args.spool_path)
    else:
        asyncio.run(
            index_documents(
                args.strategy,
                args.spool_path,
                args.checkpoint_path,
                args.result_path,
            ),
        )


if __name__ == '__main__':
    main()
