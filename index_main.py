from __future__ import annotations

from pathlib import Path

from ragnar.chunkers.section_chunker import SectionChunker
from ragnar.chunkers.subsection_chunker import SubsectionChunker
from ragnar.loaders.docling_loader import DoclingLoader
from ragnar.pipelines.flat_index_pipeline import FlatIndexPipeline
from ragnar.pipelines.hier_index_pipeline import HierarchicalIndexPipeline


async def main():
    loader = DoclingLoader()

    # Flat
    flat_pipeline = FlatIndexPipeline(
        loader=loader,
        chunker=SectionChunker(),
        collection_name='sections',
    )
    await flat_pipeline.index(Path('C:/Users/aster/Desktop/manuals'))

    # Hierarchical
    hier_pipeline = HierarchicalIndexPipeline(
        loader=loader,
        parent_chunker=SectionChunker(),
        child_chunker=SubsectionChunker(),
        parent_collection_name='sections',
        child_collection_name='subsections',
    )
    await hier_pipeline.index(Path('C:/Users/aster/Desktop/manuals'))

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
