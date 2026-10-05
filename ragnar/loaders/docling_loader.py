from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions as PdfOptions,
)
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import TableItem
from loguru import logger

from ragnar.loaders.interfaces.base_loader import BaseLoader
from ragnar.models.document import Document
from ragnar.models.document_element import DocumentElement
from ragnar.utils.memory import release_memory


def build_pdf_options() -> PdfOptions:
    """Lean PDF pipeline: text + tables, small batches, no page images.

    Docling's defaults keep page images and run 4-page batches through the
    layout/table/OCR models, which is what drives its RSS into the GBs.
    Everything here can be raised through env vars if RAM allows.
    """
    batch = int(os.getenv('DOCLING_BATCH_SIZE', '1'))

    options = PdfOptions(
        do_ocr=os.getenv('DOCLING_DO_OCR', 'false').lower() == 'true',
        do_table_structure=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        do_picture_description=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        ocr_batch_size=batch,
        layout_batch_size=batch,
        table_batch_size=batch,
    )
    options.accelerator_options = AcceleratorOptions(
        num_threads=int(os.getenv('DOCLING_NUM_THREADS', '4')),
        device=AcceleratorDevice.CPU,
    )
    return options


def ensure_docling_models() -> None:
    """Download Docling's models into DOCLING_ARTIFACTS_PATH once.

    Without this the models download lazily during the first index request
    into an ephemeral container path and are fetched again after every
    container re-create. Only downloads what build_pdf_options() uses.
    """
    artifacts = os.getenv('DOCLING_ARTIFACTS_PATH')
    if not artifacts:
        return

    target = Path(artifacts)
    if target.exists() and any(target.iterdir()):
        logger.info(f'Docling models already cached in {target}')
        return

    from docling.utils.model_downloader import download_models

    logger.info(f'Downloading Docling models into {target} (first run)')
    download_models(
        output_dir=target,
        progress=True,
        with_layout=True,
        with_tableformer=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=(
            os.getenv('DOCLING_DO_OCR', 'false').lower() == 'true'
        ),
    )
    logger.info('Docling models downloaded')


class DoclingLoader(BaseLoader):
    def __init__(self):
        self.allowed_formats = {'pdf', 'docx'}
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=build_pdf_options(),
                ),
            },
        )

    def load(self, path: Path) -> list[Document]:
        """Preserve the existing eager loader API."""
        return list(self.iter_load(path))

    def iter_load(self, path: Path) -> Iterator[Document]:
        path = Path(path)

        if path.is_file():
            files = [path]
        else:
            files = sorted(
                f
                for f in path.rglob('*')
                if f.is_file()
                and f.suffix.lstrip('.').lower() in self.allowed_formats
            )

        logger.info(f"Docling found {len(files)} supported files")

        for file_path in files:
            result = None

            try:
                logger.info(f"Converting: {file_path}")

                result = self.converter.convert(
                    str(file_path),
                    raises_on_error=False,
                )

                document = self._to_ragnar_document(result)

                # Critical: release Docling's large object graph
                del result
                result = None

                release_memory()

                yield document

                # Release Ragnar document before next PDF
                del document

            except Exception as exc:
                logger.exception(
                    f"Failed to convert {file_path}: {exc}",
                )

            finally:
                if result is not None:
                    del result

                release_memory()

    def _to_ragnar_document(self, result) -> Document:
        doc = result.document
        path = Path(result.input.file)

        elements = []

        for item, _ in doc.iterate_items():
            if isinstance(item, TableItem):
                text = item.export_to_markdown(doc)
            else:
                text = getattr(item, 'text', None)

            if not text or not text.strip():
                continue

            elements.append(
                DocumentElement(
                    text=text,
                    label=item.label.value,
                    page=getattr(item.prov[0], 'page_no', None),
                ),
            )

        doc_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]

        return Document(
            id=doc_id,
            elements=elements,
            metadata={
                'filename': path.name,
                'format': path.suffix.lstrip('.'),
                'loader': 'DoclingLoader',
                'element_count': len(elements),
            },
        )
