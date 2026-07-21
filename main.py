from __future__ import annotations

from docling.document_converter import DocumentConverter


def main():

    converter = DocumentConverter()
    result = converter.convert(
        r'',
    )
    doc = result.document
    return doc


if __name__ == '__main__':
    main()
