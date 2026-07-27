from __future__ import annotations

from docling.document_converter import DocumentConverter


def main():

    converter = DocumentConverter()
    result = converter.convert(
        r'C:\\Users\\aster\\Downloads\\HP0300-Manual.pdf',
    )
    doc = result.document
    table_md = doc.tables[0].export_to_markdown(doc)
    return doc, table_md


if __name__ == '__main__':
    main()
