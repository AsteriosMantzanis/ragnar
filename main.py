from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter
# from ragnar.loaders.docling_loader import DoclingLoader


def main():

    converter = DocumentConverter()
    result = converter.convert(
        Path('C:/Users/aster/Desktop/manuals/Hammer_Driver_Drill_HP0300.pdf'),
    )
    doc = result.document
    table_md = doc.tables[0].export_to_markdown(doc)
    return doc, table_md

    # loader = DoclingLoader()
    # documents = loader.load(
    #     Path('C:/Users/aster/Desktop/manuals/Hammer_Driver_Drill_HP0300.pdf'),
    # )
    # print(f"Loaded {len(documents)} documents.")
    # print(documents[0])


if __name__ == '__main__':
    main()
