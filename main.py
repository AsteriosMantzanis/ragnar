from __future__ import annotations

from pathlib import Path

from ragnar.loaders.docling_loader import DoclingLoader
# from docling.document_converter import DocumentConverter


def main():

    # converter = DocumentConverter()
    # result = converter.convert(
    #     Path('C:/Users/aster/Desktop/manuals/4101RH.pdf'),
    # )
    # doc = result.document
    # table_md = doc.tables[0].export_to_markdown(doc)
    # return doc, table_md

    loader = DoclingLoader()
    documents = loader.load(Path('C:/Users/aster/Desktop/manuals'))
    print(f"Loaded {len(documents)} documents.")


if __name__ == '__main__':
    main()
