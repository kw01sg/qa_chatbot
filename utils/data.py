import logging
from pathlib import Path
from typing import List, Tuple

import tabula
from haystack import Document
from haystack.nodes import PDFToTextConverter, PreProcessor

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s",
                    level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
# TODO: create handlers to log to standard output and file


def pdf_to_text_and_tables(dir_path: Path, read_tables=False) -> Tuple[List[Document], List[Document]]:
    # text passages, tables are represented as Document objects in Haystack
    converter = PDFToTextConverter(remove_numeric_tables=True, 
                                   valid_languages=["en"])

    text_results = []
    table_dfs = []
    for i in dir_path.iterdir():
        doc_pdf = converter.convert(file_path=i, meta=None)[0]
        text_results.append(doc_pdf)

        if read_tables:
            tables = tabula.read_pdf(i, pages='all')
            table_dfs.extend(tables)

    table_results = []
    if read_tables:
        for id_df, df in enumerate(table_dfs):
            document = Document(content=df, content_type="table",
                                id=f"table_{id_df}")
            table_results.append(document)

    return text_results, table_results


def preprocess_text_documents(documents: List[Document], params: dict = {}) -> List[Document]:
    default_params = {
        'clean_empty_lines': True,
        'clean_whitespace': True,
        'split_by': "word",
        'split_length': 100,
        'split_respect_sentence_boundary': True,
    }
    params = {**default_params, **params}

    preprocessor = PreProcessor(**params)
    results = preprocessor.process(documents)

    return results
