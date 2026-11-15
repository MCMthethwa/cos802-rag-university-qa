"""
ingest.py

Utilities for loading and extracting text from PDF documents used in the
Retrieval-Augmented Generation (RAG) pipeline.
"""

from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF document.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF document.

    Returns
    -------
    str
        Extracted document text.
    """

    reader = PdfReader(str(pdf_path))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_documents(data_directory):
    """
    Load all PDF documents from a directory.

    Parameters
    ----------
    data_directory : str or Path
        Directory containing PDF files.

    Returns
    -------
    list
        List of dictionaries containing source names and extracted text.
    """

    data_directory = Path(data_directory)

    documents = []

    for pdf_file in data_directory.glob("*.pdf"):

        documents.append(
            {
                "source": pdf_file.name,
                "text": extract_text_from_pdf(pdf_file)
            }
        )

    return documents


if __name__ == "__main__":

    documents = load_documents("../data/raw")

    print(f"Loaded {len(documents)} documents.")
