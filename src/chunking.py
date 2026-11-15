"""
chunking.py

Utilities for splitting university documents into
overlapping text chunks for retrieval and embedding
generation in the RAG pipeline.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Create a RecursiveCharacterTextSplitter.

    Parameters
    ----------
    chunk_size : int
        Maximum size of each chunk.

    chunk_overlap : int
        Number of overlapping characters between chunks.

    Returns
    -------
    RecursiveCharacterTextSplitter
        Configured text splitter.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split documents into chunks while preserving
    source document metadata.

    Parameters
    ----------
    documents : list
        List of dictionaries containing document text
        and source information.

    Returns
    -------
    tuple
        chunks, sources
    """

    text_splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    sources = []

    for doc in documents:

        split_chunks = text_splitter.split_text(
            doc["text"]
        )

        chunks.extend(split_chunks)

        sources.extend(
            [doc["source"]] * len(split_chunks)
        )

    return chunks, sources


if __name__ == "__main__":

    sample_document = [
        {
            "source": "sample.pdf",
            "text": (
                "Retrieval-Augmented Generation combines "
                "document retrieval with language models."
            )
        }
    ]

    chunks, sources = chunk_documents(
        sample_document
    )

    print(f"Chunks: {len(chunks)}")
    print(f"Sources: {len(sources)}")
