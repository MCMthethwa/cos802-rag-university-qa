"""
vector_store.py

Utilities for creating and querying a FAISS vector
store used in the Retrieval-Augmented Generation
(RAG) pipeline.
"""

import faiss
import numpy as np


def create_index(embeddings):
    """
    Create and populate a FAISS vector index.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Document embeddings.

    Returns
    -------
    faiss.IndexFlatL2
        FAISS vector index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_index(
    query_embedding,
    index,
    chunks,
    sources,
    top_k=5
):
    """
    Perform semantic similarity search.

    Parameters
    ----------
    query_embedding : numpy.ndarray
        Embedded user query.

    index : faiss.IndexFlatL2
        FAISS vector index.

    chunks : list
        Document chunks.

    sources : list
        Source document names.

    top_k : int
        Number of results to return.

    Returns
    -------
    list
        Retrieved chunks and their sources.
    """

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(
            {
                "source": sources[idx],
                "chunk": chunks[idx]
            }
        )

    return results


def retrieve_context(
    query_embedding,
    index,
    chunks,
    top_k=3
):
    """
    Retrieve context for answer generation.

    Parameters
    ----------
    query_embedding : numpy.ndarray
        Embedded query.

    index : faiss.IndexFlatL2
        FAISS vector index.

    chunks : list
        Document chunks.

    top_k : int
        Number of chunks to retrieve.

    Returns
    -------
    str
        Combined retrieval context.
    """

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = [
        chunks[idx]
        for idx in indices[0]
    ]

    return "\n\n".join(retrieved_chunks)


if __name__ == "__main__":

    sample_embeddings = np.random.rand        384
    ).astype("float32")

    index = create_index(
        sample_embeddings
    )

    print(
        f"Vectors Stored: {index.ntotal}"
    )
