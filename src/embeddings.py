"""
embeddings.py

Utilities for generating semantic embeddings used in the
Retrieval-Augmented Generation (RAG) pipeline.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Load a Sentence Transformer embedding model.

    Parameters
    ----------
    model_name : str
        Name of the embedding model.

    Returns
    -------
    SentenceTransformer
        Loaded embedding model.
    """

    return SentenceTransformer(model_name)


def generate_embeddings(
    chunks,
    model
):
    """
    Generate embeddings for document chunks.

    Parameters
    ----------
    chunks : list
        List of text chunks.

    model : SentenceTransformer
        Loaded embedding model.

    Returns
    -------
    numpy.ndarray
        Chunk embeddings as float32 vectors.
    """

    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    return np.array(
        embeddings
    ).astype("float32")


def generate_query_embedding(
    query,
    model
):
    """
    Generate an embedding for a user query.

    Parameters
    ----------
    query : str
        User question.

    model : SentenceTransformer
        Loaded embedding model.

    Returns
    -------
    numpy.ndarray
        Query embedding.
    """

    embedding = model.encode([query])

    return np.array(
        embedding
    ).astype("float32")


if __name__ == "__main__":

    model = load_embedding_model()

    sample_chunks = [
        "Retrieval-Augmented Generation combines retrieval and generation.",
        "FAISS is used for vector similarity search."
    ]

    embeddings = generate_embeddings(
        sample_chunks,
        model
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )
