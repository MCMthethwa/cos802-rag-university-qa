"""
main.py

Example workflow demonstrating how the RAG pipeline
connects document ingestion, chunking, embedding
generation and vector retrieval.
"""

from ingest import load_documents
from chunking import chunk_documents
from embeddings import (
    load_embedding_model,
    generate_embeddings,
    generate_query_embedding
)
from vector_store import (
    create_index,
    retrieve_context
)


def build_knowledge_base(data_directory):

    # Step 1: Load PDFs

    documents = load_documents(
        data_directory
    )

    # Step 2: Chunk documents

    chunks, sources = chunk_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    # Step 3: Load embedding model

    embedding_model = load_embedding_model()

    # Step 4: Generate embeddings

    embeddings = generate_embeddings(
        chunks,
        embedding_model
    )

    # Step 5: Create vector index

    index = create_index(
        embeddings
    )

    return (
        embedding_model,
        index,
        chunks,
        sources
    )


def query_knowledge_base(
    question,
    embedding_model,
    index,
    chunks
):

    query_embedding = generate_query_embedding(
        question,
        embedding_model
    )

    context = retrieve_context(
        query_embedding,
        index,
        chunks,
        top_k=3
    )

    return context


if __name__ == "__main__":

    embedding_model, index, chunks, sources = (
        build_knowledge_base(
            "../data/raw"
        )
    )

    question = (
        "What is the prerequisite for COS802?"
    )

    context = query_knowledge_base(
        question,
        embedding_model,
        index,
        chunks
    )

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVED CONTEXT:")
    print(context)
