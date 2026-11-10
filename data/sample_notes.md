# Sample notes: RAG basics

Retrieval-augmented generation (RAG) combines a language model with a search step.
The system first retrieves relevant passages from a document collection and then asks
the model to answer using only those passages. This reduces hallucinations and lets
you update knowledge without retraining the model.

## Chunking
Documents are split into overlapping chunks (for example 1000 characters with 200
characters of overlap). Chunks keep each piece small enough to embed and retrieve
precisely, while the overlap avoids cutting an idea in half.

## Embeddings and vector stores
An embedding model turns each chunk into a vector. A vector store such as FAISS finds
the chunks whose vectors are closest to the vector of the question.

## Parametric vs non-parametric memory
Parametric memory is the knowledge stored in the model weights. Non-parametric memory is
an external index that can be searched and updated at any time. RAG uses both.
