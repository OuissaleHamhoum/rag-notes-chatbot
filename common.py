"""Shared configuration: models and paths."""
import os

from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
CHAT_MODEL = os.getenv("CHAT_MODEL", "meta/llama-3.1-8b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "nvidia")  # "nvidia" or "fastembed"


def get_llm():
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(model=CHAT_MODEL, temperature=0.2)


def get_embedder():
    if EMBED_BACKEND == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    return NVIDIAEmbeddings(model=EMBED_MODEL)
