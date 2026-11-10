"""Step 1: read files in data/, split them into chunks, embed, and save a FAISS index."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import INDEX_DIR, get_embedder
from utils import is_noisy_chunk

DATA_DIR = Path("data")


def load_documents():
    docs = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(path)).load()
        elif path.suffix.lower() in {".md", ".txt"} and path.name.lower() != "readme.md":
            docs += TextLoader(str(path), encoding="utf-8").load()
    return docs


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit("No .pdf/.md/.txt files found in data/. Add some notes first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", ";", ",", " "],
    )
    chunks = [c for c in splitter.split_documents(docs) if not is_noisy_chunk(c.page_content)]
    print(f"Loaded {len(docs)} pages/files -> {len(chunks)} clean chunks")

    FAISS.from_documents(chunks, get_embedder()).save_local(INDEX_DIR)
    print(f"Saved index to ./{INDEX_DIR}")


if __name__ == "__main__":
    main()
