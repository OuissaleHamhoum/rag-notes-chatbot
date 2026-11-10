# Chat with my notes (RAG chatbot)

A small retrieval-augmented generation (RAG) project built after completing the NVIDIA DLI course
**Building RAG Agents with LLMs**. Ask questions about your own PDFs, notes or papers and get answers
grounded in your documents, with the sources listed.

## How it works
```
data/*.pdf|md|txt -> split into chunks -> embeddings -> FAISS index
question -> retrieve top chunks -> reorder -> prompt -> LLM -> answer + sources
```

| File | Purpose |
|---|---|
| `build_index.py` | Load documents, chunk them, drop noisy chunks, build the FAISS index |
| `rag.py` | Retrieval and generation pipeline (LangChain) |
| `app.py` | Gradio chat window (Basic vs RAG mode) |
| `server.py` | Optional LangServe API: `/basic_chat`, `/retriever`, `/generator` |
| `evaluate.py` | Optional LLM-as-a-judge comparison of Basic vs RAG |
| `utils.py`, `tests/` | Helper functions and unit tests |

## Setup
```bash
git clone https://github.com/<your-username>/rag-notes-chatbot.git
cd rag-notes-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your NVIDIA_API_KEY inside .env
```
Get a free API key at https://build.nvidia.com. Never commit your `.env` file.

## Run
```bash
python build_index.py   # 1) build the index from data/
python app.py           # 2) open http://localhost:7860
python server.py        # optional: API docs at http://localhost:9012/docs
python evaluate.py      # optional: Basic vs RAG comparison
pytest -q               # unit tests
```

## What I learned
- Chunking, embeddings and vector search with FAISS
- Building a RAG chain with LangChain and serving it with LangServe
- Evaluating answers with an LLM judge
- Filtering noisy chunks before indexing improves answers

## Certificate
NVIDIA Deep Learning Institute: Building RAG Agents with LLMs (add your certificate link here)
