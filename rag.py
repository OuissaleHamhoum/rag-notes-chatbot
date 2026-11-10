"""The RAG pipeline: retrieve -> reorder -> build context -> generate."""
from functools import lru_cache

from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from common import INDEX_DIR, get_embedder, get_llm
from utils import docs2str, unique_sources

chat_prompt = ChatPromptTemplate.from_template(
    "You are a helpful study assistant answering questions about the user's own documents.\n\n"
    "Retrieved context:\n{context}\n\n"
    "Question: {input}\n\n"
    "Instructions: Answer using ONLY the retrieved context. Start with a direct answer in 1-2 sentences,"
    " then add the concrete supporting details from the context (numbers, names, definitions)."
    " Mention which document a detail comes from. Do not repeat yourself."
    " If the context does not contain the answer, say you could not find it in the documents."
)


@lru_cache(maxsize=1)
def get_store():
    return FAISS.load_local(INDEX_DIR, get_embedder(), allow_dangerous_deserialization=True)


def get_retriever(k: int = 6):
    return get_store().as_retriever(search_kwargs={"k": k})


def _to_dict(x):
    return x if isinstance(x, dict) else {"input": str(x), "context": ""}


def get_generator():
    """Accepts {"input": ..., "context": ...} and returns plain text."""
    return RunnableLambda(_to_dict) | chat_prompt | get_llm() | StrOutputParser()


def rag_answer(question: str):
    docs = get_retriever().invoke(question)
    docs = LongContextReorder().transform_documents(docs)
    answer = get_generator().invoke({"input": question, "context": docs2str(docs)})
    return answer, unique_sources(docs)


def basic_answer(question: str) -> str:
    return get_llm().invoke(question).content
