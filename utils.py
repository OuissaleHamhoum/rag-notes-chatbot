"""Small helpers with no heavy dependencies (easy to unit-test)."""
import os
import re

## Chunks full of example traces (e.g. ReAct appendix) make poor retrieval context
_NOISY = re.compile(r"(Thought \d+:|Action \d+:|Observation \d+:|Search\[|Finish\[|Success:)")


def is_noisy_chunk(text: str, min_len: int = 100) -> bool:
    """True if a chunk is too short or looks like an example trace."""
    if len(text.strip()) < min_len:
        return True
    return len(_NOISY.findall(text)) >= 2


def docs2str(docs, title: str = "Document") -> str:
    """Turn retrieved chunks into one context string with source labels."""
    out = ""
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        name = meta.get("title") or meta.get("Title") or os.path.basename(str(meta.get("source", title)))
        out += f"[Quote from {name}] {getattr(doc, 'page_content', str(doc))}\n"
    return out


def unique_sources(docs) -> list:
    """File names of the sources used, without duplicates, in order."""
    seen, names = set(), []
    for doc in docs:
        src = (getattr(doc, "metadata", {}) or {}).get("source")
        if src:
            name = os.path.basename(str(src))
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names
