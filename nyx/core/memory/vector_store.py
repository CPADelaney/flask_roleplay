"""FAISS backed vector store used by :mod:`nyx.core`."""

import atexit
import json
import os
import asyncio
import uuid
import functools
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Simple FAISS-backed vector store
_model = SentenceTransformer("all-MiniLM-L6-v2")
_dim = _model.get_sentence_embedding_dimension()

_index = faiss.IndexIDMap(faiss.IndexFlatIP(_dim))
_lock = asyncio.Lock()

_store: Dict[int, Dict[str, Dict]] = {}
_uid_to_faiss: Dict[str, int] = {}
_memory_uid: Dict[str, str] = {}
_next_id = 0

_DATA_PATH = os.path.join(os.path.dirname(__file__), "vector_store")

def _to_vec(texts: List[str]) -> np.ndarray:
    """Encode text to normalized float32 vectors."""
    vec = _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32").reshape(len(texts), -1)


async def _remove_by_uid(uid: str) -> None:
    """Remove an entry by its uid if present."""
    idx = _uid_to_faiss.pop(uid, None)
    if idx is None:
        return
    async with _lock:
        _index.remove_ids(np.array([idx], dtype="int64"))
    entry = _store.pop(idx, None)
    if entry:
        mid = entry["meta"].get("memory_id")
        if mid and _memory_uid.get(mid) == uid:
            del _memory_uid[mid]


async def add(text: str, meta: Dict) -> str:
    """Add text with metadata to the store and return generated id."""
    global _next_id
    mid = meta.get("memory_id")
    if mid and mid in _memory_uid:
        await _remove_by_uid(_memory_uid[mid])

    uid = str(uuid.uuid4())
    vec = _to_vec([text])
    async with _lock:
        _index.add_with_ids(vec, np.array([_next_id], dtype="int64"))

    stored_meta = {**meta, "uid": uid}
    _store[_next_id] = {"text": text, "meta": stored_meta}
    _uid_to_faiss[uid] = _next_id
    if mid:
        _memory_uid[mid] = uid

    _next_id += 1
    return uid


async def _faiss_search(vec, k):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(_index.search, vec, k))


async def query(query_text: str, k: int = 5) -> List[Dict]:
    """Return top-``k`` texts with similarity scores."""
    if _index.ntotal == 0:
        return []
    vec = _to_vec([query_text])
    D, I = await _faiss_search(vec, k)
    async with _lock:
        entries = [
            (float(score), _store.get(idx))
            for score, idx in zip(D[0], I[0])
            if idx != -1
        ]

    results = [
        {
            "text": entry["text"],
            "meta": entry["meta"],
            "score": score,
        }
        for score, entry in entries
        if entry
    ]
    return results


async def save(path: str = _DATA_PATH) -> None:
    """Persist the store to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with _lock:
        faiss.write_index(_index, f"{path}.index")
        data = {
            "next_id": _next_id,
            "store": _store,
            "uid_to_faiss": _uid_to_faiss,
            "memory_uid": _memory_uid,
        }
        with open(f"{path}.json", "w") as f:
            json.dump(data, f)


def load(path: str = _DATA_PATH) -> None:
    """Load the store from disk if available."""
    global _next_id, _store, _uid_to_faiss, _memory_uid, _index
    if os.path.exists(f"{path}.index"):
        _index = faiss.read_index(f"{path}.index")
        with open(f"{path}.json", "r") as f:
            data = json.load(f)
        _next_id = data.get("next_id", 0)
        _store = {int(k): v for k, v in data.get("store", {}).items()}
        _uid_to_faiss = {k: int(v) for k, v in data.get("uid_to_faiss", {}).items()}
        _memory_uid = data.get("memory_uid", {})


# Initialise store on import and register persistence hook
load(_DATA_PATH)

def _sync_save(path: str = _DATA_PATH) -> None:
    asyncio.run(save(path))

atexit.register(_sync_save, _DATA_PATH)

