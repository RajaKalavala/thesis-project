"""Build and load the dense (ChromaDB) and sparse (BM25) retrieval indices.

ChromaDB:
- One persistent collection named `medqa_textbooks_bge_400` at
  `data/indices/chroma_textbooks/`, with cosine HNSW.
- Metadata per chunk: `{book_name, n_tokens}`.

BM25 (rank-bm25, Okapi):
- Tokenisation: lowercase + simple alphanumeric word split.
- Saved as a pickle of `{"index": BM25Okapi, "chunk_ids": [...]}` — the
  `chunk_ids` list preserves the row ordering so a BM25 score index maps
  back to the right chunk.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

CHROMA_COLLECTION_NAME = "medqa_textbooks_bge_400"
CHROMA_HNSW_METADATA = {"hnsw:space": "cosine"}
CHROMA_ADD_BATCH = 1000

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + simple alphanumeric word split for BM25."""
    return _WORD_RE.findall(text.lower())


def _chroma_client(persist_dir: Path) -> chromadb.api.client.Client:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def build_chroma(
    chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
    persist_dir: Path,
    batch_size: int = CHROMA_ADD_BATCH,
    overwrite: bool = True,
) -> Any:
    """Build the persistent ChromaDB collection from chunks + embeddings.

    `chunks_df` row order MUST match `embeddings` row order.
    Returns the populated collection.
    """
    if len(chunks_df) != len(embeddings):
        raise ValueError(
            f"chunks_df rows ({len(chunks_df)}) != embeddings rows ({len(embeddings)})"
        )

    client = _chroma_client(persist_dir)
    if overwrite:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    coll = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata=CHROMA_HNSW_METADATA,
    )

    n = len(chunks_df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = chunks_df.iloc[start:end]
        coll.add(
            ids=batch["chunk_id"].tolist(),
            embeddings=embeddings[start:end].tolist(),
            documents=batch["text"].tolist(),
            metadatas=[
                {"book_name": str(b), "n_tokens": int(t)}
                for b, t in zip(batch["book_name"], batch["n_tokens"])
            ],
        )
    return coll


def load_chroma(persist_dir: Path) -> Any:
    """Load the ChromaDB collection from disk."""
    client = _chroma_client(persist_dir)
    return client.get_collection(CHROMA_COLLECTION_NAME)


def build_bm25(chunks_df: pd.DataFrame, out_path: Path) -> dict:
    """Build BM25 over chunk text, save as pickle, and return the payload dict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenised = [_tokenize(t) for t in chunks_df["text"]]
    index = BM25Okapi(tokenised)
    payload = {
        "index": index,
        "chunk_ids": chunks_df["chunk_id"].tolist(),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def load_bm25(path: Path) -> dict:
    """Load the pickled BM25 payload `{"index", "chunk_ids"}`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def bm25_top_k(query: str, bm25_payload: dict, k: int = 5) -> list[tuple[str, float]]:
    """Score `query` against BM25 and return top-k as [(chunk_id, score), …]."""
    bm25 = bm25_payload["index"]
    chunk_ids = bm25_payload["chunk_ids"]
    scores = bm25.get_scores(_tokenize(query))
    top_idx = np.argsort(-scores)[:k]
    return [(chunk_ids[i], float(scores[i])) for i in top_idx]
