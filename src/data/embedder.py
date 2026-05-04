"""BGE-large-en-v1.5 embedder wrapper for the thesis.

Conventions (from BGE v1.5 model card):
- **Passages** are embedded *without* a prefix.
- **Queries** are embedded with the retrieval-query prefix
  `"Represent this sentence for searching relevant passages: "`.

All embeddings are L2-normalised so cosine similarity reduces to a dot product
and ChromaDB's `hnsw:space="cosine"` returns calibrated distances.
"""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 1024


def best_device() -> str:
    """Pick MPS if available (Apple Silicon), else CPU. CUDA is not in scope."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def load_bge(device: str | None = None) -> SentenceTransformer:
    """Load BAAI/bge-large-en-v1.5 onto the chosen device."""
    device = device or best_device()
    return SentenceTransformer(BGE_MODEL_NAME, device=device)


def embed_passages(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed passages (NO prefix). Returns float32 (n, 1024), L2-normalised."""
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32, copy=False)


def embed_queries(
    model: SentenceTransformer,
    queries: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Embed queries WITH the BGE retrieval prefix. Returns float32 (n, 1024)."""
    prefixed = [BGE_QUERY_PREFIX + q for q in queries]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32, copy=False)
