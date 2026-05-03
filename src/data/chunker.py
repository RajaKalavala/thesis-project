"""Recursive token-based chunker for the medical-textbook corpus.

400-token chunks with 80-token (20%) overlap, using tiktoken `cl100k_base`
for token counting. Produces deterministic chunk IDs of the form
`<book_name>_chunk_<NNNNN>`.

Decision rationale (locked in plan.md §0 #5):
- 400 tokens fits comfortably under BGE-large's 512-token max input.
- 80-token (20%) overlap is the 2024-25 medical-RAG standard, protecting
  against information loss at chunk boundaries.
- cl100k_base is used as a uniform sizing metric. BGE has its own tokenizer
  but 400 cl100k tokens is reliably <= 512 BGE tokens.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS = 30
TOKENIZER_NAME = "cl100k_base"
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER_NAME,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=SEPARATORS,
    )


def chunk_textbook(text: str, book_name: str) -> pd.DataFrame:
    """Chunk a single textbook's full text.

    Returns a DataFrame with columns: chunk_id, book_name, text, n_tokens, n_chars.
    Chunks below MIN_CHUNK_TOKENS are dropped (boilerplate / table residue).
    Chunk IDs are sequential within the book and zero-padded to 5 digits.
    """
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    splitter = _build_splitter()
    raw_chunks = splitter.split_text(text)

    rows = []
    for raw in raw_chunks:
        n_tokens = len(enc.encode(raw))
        if n_tokens < MIN_CHUNK_TOKENS:
            continue
        rows.append({
            "book_name": book_name,
            "text": raw,
            "n_tokens": n_tokens,
            "n_chars": len(raw),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(
            0,
            "chunk_id",
            [f"{book_name}_chunk_{i:05d}" for i in range(len(df))],
        )
    return df


def chunk_corpus(
    textbooks_dir: Path,
    files: Iterable[Path] | None = None,
) -> pd.DataFrame:
    """Chunk every .txt in `textbooks_dir` (or the explicit `files` list).

    Concatenates results across books and returns a single DataFrame.
    """
    if files is None:
        files = sorted(Path(textbooks_dir).glob("*.txt"))
    frames = []
    for fp in files:
        text = Path(fp).read_text(encoding="utf-8")
        frames.append(chunk_textbook(text, Path(fp).stem))
    return pd.concat(frames, ignore_index=True)
