#!/usr/bin/env python3
"""
Total Recall — Embedding Utilities
Shared module for Ollama-based embedding, BLOB serialization, and cosine similarity.
"""

import struct
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


def get_embedding(text: str) -> list[float]:
    """Embed a single text string via Ollama. Returns 768-dim float list."""
    text = text[:8000]  # nomic-embed-text: 8192 token window, ~8K chars safe for all content types
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]


def floats_to_blob(vec: list[float]) -> bytes:
    """Pack float list into a compact binary BLOB (little-endian float32)."""
    return struct.pack(f'<{len(vec)}f', *vec)


def blob_to_floats(blob: bytes) -> list[float]:
    """Unpack a BLOB back into a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f'<{n}f', blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure stdlib, no numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
