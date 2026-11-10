import os
import numpy as np

# Lightweight embedding model — works locally without an API key.
MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_model = None
_model_failed = False          # set True when offline and no local cache


def get_embedding_model():
    global _model, _model_failed

    if _model is not None or _model_failed:
        return _model

    # ── 1. Try loading from local cache (no network) ──
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (local cache)...")
        _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        print("Embedding model loaded from local cache.")
        return _model
    except Exception:
        pass

    # ── 2. Try downloading with a strict timeout ──
    try:
        import socket
        socket.setdefaulttimeout(15)          # 15 s global socket timeout
        from sentence_transformers import SentenceTransformer
        print("Downloading embedding model (may take a moment)...")
        _model = SentenceTransformer(MODEL_NAME)
        socket.setdefaulttimeout(None)
        print("Embedding model downloaded and loaded.")
        return _model
    except Exception as e:
        socket.setdefaulttimeout(None)
        print(f"[RAG] Embedding model unavailable ({e}). RAG will return empty results.")
        _model_failed = True
        return None


def create_embedding(text: str) -> list:
    """
    Convert text into an embedding vector.
    Returns a zero vector if the model is unavailable (offline fallback).
    """
    model = get_embedding_model()
    if model is None:
        # Return a 384-dim zero vector (matches all-MiniLM-L6-v2 output size)
        return [0.0] * 384

    vector = model.encode(
        text,
        normalize_embeddings=True,
    )
    return vector.tolist()