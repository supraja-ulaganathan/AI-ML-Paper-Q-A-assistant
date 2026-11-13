"""Lightweight embeddings (ONNX, no PyTorch). Same model as before: all-MiniLM-L6-v2."""
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

_model = None


def embed(texts):
    global _model
    if _model is None:
        _model = TextEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(Path(__file__).parent / "models"),
        )
    vecs = np.array(list(_model.embed(texts)), dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.tolist()