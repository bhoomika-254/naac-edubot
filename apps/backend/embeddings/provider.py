"""Embedding provider abstraction with a lightweight serverless-safe fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import List, Sequence

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class SimpleHashEmbedder:
    """Deterministic hash-based embedder that avoids heavyweight ML dependencies."""

    def __init__(self, embedding_dim: int = 384) -> None:
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        self.embedding_dim = int(embedding_dim)

    def encode(
        self,
        texts: Sequence[str],
        normalize_embeddings: bool = False,
        batch_size: int = 128,
        show_progress_bar: bool = False,
    ) -> List[List[float]]:
        """Return deterministic vectors for all texts with SentenceTransformer-compatible signature."""
        _ = batch_size
        _ = show_progress_bar

        vectors: List[List[float]] = []
        for text in texts:
            vectors.append(self._embed_text(str(text or ""), normalize_embeddings))
        return vectors

    def _embed_text(self, text: str, normalize_embeddings: bool) -> List[float]:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        vector = [0.0] * self.embedding_dim

        if not tokens:
            return vector

        # Mix unigram and bigram signals with decayed weights.
        for idx, token in enumerate(tokens):
            weight = 1.0 / (1.0 + math.log1p(idx + 1.0))
            self._accumulate_feature(vector, f"w:{token}", weight)

            if idx + 1 < len(tokens):
                bigram = f"{token}_{tokens[idx + 1]}"
                self._accumulate_feature(vector, f"b:{bigram}", 0.7 * weight)

        if normalize_embeddings:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0.0:
                vector = [value / norm for value in vector]

        return vector

    def _accumulate_feature(self, vector: List[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        hashed = int.from_bytes(digest, byteorder="big", signed=False)
        index = hashed % self.embedding_dim
        sign = 1.0 if (hashed & 1) == 0 else -1.0
        vector[index] += sign * weight


def build_embedder(
    embedding_provider: str,
    embedding_model: str,
    embedding_dim: int,
    embedding_device: str = "cpu",
):
    """Build embedder instance based on provider preference.

    Supported providers:
    - `simple`: built-in hash embedder (default, serverless-safe)
    - `sentence-transformers`: SentenceTransformer model (optional dependency)
    - `auto`: sentence-transformers when installed, otherwise simple
    """

    provider = (embedding_provider or "simple").strip().lower()

    if provider in {"sentence-transformers", "sentence_transformers", "st", "auto"}:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info(
                "Using sentence-transformers embedder (model=%s, device=%s)",
                embedding_model,
                embedding_device,
            )
            return SentenceTransformer(embedding_model, device=embedding_device)
        except Exception as exc:
            logger.warning(
                "Could not initialize sentence-transformers provider (%s). "
                "Falling back to simple hash embedder.",
                exc,
            )

    logger.info("Using simple hash embedder (dim=%s)", embedding_dim)
    return SimpleHashEmbedder(embedding_dim=embedding_dim)
