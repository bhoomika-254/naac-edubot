"""Simple in-memory vector store for local development and serverless fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..embeddings.provider import build_embedder


@dataclass
class _VectorRecord:
    document: str
    metadata: Dict[str, Any]
    embedding: List[float]


class LocalVectorStore:
    """Minimal vector backend that lives entirely in memory."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "simple",
        embedding_dim: int = 384,
        embedding_device: str = "cpu",
        embedding_batch_size: int = 128,
    ) -> None:
        self.embedding_batch_size = max(int(embedding_batch_size or 128), 8)
        self.embedder = build_embedder(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embedding_device=embedding_device,
        )
        self.naac_records: List[_VectorRecord] = []
        self.mvsr_records: List[_VectorRecord] = []

    # ------------------------------------------------------------------
    # Public ingestion helpers
    # ------------------------------------------------------------------
    def add_naac_documents(self, documents: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        self._append_records(self.naac_records, documents, metadatas, doc_type="requirement")

    def add_mvsr_documents(self, documents: Sequence[str], metadatas: Sequence[Dict[str, Any]]) -> None:
        self._append_records(self.mvsr_records, documents, metadatas, doc_type="evidence")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def query_naac_requirements(
        self,
        query_text: str,
        n_results: int = 5,
        criterion_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._query_records(self.naac_records, query_text, n_results, ("criterion", criterion_filter))

    def query_mvsr_evidence(
        self,
        query_text: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._query_records(self.mvsr_records, query_text, n_results, ("category", category_filter))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_collection_stats(self) -> Dict[str, int]:
        return {
            "naac_requirements_count": len(self.naac_records),
            "mvsr_evidence_count": len(self.mvsr_records),
            "total_documents": len(self.naac_records) + len(self.mvsr_records),
        }

    def update_naac_version(self, old_version: str, new_version: str) -> None:
        """Mark matching NAAC rows as archived for version transitions."""
        for record in self.naac_records:
            current_version = str(record.metadata.get("version", "")).strip()
            if current_version == str(old_version).strip():
                record.metadata["status"] = "archived"
                record.metadata["archived_version"] = str(new_version)

    def health_check(self) -> Dict[str, Any]:
        stats = self.get_collection_stats()
        stats.update({"ok": True, "backend": "local-memory"})
        return stats

    def consolidate_single_row_mode(self) -> None:
        """Compatibility shim for the Supabase backend interface."""
        return

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _append_records(
        self,
        store: List[_VectorRecord],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        doc_type: str,
    ) -> None:
        if not documents or not metadatas:
            return
        if len(documents) != len(metadatas):
            raise ValueError("Documents and metadata must have the same length")

        embeddings = self._encode(list(documents))
        for doc, metadata, embedding in zip(documents, metadatas, embeddings, strict=False):
            clean_meta = dict(metadata or {})
            clean_meta.setdefault("type", doc_type)
            store.append(_VectorRecord(document=str(doc), metadata=clean_meta, embedding=embedding))

    def _query_records(
        self,
        store: List[_VectorRecord],
        query_text: str,
        n_results: int,
        filter_pair: Tuple[str, Optional[str]],
    ) -> Dict[str, Any]:
        if not store:
            return {"documents": [], "metadatas": [], "distances": []}

        filter_key, filter_value = filter_pair
        candidates = [
            record
            for record in store
            if not filter_value or record.metadata.get(filter_key) == filter_value
        ]
        if not candidates:
            candidates = store

        query_embedding = self._encode([query_text])[0]

        ranked: List[Tuple[float, _VectorRecord]] = []
        for record in candidates:
            similarity = self._cosine_similarity(record.embedding, query_embedding)
            ranked.append((similarity, record))

        ranked.sort(key=lambda row: row[0], reverse=True)
        top_rows = ranked[: max(int(n_results or 0), 0)]

        documents = [row[1].document for row in top_rows]
        metadatas = [row[1].metadata for row in top_rows]
        distances = [float(1 - row[0]) for row in top_rows]

        return {"documents": documents, "metadatas": metadatas, "distances": distances}

    def _encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
        )

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        return [list(map(float, embedding)) for embedding in embeddings]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        dot = 0.0
        left_norm_sq = 0.0
        right_norm_sq = 0.0
        for left_value, right_value in zip(left, right):
            dot += left_value * right_value
            left_norm_sq += left_value * left_value
            right_norm_sq += right_value * right_value

        if left_norm_sq <= 0.0 or right_norm_sq <= 0.0:
            return 0.0

        return dot / (math.sqrt(left_norm_sq) * math.sqrt(right_norm_sq) + 1e-10)
