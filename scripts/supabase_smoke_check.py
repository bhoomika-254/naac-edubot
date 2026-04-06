"""Supabase integration smoke checks for backend CRUD paths.

Runs a lightweight end-to-end verification against configured SUPABASE_DB_URL for:
- SupabaseVectorStore (connect, insert, query, update metadata)
- ServerlessStateStore (staged upload and ingestion status CRUD)
- ConversationMemoryStore (insert and retrieval)

This script creates uniquely tagged rows and cleans them up before exiting.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List
from uuid import uuid4

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.backend.config.settings import get_settings
from apps.backend.db.serverless_state_store import ServerlessStateStore
from apps.backend.db.supabase_store import SupabaseVectorStore
from apps.backend.memory.memory_store import ConversationMemoryStore, MemoryIdentity


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) < 14:
        return "(present but short)"
    return f"{value[:10]}...{value[-4:]}"


def run_checks() -> List[CheckResult]:
    results: List[CheckResult] = []
    settings = get_settings()

    supabase_url = settings.supabase_db_url or ""
    if not supabase_url:
        return [
            CheckResult(
                name="Config",
                ok=False,
                details="SUPABASE_DB_URL is missing in runtime settings.",
            )
        ]

    smoke_id = uuid4().hex[:10]
    naac_file_hash = f"smoke_naac_{smoke_id}"
    mvsr_file_hash = f"smoke_mvsr_{smoke_id}"
    naac_version = f"smoke_version_{smoke_id}"
    status_file_path = f"memory://upload/smoke/{smoke_id}.pdf"
    staged_token = f"smoke-token-{smoke_id}"
    memory_identity = MemoryIdentity(
        tenant_id=f"smoke-tenant-{smoke_id}",
        user_id=f"smoke-user-{smoke_id}",
        conversation_id=f"smoke-conv-{smoke_id}",
    )

    vector_store = None
    state_store = None
    memory_store = None

    print("== Supabase Smoke Check ==")
    print(f"VECTOR_BACKEND={settings.vector_backend}")
    print(f"SUPABASE_TABLE={settings.supabase_table}")
    print(f"SUPABASE_DB_URL={mask_secret(supabase_url)}")

    try:
        conn = psycopg2.connect(supabase_url, connect_timeout=10)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        conn.close()
        results.append(CheckResult("Raw DB connectivity", True, "SELECT 1 succeeded"))
    except Exception as exc:
        results.append(CheckResult("Raw DB connectivity", False, str(exc)))
        return results

    try:
        vector_store = SupabaseVectorStore(
            db_url=supabase_url,
            table_name=settings.supabase_table,
            embedding_model=settings.embedding_model,
            embedding_provider=settings.embedding_provider,
            embedding_dim=settings.embedding_dim,
            embedding_device=settings.embedding_device,
            embedding_batch_size=settings.embedding_batch_size,
            insert_batch_size=settings.vector_insert_batch_size,
        )

        health = vector_store.health_check()
        results.append(
            CheckResult(
                "SupabaseVectorStore.health_check",
                bool(health.get("ok")),
                f"table_exists={health.get('table_exists')} total_rows={health.get('total_rows', 'n/a')}",
            )
        )

        vector_store.add_naac_documents(
            [
                "Smoke NAAC requirement text about curriculum design and quality assurance.",
            ],
            [
                {
                    "criterion": "1",
                    "section_header": "Smoke NAAC Section",
                    "file_hash": naac_file_hash,
                    "source_file": f"smoke_naac_{smoke_id}.pdf",
                    "version": naac_version,
                }
            ],
        )
        vector_store.add_mvsr_documents(
            [
                "Smoke MVSR evidence text describing institutional implementation and records.",
            ],
            [
                {
                    "category": "curriculum",
                    "section_header": "Smoke MVSR Section",
                    "file_hash": mvsr_file_hash,
                    "source_file": f"smoke_mvsr_{smoke_id}.pdf",
                }
            ],
        )

        naac_q = vector_store.query_naac_requirements(
            "curriculum quality requirement",
            n_results=3,
            criterion_filter="1",
        )
        mvsr_q = vector_store.query_mvsr_evidence(
            "institutional implementation evidence",
            n_results=3,
            category_filter="curriculum",
        )

        naac_ok = len(naac_q.get("documents", [])) > 0
        mvsr_ok = len(mvsr_q.get("documents", [])) > 0
        results.append(
            CheckResult(
                "SupabaseVectorStore CRUD",
                naac_ok and mvsr_ok,
                f"naac_hits={len(naac_q.get('documents', []))} mvsr_hits={len(mvsr_q.get('documents', []))}",
            )
        )

        vector_store.update_naac_version(naac_version, f"archived_{naac_version}")
        with psycopg2.connect(supabase_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT metadata->>'status'
                    FROM {settings.supabase_table}
                    WHERE (metadata->>'file_hash') = %s
                    LIMIT 1;
                    """,
                    (naac_file_hash,),
                )
                row = cur.fetchone()
        archived_ok = bool(row and row[0] == "archived")
        results.append(
            CheckResult(
                "SupabaseVectorStore.update_naac_version",
                archived_ok,
                f"status={row[0] if row else 'not-found'}",
            )
        )

    except Exception as exc:
        results.append(CheckResult("SupabaseVectorStore", False, str(exc)))

    try:
        state_store = ServerlessStateStore(
            db_url=supabase_url,
            staged_upload_ttl_minutes=45,
            ingestion_status_ttl_hours=24,
        )
        state_store.initialize_schema()
        state_store.cleanup_expired()

        state_store.set_staged_upload(
            staged_token,
            content=b"smoke staged content",
            filename=f"smoke_{smoke_id}.pdf",
            document_type="naac_requirement",
        )
        staged_payload = state_store.get_staged_upload(staged_token)

        state_store.set_ingestion_status(
            status_file_path,
            {
                "file_path": status_file_path,
                "status": "queued",
                "message": "Smoke ingestion queued",
            },
        )
        status_payload = state_store.get_ingestion_statuses([status_file_path])
        active_flag = state_store.has_active_ingestion(["queued", "processing"])

        removed_staged = state_store.remove_staged_upload(staged_token)
        state_store.remove_ingestion_status(status_file_path)
        status_after_remove = state_store.get_ingestion_statuses([status_file_path])

        state_ok = (
            staged_payload is not None
            and staged_payload.get("content") == b"smoke staged content"
            and status_file_path in status_payload
            and active_flag is True
            and removed_staged is not None
            and status_file_path not in status_after_remove
        )
        results.append(
            CheckResult(
                "ServerlessStateStore CRUD",
                state_ok,
                (
                    f"staged_found={staged_payload is not None} "
                    f"status_found={status_file_path in status_payload} "
                    f"active={active_flag} removed={removed_staged is not None}"
                ),
            )
        )

    except Exception as exc:
        results.append(CheckResult("ServerlessStateStore", False, str(exc)))

    try:
        memory_store = ConversationMemoryStore(
            db_url=supabase_url,
            embedding_model=settings.embedding_model,
            embedding_provider=settings.embedding_provider,
            embedding_dim=settings.embedding_dim,
            embedding_device=settings.embedding_device,
            short_ttl_days=7,
            long_ttl_days=365,
            short_limit=20,
            long_top_k=6,
        )
        memory_store.initialize_schema()

        memory_store.add_messages(
            memory_identity,
            [
                {
                    "role": "user",
                    "content": "Smoke memory message one about NAAC requirements.",
                    "metadata": {"source": "smoke"},
                },
                {
                    "role": "assistant",
                    "content": "Smoke memory response about institutional evidence.",
                    "metadata": {"source": "smoke"},
                },
            ],
        )

        with psycopg2.connect(supabase_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM conversation_memory_short
                    WHERE tenant_id = %s AND user_id = %s AND conversation_id = %s;
                    """,
                    (memory_identity.tenant_id, memory_identity.user_id, memory_identity.conversation_id),
                )
                identity_short_count = int(cur.fetchone()[0])

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM conversation_memory_long
                    WHERE tenant_id = %s AND user_id = %s AND conversation_id = %s;
                    """,
                    (memory_identity.tenant_id, memory_identity.user_id, memory_identity.conversation_id),
                )
                identity_long_count = int(cur.fetchone()[0])

                cur.execute(
                    """
                    SELECT role, content
                    FROM conversation_memory_long
                    WHERE tenant_id = %s AND user_id = %s AND conversation_id = %s
                      AND expires_at > now()
                    ORDER BY created_at DESC
                    LIMIT 6;
                    """,
                    (memory_identity.tenant_id, memory_identity.user_id, memory_identity.conversation_id),
                )
                plain_rows = cur.fetchall()

                query_embedding = memory_store.embedder.encode(["institutional evidence"], normalize_embeddings=False)[0]
                emb_str = memory_store._to_vector_literal(query_embedding)
                cur.execute(
                    """
                    WITH scoped AS MATERIALIZED (
                        SELECT role, content, metadata, created_at, embedding
                        FROM conversation_memory_long
                        WHERE tenant_id = %s AND user_id = %s AND conversation_id = %s
                          AND expires_at > now()
                    )
                    SELECT role, content, (embedding <=> %s::vector) AS distance
                    FROM scoped
                    ORDER BY distance
                    LIMIT 6;
                    """,
                    (
                        memory_identity.tenant_id,
                        memory_identity.user_id,
                        memory_identity.conversation_id,
                        emb_str,
                    ),
                )
                distance_rows = cur.fetchall()

        memory_context = memory_store.get_context(memory_identity, "institutional evidence")
        memory_health = memory_store.get_health()

        short_len = len(memory_context.get("short_term", []))
        long_len = len(memory_context.get("long_term", []))
        memory_ok = (
            short_len >= 2
            and long_len >= 1
            and identity_short_count >= 2
            and identity_long_count >= 2
            and bool(memory_health.get("ok"))
        )

        results.append(
            CheckResult(
                "ConversationMemoryStore CRUD",
                memory_ok,
                (
                    f"context_short={short_len} context_long={long_len} "
                    f"db_short={identity_short_count} db_long={identity_long_count} "
                    f"plain_rows={len(plain_rows)} distance_rows={len(distance_rows)} "
                    f"health_ok={memory_health.get('ok')}"
                ),
            )
        )

    except Exception as exc:
        results.append(CheckResult("ConversationMemoryStore", False, str(exc)))

    # Cleanup smoke rows.
    try:
        with psycopg2.connect(supabase_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {settings.supabase_table}
                    WHERE (metadata->>'file_hash') IN (%s, %s);
                    """,
                    (naac_file_hash, mvsr_file_hash),
                )
                cur.execute(
                    """
                    DELETE FROM conversation_memory_short
                    WHERE tenant_id = %s AND user_id = %s AND conversation_id = %s;
                    """,
                    (memory_identity.tenant_id, memory_identity.user_id, memory_identity.conversation_id),
                )
                cur.execute(
                    """
                    DELETE FROM conversation_memory_long
                    WHERE tenant_id = %s AND user_id = %s AND conversation_id = %s;
                    """,
                    (memory_identity.tenant_id, memory_identity.user_id, memory_identity.conversation_id),
                )
                cur.execute("DELETE FROM staged_uploads WHERE token = %s;", (staged_token,))
                cur.execute("DELETE FROM ingestion_status WHERE file_path = %s;", (status_file_path,))
            conn.commit()
        results.append(CheckResult("Cleanup", True, "Smoke rows removed"))
    except Exception as exc:
        results.append(CheckResult("Cleanup", False, str(exc)))

    return results


def main() -> int:
    try:
        results = run_checks()
    except Exception:
        print("Unexpected failure in smoke check")
        traceback.print_exc()
        return 1

    print("\n== Results ==")
    failed = 0
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name} :: {result.details}")
        if not result.ok:
            failed += 1

    print(f"\nSummary: {len(results) - failed}/{len(results)} checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
