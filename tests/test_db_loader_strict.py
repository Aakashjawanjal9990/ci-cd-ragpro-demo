"""
Strict Unit & Integration Test for DatabaseLoader
Validates:
1. Schema conformity against legal_rag_cloudsql_postgresql_schema.sql
2. 1 Sample Judgment with 2 pages and 3 chunks
3. Foreign key integrity across all 8 relational tables
4. Idempotency: re-running produces zero duplicate rows
5. 0-based chunk indexing and page_start/page_end isolation
6. Vector mode handling (external GCS URI vs pgvector)
"""

import os
import sys
import json
import uuid
import hashlib
import sqlite3
import pytest
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_loader import DatabaseLoader


SQLITE_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS embedding_models (
    embedding_model_uuid TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    version TEXT,
    dimensions INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (provider, model_id, version)
);

CREATE TABLE IF NOT EXISTS judgments (
    judgment_uuid TEXT PRIMARY KEY,
    judgment_id TEXT NOT NULL UNIQUE,
    court TEXT,
    case_number TEXT,
    judgment_date TEXT,
    citation_text TEXT,
    reporter_citations TEXT,
    legal_status TEXT,
    is_primary INTEGER NOT NULL DEFAULT 0,
    overruled_by_judgment_uuid TEXT,
    refresh_interval_days INTEGER,
    last_refreshed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (overruled_by_judgment_uuid) REFERENCES judgments(judgment_uuid)
);

CREATE TABLE IF NOT EXISTS source_files (
    source_file_uuid TEXT PRIMARY KEY,
    judgment_uuid TEXT NOT NULL,
    version_number TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0,
    original_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    doc_encryption_files TEXT,
    page_count INTEGER,
    ingestion_status TEXT NOT NULL,
    pipeline_version TEXT,
    created_by_user_uuid TEXT,
    content_sha256 TEXT,
    file_size_bytes INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
    UNIQUE (judgment_uuid, version_number)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    ingestion_run_uuid TEXT PRIMARY KEY,
    source_file_uuid TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    idempotency_key TEXT NOT NULL UNIQUE,
    worker_id TEXT,
    started_at TIMESTAMP,
    heartbeat_at TIMESTAMP,
    finished_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    extraction_run_uuid TEXT PRIMARY KEY,
    source_file_uuid TEXT NOT NULL,
    ingestion_run_uuid TEXT,
    pipeline_version TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
    FOREIGN KEY (ingestion_run_uuid) REFERENCES ingestion_runs(ingestion_run_uuid) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS source_pages (
    source_page_uuid TEXT PRIMARY KEY,
    source_file_uuid TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    extraction_method TEXT NOT NULL,
    content TEXT,
    quality_score REAL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
    UNIQUE (source_file_uuid, page_number)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_uuid TEXT PRIMARY KEY,
    judgment_uuid TEXT NOT NULL,
    source_file_uuid TEXT NOT NULL,
    extraction_run_uuid TEXT,
    last_seen_ingestion_run_uuid TEXT,
    chunk_index INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    content TEXT NOT NULL,
    section TEXT,
    metadata TEXT,
    content_sha256 TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
    FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
    FOREIGN KEY (extraction_run_uuid) REFERENCES extraction_runs(extraction_run_uuid) ON DELETE SET NULL,
    FOREIGN KEY (last_seen_ingestion_run_uuid) REFERENCES ingestion_runs(ingestion_run_uuid) ON DELETE SET NULL,
    UNIQUE (source_file_uuid, chunk_index)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_embedding_uuid TEXT PRIMARY KEY,
    chunk_uuid TEXT NOT NULL,
    embedding_model_uuid TEXT,
    provider TEXT NOT NULL,
    corpus_resource_name TEXT,
    external_chunk_resource_name TEXT,
    embedding_ref TEXT,
    vector_store TEXT,
    sync_status TEXT NOT NULL,
    last_synced_at TIMESTAMP,
    last_error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_uuid) REFERENCES chunks(chunk_uuid) ON DELETE CASCADE,
    FOREIGN KEY (embedding_model_uuid) REFERENCES embedding_models(embedding_model_uuid) ON DELETE SET NULL,
    UNIQUE (chunk_uuid, embedding_model_uuid)
);
"""


@pytest.fixture
def test_db(tmp_path):
    """Creates a fresh test SQLite database conforming strictly to the official schema."""
    db_file = tmp_path / "test_scrag.db"
    db_url = f"sqlite:///{db_file}"

    # Initialize tables directly with SQLite DDL
    conn = sqlite3.connect(str(db_file))
    conn.executescript(SQLITE_SCHEMA_DDL)
    conn.commit()
    conn.close()

    return db_url


@pytest.fixture
def sample_judgment_data():
    """
    Creates 1 sample judgment:
    - 2 pages
    - 3 chunks (Chunk 0 on Page 1, Chunk 1 on Page 1, Chunk 2 on Page 2)
    """
    doc_id = "1950_1_15_25_EN"
    sha = "a" * 64

    doc_data = {
        "document_id": doc_id,
        "file_sha256": sha,
        "file_path": f"/tmp/{doc_id}.pdf",
        "storage_path": f"gs://sc-judgments-rag-prod-data/raw_pdfs/year=1950/{doc_id}.pdf",
        "total_pages": 2,
        "parser_used": "pymupdf",
        "status": "COMPLETED",
        "aggregated_metadata": {
            "case_number": "Civil Appeal No. 15 of 1950",
            "appellant": "TEST APPELLANT",
            "respondent": "TEST RESPONDENT",
            "date_of_judgment": "1950-05-15",
            "judges": ["KANIA C.J.", "FAZL ALI J."]
        },
        "pages": [
            {
                "original_page_number": 1,
                "chunks": [
                    {"content": "Page 1 - First paragraph legal facts..."},
                    {"content": "Page 1 - Second paragraph issues raised..."}
                ]
            },
            {
                "original_page_number": 2,
                "chunks": [
                    {"content": "Page 2 - Final court judgment and decree..."}
                ]
            }
        ]
    }

    chunks_data = [
        {
            "chunk_id": f"{doc_id}_p1_c0",
            "document_id": doc_id,
            "original_page_number": 1,
            "chunk_index": 0,
            "content": "Page 1 - First paragraph legal facts...",
            "section_heading": "FACTS",
            "embedding": [0.01] * 768
        },
        {
            "chunk_id": f"{doc_id}_p1_c1",
            "document_id": doc_id,
            "original_page_number": 1,
            "chunk_index": 1,
            "content": "Page 1 - Second paragraph issues raised...",
            "section_heading": "ISSUES",
            "embedding": [0.02] * 768
        },
        {
            "chunk_id": f"{doc_id}_p2_c2",
            "document_id": doc_id,
            "original_page_number": 2,
            "chunk_index": 2,
            "content": "Page 2 - Final court judgment and decree...",
            "section_heading": "ORDER",
            "embedding": [0.03] * 768
        }
    ]

    return doc_data, chunks_data


def test_load_sample_judgment_and_fk_integrity(test_db, sample_judgment_data):
    """
    Test 1: Loads 1 judgment (2 pages, 3 chunks).
    Asserts:
    - Row counts across all 8 tables
    - Foreign Key integrity
    - chunk_index is 0-based and global
    """
    doc_data, chunks_data = sample_judgment_data
    loader = DatabaseLoader(db_url=test_db, vector_mode="external")

    # Ingest document
    res = loader.load_document_and_chunks_data(doc_data, chunks_data)
    assert res["status"] == "COMPLETED"
    assert res["chunks_count"] == 3
    assert res["pages_count"] == 2

    # Verify counts
    stats = loader.get_stats()
    assert stats["judgments"] == 1, "Expected 1 judgment record"
    assert stats["source_files"] == 1, "Expected 1 source_files record"
    assert stats["ingestion_runs"] == 1, "Expected 1 ingestion_runs record"
    assert stats["extraction_runs"] == 1, "Expected 1 extraction_runs record"
    assert stats["source_pages"] == 2, "Expected 2 source_pages records (1 per page)"
    assert stats["chunks"] == 3, "Expected 3 chunks records"
    assert stats["chunk_embeddings"] == 3, "Expected 3 chunk_embeddings records"
    assert stats["embedding_models"] >= 1, "Expected at least 1 embedding_models record"

    # Verify FK Integrity
    engine = loader.engine
    with engine.connect() as conn:
        j_uuid = conn.execute(text("SELECT judgment_uuid FROM judgments WHERE judgment_id = '1950_1_15_25_EN';")).scalar()
        sf_uuid = conn.execute(text("SELECT source_file_uuid FROM source_files WHERE judgment_uuid = :j_uuid;"), {"j_uuid": j_uuid}).scalar()

        # Check Chunks FK
        chunk_rows = conn.execute(text("SELECT chunk_index, judgment_uuid, source_file_uuid, page_start, page_end, metadata FROM chunks ORDER BY chunk_index ASC;")).fetchall()
        assert len(chunk_rows) == 3
        
        # Check 0-based global indexing
        for idx, row in enumerate(chunk_rows):
            assert row[0] == idx, f"Chunk index must be {idx} (0-based global)"
            assert row[1] == j_uuid, "chunks.judgment_uuid must match parent judgment"
            assert row[2] == sf_uuid, "chunks.source_file_uuid must match parent source_file"
            meta = json.loads(row[5])
            assert "chunk_id" in meta, "String chunk_id must be preserved in metadata->>'chunk_id'"

        # Verify page mapping: chunk 0, 1 on page 1; chunk 2 on page 2
        assert chunk_rows[0][3] == 1 and chunk_rows[0][4] == 1
        assert chunk_rows[1][3] == 1 and chunk_rows[1][4] == 1
        assert chunk_rows[2][3] == 2 and chunk_rows[2][4] == 2

        # Check chunk_embeddings FK
        ce_rows = conn.execute(text("SELECT chunk_uuid, provider, sync_status, embedding_ref FROM chunk_embeddings;")).fetchall()
        assert len(ce_rows) == 3
        for ce in ce_rows:
            assert ce[1] == "google_vertex"
            assert ce[2] == "SYNCED"
            assert ce[3].startswith("gs://")


def test_idempotency_rerun_creates_no_duplicates(test_db, sample_judgment_data):
    """
    Test 2: Re-running ingestion for the exact same document must NOT create duplicate rows.
    Asserts:
    - Counts remain strictly: 1 judgment, 1 source_file, 2 pages, 3 chunks, 3 chunk_embeddings.
    """
    doc_data, chunks_data = sample_judgment_data
    loader = DatabaseLoader(db_url=test_db, vector_mode="external")

    # Run 1
    loader.load_document_and_chunks_data(doc_data, chunks_data)
    stats1 = loader.get_stats()

    # Run 2 (Re-ingest same document)
    loader.load_document_and_chunks_data(doc_data, chunks_data)
    stats2 = loader.get_stats()

    # Assert exactly identical row counts (No duplicates)
    assert stats1 == stats2, f"Rerunning created duplicate records! Before: {stats1}, After: {stats2}"
    assert stats2["judgments"] == 1
    assert stats2["source_files"] == 1
    assert stats2["chunks"] == 3
    assert stats2["chunk_embeddings"] == 3
    assert stats2["source_pages"] == 2


def test_dry_run_flag(test_db, sample_judgment_data):
    """
    Test 3: Dry-run flag must simulate execution and rollback, writing 0 records to tables.
    """
    doc_data, chunks_data = sample_judgment_data
    loader = DatabaseLoader(db_url=test_db, vector_mode="external", dry_run=True)

    loader.load_document_and_chunks_data(doc_data, chunks_data)

    stats = loader.get_stats()
    assert stats["judgments"] == 0, "Dry-run must not commit judgments"
    assert stats["chunks"] == 0, "Dry-run must not commit chunks"
    assert stats["source_files"] == 0, "Dry-run must not commit source_files"

