"""
Database Loader Module for Indian Supreme Court RAG Pipeline
Official Ingestion Loader for Cloud SQL PostgreSQL 16 (database: "scrag", schema: public)
and Local SQLite Dev/Test environments.

Author: Senior Python + PostgreSQL Ingestion Engineer
"""

import os
import sys
import re
import json
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine, Connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DBLoader] %(message)s"
)
logger = logging.getLogger("DBLoader")

# Default Embedding Model Constants
DEFAULT_MODEL_ID = "text-embedding-004"
DEFAULT_PROVIDER = "google_vertex"
DEFAULT_DIMENSIONS = 768
DEFAULT_MODEL_VERSION = "v1"


def clean_sha256(val: Optional[str]) -> Optional[str]:
    """Ensures a string is a strictly valid 64-character hex SHA-256 string, or returns None."""
    if val and isinstance(val, str):
        v = val.strip()
        if re.match(r"^[0-9A-Fa-f]{64}$", v):
            return v.lower()
    return None


def parse_date_safe(date_str: Optional[str], expected_year: Optional[int] = None) -> Optional[str]:
    """
    Safely parses freeform judgment date strings into strict ISO YYYY-MM-DD for PostgreSQL DATE type.
    Rejects historical lower-court dates mentioned in the body facts (Issue C3).
    """
    if not date_str or not isinstance(date_str, str):
        return None
    cleaned = re.sub(r"\s+", " ", date_str).strip()
    cleaned = re.sub(r"\b0th\b", "20th", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)

    # Try direct ISO match first (YYYY-MM-DD)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", cleaned)
    if m:
        dt_str = m.group(0)
        yr = int(m.group(1))
        if expected_year and (yr < expected_year - 1 or yr > expected_year + 1):
            return None
        return dt_str

    # Try dateutil parser
    try:
        from dateutil import parser as date_parser
        dt = date_parser.parse(cleaned, fuzzy=True)
        if expected_year and (dt.year < expected_year - 1 or dt.year > expected_year + 1):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def resolve_database_url(candidate_url: Optional[str] = None) -> Optional[str]:
    """
    Resolves the active DATABASE_URL from candidates, environment, or Google Secret Manager.
    Rejects placeholder values like '<db_user>'.
    """
    url = candidate_url or os.environ.get("DATABASE_URL")
    if url and "<" not in url and ">" not in url:
        return url

    # Attempt Google Cloud Secret Manager if candidate contains placeholders or is missing
    try:
        from google.cloud import secretmanager
        project = os.environ.get("GCP_PROJECT", "sc-judgments-rag")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project}/secrets/scj-database-url/versions/latest"
        response = client.access_secret_version(request={"name": name})
        secret_val = response.payload.data.decode("UTF-8").strip()
        if secret_val and "<" not in secret_val and ">" not in secret_val:
            logger.info("Successfully resolved DATABASE_URL from Secret Manager 'scj-database-url'.")
            return secret_val
    except Exception as e:
        logger.debug(f"Secret Manager resolution skipped/failed: {e}")

    return url if (url and "<" not in url and ">" not in url) else None


class DatabaseLoader:
    """
    Production Ingestion Loader matching legal_rag_cloudsql_postgresql_schema.sql.
    Executes in strict FK order across one transaction per document.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        vector_mode: Optional[str] = None,
        dry_run: bool = False
    ):
        self.dry_run = dry_run
        self.vector_mode = (vector_mode or os.environ.get("VECTOR_MODE", "external")).lower()
        if self.vector_mode not in ("pgvector", "external"):
            raise ValueError(f"Invalid VECTOR_MODE='{self.vector_mode}'. Must be 'pgvector' or 'external'.")

        self.db_optional = os.environ.get("DB_OPTIONAL", "false").lower() in ("true", "1")
        self.raw_db_url = resolve_database_url(db_url)
        self.engine: Optional[Engine] = None
        self.is_postgres = False
        self.is_sqlite = False
        self.has_chunk_id_col = False
        self.has_embedding_col = False

        self.report_metrics = {
            "embedding_models": {"inserted": 0, "updated": 0},
            "judgments": {"inserted": 0, "updated": 0},
            "source_files": {"inserted": 0, "updated": 0},
            "ingestion_runs": {"inserted": 0, "updated": 0},
            "extraction_runs": {"inserted": 0, "updated": 0},
            "source_pages": {"inserted": 0, "updated": 0},
            "chunks": {"inserted": 0, "updated": 0},
            "chunk_embeddings": {"inserted": 0, "updated": 0}
        }

        if not self.raw_db_url:
            msg = "DATABASE_URL is not configured or contains unpopulated placeholders."
            if self.db_optional:
                logger.warning(f"{msg} Operating in GCS-only mode because DB_OPTIONAL=true.")
                return
            logger.error(msg)
            raise RuntimeError(msg)

        self.is_sqlite = self.raw_db_url.startswith("sqlite")
        self.is_postgres = self.raw_db_url.startswith("postgresql") or self.raw_db_url.startswith("postgres")

        try:
            if self.is_postgres:
                self.engine = create_engine(
                    self.raw_db_url,
                    pool_size=int(os.environ.get("DB_POOL_SIZE", "10")),
                    max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "20")),
                    pool_recycle=1800,
                    pool_pre_ping=True
                )
            else:
                self.engine = create_engine(self.raw_db_url)
                # Enforce SQLite foreign keys on every connection (Issue H9)
                @sa.event.listens_for(self.engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()

                # Auto-initialize SQLite schema for test and local dev environments
                self.init_sqlite_schema()

            # Execute mandatory startup preflight inspection
            self.run_preflight()

            # Ensure embedding model seeded in database
            self.seed_embedding_model()

        except Exception as e:
            logger.exception(f"DatabaseLoader initialization failed: {e}")
            if self.db_optional:
                logger.warning("DB_OPTIONAL=true: Continuing despite database connection failure.")
                self.engine = None
            else:
                raise

    def init_sqlite_schema(self):
        """Creates tables conforming to the official schema in SQLite for local testing."""
        ddl = """
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
            chunk_id TEXT,
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
        with self.engine.begin() as conn:
            for statement in ddl.split(";"):
                stmt = statement.strip()
                if stmt:
                    conn.execute(text(stmt))

    def run_preflight(self):
        """
        Connects, verifies current_database() and current_user, inspects schema,
        checks required tables, vector extensions, and chunk column existence.
        Fails fast and clearly if schema invariants are not met.
        """
        logger.info("Starting database preflight check...")
        required_tables = [
            "embedding_models",
            "judgments",
            "source_files",
            "ingestion_runs",
            "extraction_runs",
            "source_pages",
            "chunks",
            "chunk_embeddings"
        ]

        with self.engine.connect() as conn:
            # 1. Print connection identity
            if self.is_postgres:
                res = conn.execute(text("SELECT current_database(), current_user, version();")).fetchone()
                db_name, db_user, pg_ver = res[0], res[1], res[2]
                logger.info(f"Connected to Cloud SQL PostgreSQL: DB='{db_name}', User='{db_user}'")
                logger.info(f"Database Engine Version: {pg_ver}")
            else:
                db_name, db_user = "sqlite_dev", "main"
                logger.info("Connected to SQLite local database.")

            # 2. Inspect existing tables in public schema
            insp = inspect(self.engine)
            existing_tables = set(insp.get_table_names(schema="public" if self.is_postgres else None))

            missing_tables = [t for t in required_tables if t not in existing_tables]
            if missing_tables:
                err_msg = (
                    f"Preflight Check FAILED: Missing required table(s) {missing_tables} "
                    f"in database '{db_name}'. Ensure legal_rag_cloudsql_postgresql_schema.sql has been applied."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            # 3. Inspect columns in 'chunks' table
            chunk_cols = {c["name"] for c in insp.get_columns("chunks", schema="public" if self.is_postgres else None)}
            self.has_chunk_id_col = "chunk_id" in chunk_cols
            self.has_embedding_col = "embedding" in chunk_cols

            logger.info(
                f"Schema Inspection: 'chunks' columns count={len(chunk_cols)}. "
                f"has_chunk_id={self.has_chunk_id_col}, has_embedding={self.has_embedding_col}"
            )

            # 4. Check Vector requirements if VECTOR_MODE == 'pgvector'
            if self.vector_mode == "pgvector":
                if self.is_postgres:
                    ext_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector';")).fetchone()
                    if not ext_check:
                        err_msg = (
                            "Preflight Check FAILED: VECTOR_MODE='pgvector' requested, but extension 'vector' "
                            "is not enabled in Cloud SQL. Run 'CREATE EXTENSION IF NOT EXISTS vector;'."
                        )
                        logger.error(err_msg)
                        raise RuntimeError(err_msg)

                if not self.has_embedding_col:
                    err_msg = (
                        "Preflight Check FAILED: VECTOR_MODE='pgvector' requested, but column 'chunks.embedding' "
                        "does not exist in table 'chunks'. Run 'ALTER TABLE chunks ADD COLUMN embedding vector(768);'."
                    )
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

            logger.info(f"Preflight Check PASSED: DB='{db_name}', User='{db_user}', VECTOR_MODE='{self.vector_mode}'.")

    def seed_embedding_model(self):
        """Seeds the standard Google Vertex AI embedding model in embedding_models."""
        if not self.engine:
            return

        with self.engine.begin() as conn:
            model_uuid = str(uuid.uuid4())
            if self.is_postgres:
                stmt = text("""
                    INSERT INTO embedding_models (
                        embedding_model_uuid, model_id, provider, version, dimensions, is_active
                    ) VALUES (
                        :uuid, :model_id, :provider, :version, :dim, TRUE
                    )
                    ON CONFLICT (provider, model_id, version) DO UPDATE SET
                        dimensions = EXCLUDED.dimensions,
                        is_active = TRUE;
                """)
            else:
                stmt = text("""
                    INSERT INTO embedding_models (
                        embedding_model_uuid, model_id, provider, version, dimensions, is_active
                    ) VALUES (
                        :uuid, :model_id, :provider, :version, :dim, 1
                    )
                    ON CONFLICT (provider, model_id, version) DO UPDATE SET
                        dimensions = excluded.dimensions,
                        is_active = 1;
                """)

            conn.execute(stmt, {
                "uuid": model_uuid,
                "model_id": DEFAULT_MODEL_ID,
                "provider": DEFAULT_PROVIDER,
                "version": DEFAULT_MODEL_VERSION,
                "dim": DEFAULT_DIMENSIONS
            })
            self.report_metrics["embedding_models"]["inserted"] += 1

    def get_embedding_model_uuid(self, conn: Connection) -> str:
        """Retrieves the active embedding_model_uuid for text-embedding-004."""
        row = conn.execute(text("""
            SELECT embedding_model_uuid FROM embedding_models
            WHERE provider = :provider AND model_id = :model_id AND version = :version
            LIMIT 1;
        """), {
            "provider": DEFAULT_PROVIDER,
            "model_id": DEFAULT_MODEL_ID,
            "version": DEFAULT_MODEL_VERSION
        }).fetchone()

        if row:
            return str(row[0])
        new_uuid = str(uuid.uuid4())
        return new_uuid

    def load_document_and_chunks(
        self,
        doc_json_path: str,
        chunks_json_path: str,
        gcs_base_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """Loads a document and its chunks JSON files from disk into Cloud SQL."""
        with open(doc_json_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        return self.load_document_and_chunks_data(doc_data, chunks_data, gcs_base_uri=gcs_base_uri)

    def load_document_and_chunks_data(
        self,
        doc_data: Dict[str, Any],
        chunks_data: List[Dict[str, Any]],
        gcs_base_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Core Atomic Ingestion Function.
        Executes strictly in FK order within a single transaction per document:
        1. judgments
        2. source_files
        3. ingestion_runs
        4. extraction_runs
        5. source_pages
        6. chunks (0-based chunk_index, page_start/page_end)
        7. chunk_embeddings
        """
        if not self.engine:
            if self.db_optional:
                return {"status": "SKIPPED_NO_DB"}
            raise RuntimeError("Database engine not initialized. Cannot load document.")

        doc_id = doc_data.get("document_id") or "UNKNOWN_DOC"
        file_sha256 = clean_sha256(doc_data.get("file_sha256"))
        file_path = doc_data.get("file_path") or f"{doc_id}.pdf"
        storage_path = gcs_base_uri or doc_data.get("storage_path") or f"gs://sc-judgments-rag-prod-data/raw_pdfs/{doc_id}.pdf"
        total_pages = doc_data.get("total_pages", 0)
        parser_used = doc_data.get("parser_used", "pymupdf")
        status = doc_data.get("status", "COMPLETED")
        meta = doc_data.get("aggregated_metadata") or {}

        # Pre-generate UUIDv4 primary keys in Python (schema has NO defaults)
        judgment_uuid = str(uuid.uuid4())
        source_file_uuid = str(uuid.uuid4())
        ingestion_run_uuid = str(uuid.uuid4())
        extraction_run_uuid = str(uuid.uuid4())

        # Clean metadata inputs
        m_yr = re.search(r"^(\d{4})", doc_id)
        expected_year = int(m_yr.group(1)) if m_yr else None
        parsed_date = parse_date_safe(meta.get("date_of_judgment"), expected_year)
        case_no = meta.get("case_number")
        if case_no and (case_no.strip().upper() == "S.C.R." or not re.search(r'\d+', case_no)):
            case_no = None
        appellant = meta.get("appellant")
        respondent = meta.get("respondent")
        cit_text = case_no or (f"{appellant} v. {respondent}" if (appellant and respondent) else doc_id)

        # Calculate file size safely
        file_size = 0
        if file_path and os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
            except OSError:
                file_size = 0

        # Dry Run reporting check
        if self.dry_run:
            logger.info(f"[DRY RUN] Simulating atomic transaction for doc_id='{doc_id}' ({len(chunks_data)} chunks)...")

        # Open transactional block - single transaction per document, rollback on error
        conn = self.engine.connect()
        trans = conn.begin()

        try:
            # ------------------------------------------------------------------
            # 1. embedding_models lookup
            # ------------------------------------------------------------------
            model_uuid = self.get_embedding_model_uuid(conn)

            # ------------------------------------------------------------------
            # 2. Upsert judgments
            # ------------------------------------------------------------------
            # Lookup existing UUID if present to maintain FK consistency
            existing_j = conn.execute(
                text("SELECT judgment_uuid FROM judgments WHERE judgment_id = :jid;"),
                {"jid": doc_id}
            ).fetchone()
            if existing_j:
                judgment_uuid = str(existing_j[0])

            upsert_j_sql = text("""
                INSERT INTO judgments (
                    judgment_uuid, judgment_id, court, case_number,
                    judgment_date, citation_text, legal_status, is_primary,
                    created_at, updated_at
                ) VALUES (
                    :j_uuid, :j_id, 'Supreme Court of India', :case_num,
                    :j_date, :cit_text, 'REPORTED', FALSE,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                ON CONFLICT (judgment_id) DO UPDATE SET
                    court = EXCLUDED.court,
                    case_number = COALESCE(EXCLUDED.case_number, judgments.case_number),
                    judgment_date = COALESCE(EXCLUDED.judgment_date, judgments.judgment_date),
                    citation_text = COALESCE(EXCLUDED.citation_text, judgments.citation_text),
                    updated_at = CURRENT_TIMESTAMP;
            """)
            conn.execute(upsert_j_sql, {
                "j_uuid": judgment_uuid,
                "j_id": doc_id,
                "case_num": case_no,
                "j_date": parsed_date,
                "cit_text": cit_text
            })
            self.report_metrics["judgments"]["inserted" if not existing_j else "updated"] += 1

            # ------------------------------------------------------------------
            # 3. Upsert source_files
            # ------------------------------------------------------------------
            # Enforce source_files_one_current_per_judgment_uq
            conn.execute(
                text("UPDATE source_files SET is_current = FALSE WHERE judgment_uuid = :j_uuid;"),
                {"j_uuid": judgment_uuid}
            )

            existing_sf = conn.execute(
                text("SELECT source_file_uuid FROM source_files WHERE judgment_uuid = :j_uuid AND version_number = 'v1.0';"),
                {"j_uuid": judgment_uuid}
            ).fetchone()
            if existing_sf:
                source_file_uuid = str(existing_sf[0])

            upsert_sf_sql = text("""
                INSERT INTO source_files (
                    source_file_uuid, judgment_uuid, version_number, is_current,
                    original_filename, storage_path, page_count, ingestion_status,
                    pipeline_version, content_sha256, file_size_bytes,
                    created_at, updated_at
                ) VALUES (
                    :sf_uuid, :j_uuid, 'v1.0', TRUE,
                    :orig_name, :store_path, :pg_count, :status,
                    :pipe_ver, :sha256, :size,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                ON CONFLICT (judgment_uuid, version_number) DO UPDATE SET
                    is_current = TRUE,
                    storage_path = EXCLUDED.storage_path,
                    page_count = EXCLUDED.page_count,
                    ingestion_status = EXCLUDED.ingestion_status,
                    pipeline_version = EXCLUDED.pipeline_version,
                    content_sha256 = COALESCE(EXCLUDED.content_sha256, source_files.content_sha256),
                    file_size_bytes = COALESCE(EXCLUDED.file_size_bytes, source_files.file_size_bytes),
                    updated_at = CURRENT_TIMESTAMP;
            """)
            conn.execute(upsert_sf_sql, {
                "sf_uuid": source_file_uuid,
                "j_uuid": judgment_uuid,
                "orig_name": Path(file_path).name if file_path else f"{doc_id}.pdf",
                "store_path": storage_path,
                "pg_count": total_pages,
                "status": status,
                "pipe_ver": parser_used,
                "sha256": file_sha256,
                "size": file_size
            })
            self.report_metrics["source_files"]["inserted" if not existing_sf else "updated"] += 1

            # ------------------------------------------------------------------
            # 4. Upsert ingestion_runs (LOG TABLE)
            # ------------------------------------------------------------------
            idempotency_key = f"{doc_id}_v1.0"
            existing_ir = conn.execute(
                text("SELECT ingestion_run_uuid FROM ingestion_runs WHERE idempotency_key = :ikey;"),
                {"ikey": idempotency_key}
            ).fetchone()
            if existing_ir:
                ingestion_run_uuid = str(existing_ir[0])

            upsert_ir_sql = text("""
                INSERT INTO ingestion_runs (
                    ingestion_run_uuid, source_file_uuid, pipeline_name,
                    processing_status, error_message, retry_count, max_attempts,
                    idempotency_key, worker_id, started_at, finished_at, created_at
                ) VALUES (
                    :ir_uuid, :sf_uuid, 'scj-data-ingestion-v1',
                    :status, :err, 0, 3,
                    :ikey, :worker, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    processing_status = EXCLUDED.processing_status,
                    error_message = EXCLUDED.error_message,
                    finished_at = CURRENT_TIMESTAMP;
            """)
            conn.execute(upsert_ir_sql, {
                "ir_uuid": ingestion_run_uuid,
                "sf_uuid": source_file_uuid,
                "status": status,
                "err": doc_data.get("error_message"),
                "ikey": idempotency_key,
                "worker": os.environ.get("HOSTNAME", "worker-cloudrun")
            })
            self.report_metrics["ingestion_runs"]["inserted" if not existing_ir else "updated"] += 1

            # ------------------------------------------------------------------
            # 5. Upsert extraction_runs (LOG TABLE)
            # ------------------------------------------------------------------
            existing_er = conn.execute(
                text("SELECT extraction_run_uuid FROM extraction_runs WHERE source_file_uuid = :sf_uuid AND pipeline_version = :pipe_ver;"),
                {"sf_uuid": source_file_uuid, "pipe_ver": parser_used}
            ).fetchone()

            if existing_er:
                extraction_run_uuid = str(existing_er[0])
                conn.execute(text("""
                    UPDATE extraction_runs
                    SET processing_status = :status, error_message = :err, finished_at = CURRENT_TIMESTAMP
                    WHERE extraction_run_uuid = :er_uuid;
                """), {
                    "er_uuid": extraction_run_uuid,
                    "status": status,
                    "err": doc_data.get("error_message")
                })
                self.report_metrics["extraction_runs"]["updated"] += 1
            else:
                conn.execute(text("""
                    INSERT INTO extraction_runs (
                        extraction_run_uuid, source_file_uuid, ingestion_run_uuid,
                        pipeline_version, processing_status, error_message,
                        retry_count, max_attempts, started_at, finished_at, created_at
                    ) VALUES (
                        :er_uuid, :sf_uuid, :ir_uuid,
                        :pipe_ver, :status, :err,
                        0, 3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    );
                """), {
                    "er_uuid": extraction_run_uuid,
                    "sf_uuid": source_file_uuid,
                    "ir_uuid": ingestion_run_uuid,
                    "pipe_ver": parser_used,
                    "status": status,
                    "err": doc_data.get("error_message")
                })
                self.report_metrics["extraction_runs"]["inserted"] += 1

            # ------------------------------------------------------------------
            # 6. Upsert source_pages
            # ------------------------------------------------------------------
            pages = doc_data.get("pages", [])
            ocr_pages = set(doc_data.get("ocr_applied_pages", []))
            for p in pages:
                p_num = max(1, int(p.get("original_page_number", 1)))
                p_uuid = str(uuid.uuid4())
                p_method = "tesseract_ocr" if p_num in ocr_pages else parser_used
                p_score = 0.70 if p_num in ocr_pages else 0.95
                p_content = "\n\n".join(c.get("content", "") for c in p.get("chunks", []))

                upsert_sp_sql = text("""
                    INSERT INTO source_pages (
                        source_page_uuid, source_file_uuid, page_number,
                        extraction_method, content, quality_score, created_at
                    ) VALUES (
                        :p_uuid, :sf_uuid, :p_num,
                        :p_method, :content, :score, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (source_file_uuid, page_number) DO UPDATE SET
                        content = EXCLUDED.content,
                        quality_score = EXCLUDED.quality_score;
                """)
                conn.execute(upsert_sp_sql, {
                    "p_uuid": p_uuid,
                    "sf_uuid": source_file_uuid,
                    "p_num": p_num,
                    "p_method": p_method,
                    "content": p_content,
                    "score": p_score
                })
                self.report_metrics["source_pages"]["inserted"] += 1

            # ------------------------------------------------------------------
            # 7. Upsert chunks (0-based chunk_index, page_start/page_end)
            # ------------------------------------------------------------------
            # Build dynamic column list matching preflight inspection
            chunk_insert_cols = [
                "chunk_uuid", "judgment_uuid", "source_file_uuid",
                "extraction_run_uuid", "last_seen_ingestion_run_uuid",
                "chunk_index", "page_start", "page_end", "content",
                "section", "metadata", "content_sha256", "is_active",
                "created_at", "updated_at"
            ]
            chunk_val_placeholders = [
                ":chunk_uuid", ":judgment_uuid", ":source_file_uuid",
                ":extraction_run_uuid", ":last_seen_ingestion_run_uuid",
                ":chunk_index", ":page_start", ":page_end", ":content",
                ":section", ":metadata", ":content_sha256", ":is_active",
                "CURRENT_TIMESTAMP", "CURRENT_TIMESTAMP"
            ]

            if self.has_chunk_id_col:
                chunk_insert_cols.append("chunk_id")
                chunk_val_placeholders.append(":chunk_id")

            if self.vector_mode == "pgvector" and self.has_embedding_col:
                chunk_insert_cols.append("embedding")
                chunk_val_placeholders.append(":embedding")

            cols_str = ", ".join(chunk_insert_cols)
            vals_str = ", ".join(chunk_val_placeholders)

            upsert_chunk_sql = text(f"""
                INSERT INTO chunks ({cols_str})
                VALUES ({vals_str})
                ON CONFLICT (source_file_uuid, chunk_index) DO UPDATE SET
                    page_start = EXCLUDED.page_start,
                    page_end = EXCLUDED.page_end,
                    content = EXCLUDED.content,
                    section = EXCLUDED.section,
                    metadata = EXCLUDED.metadata,
                    content_sha256 = EXCLUDED.content_sha256,
                    is_active = EXCLUDED.is_active,
                    {"embedding = EXCLUDED.embedding," if (self.vector_mode == "pgvector" and self.has_embedding_col) else ""}
                    updated_at = CURRENT_TIMESTAMP;
            """)

            chunk_records = []
            chunk_embed_records = []

            for global_idx, c in enumerate(chunks_data):
                # 0-BASED GLOBAL CHUNK INDEX (Invariant from prompt)
                chunk_idx = global_idx
                page_start = max(1, int(c.get("original_page_number", 1)))
                page_end = max(page_start, int(c.get("page_end", page_start)))
                content_text = c.get("content", "")
                c_sha = clean_sha256(hashlib.sha256(content_text.encode("utf-8")).hexdigest())

                # String chunk_id format (e.g. 1950_1_15_25_EN_p1_c1)
                str_chunk_id = c.get("chunk_id") or f"{doc_id}_p{page_start}_c{chunk_idx}"

                # Ensure metadata is strictly a JSON object (dict)
                chunk_meta = {
                    "chunk_id": str_chunk_id,
                    "case_number": case_no,
                    "appellant": appellant,
                    "respondent": respondent,
                    "judges": meta.get("judges", []),
                    "date_of_judgment": parsed_date or meta.get("date_of_judgment"),
                    "section_heading": c.get("section_heading")
                }
                meta_json = json.dumps(chunk_meta)

                # Lookup existing chunk_uuid by natural key (source_file_uuid, chunk_index)
                existing_c = conn.execute(text("""
                    SELECT chunk_uuid FROM chunks
                    WHERE source_file_uuid = :sf_uuid AND chunk_index = :c_idx;
                """), {"sf_uuid": source_file_uuid, "c_idx": chunk_idx}).fetchone()

                c_uuid = str(existing_c[0]) if existing_c else str(uuid.uuid4())

                # Vector handling
                emb_val = c.get("embedding")
                emb_str = json.dumps(emb_val) if (isinstance(emb_val, list) and len(emb_val) == DEFAULT_DIMENSIONS) else None

                rec = {
                    "chunk_uuid": c_uuid,
                    "judgment_uuid": judgment_uuid,
                    "source_file_uuid": source_file_uuid,
                    "extraction_run_uuid": extraction_run_uuid,
                    "last_seen_ingestion_run_uuid": ingestion_run_uuid,
                    "chunk_index": chunk_idx,
                    "page_start": page_start,
                    "page_end": page_end,
                    "content": content_text,
                    "section": c.get("section_heading"),
                    "metadata": meta_json,
                    "content_sha256": c_sha,
                    "is_active": True
                }

                if self.has_chunk_id_col:
                    rec["chunk_id"] = str_chunk_id

                if self.vector_mode == "pgvector" and self.has_embedding_col:
                    rec["embedding"] = emb_str

                chunk_records.append(rec)

                # --------------------------------------------------------------
                # 8. Prepare chunk_embeddings records
                # --------------------------------------------------------------
                ce_uuid = str(uuid.uuid4())
                external_uri = f"{storage_path}#{str_chunk_id}"

                chunk_embed_records.append({
                    "ce_uuid": ce_uuid,
                    "chunk_uuid": c_uuid,
                    "model_uuid": model_uuid,
                    "provider": DEFAULT_PROVIDER,
                    "emb_ref": external_uri if (self.vector_mode == "external" or not self.has_embedding_col) else None,
                    "store": "pgvector" if self.vector_mode == "pgvector" else "gcs_external",
                    "status": "SYNCED" if emb_val else "PENDING"
                })

            # Execute chunks in batches of 200
            for i in range(0, len(chunk_records), 200):
                conn.execute(upsert_chunk_sql, chunk_records[i:i + 200])

            self.report_metrics["chunks"]["inserted"] += len(chunk_records)

            # ------------------------------------------------------------------
            # Execute chunk_embeddings upserts
            # ------------------------------------------------------------------
            upsert_ce_sql = text("""
                INSERT INTO chunk_embeddings (
                    chunk_embedding_uuid, chunk_uuid, embedding_model_uuid,
                    provider, embedding_ref, vector_store, sync_status,
                    last_synced_at, created_at, updated_at
                ) VALUES (
                    :ce_uuid, :chunk_uuid, :model_uuid,
                    :provider, :emb_ref, :store, :status,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                ON CONFLICT (chunk_uuid, embedding_model_uuid) DO UPDATE SET
                    embedding_ref = COALESCE(EXCLUDED.embedding_ref, chunk_embeddings.embedding_ref),
                    vector_store = EXCLUDED.vector_store,
                    sync_status = EXCLUDED.sync_status,
                    last_synced_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP;
            """)

            for i in range(0, len(chunk_embed_records), 200):
                conn.execute(upsert_ce_sql, chunk_embed_records[i:i + 200])

            self.report_metrics["chunk_embeddings"]["inserted"] += len(chunk_embed_records)

            # Dry Run vs Real Commit
            if self.dry_run:
                trans.rollback()
                logger.info(f"[DRY RUN] Rolled back transaction for doc_id='{doc_id}'. Zero rows written.")
            else:
                trans.commit()
                logger.info(f"Successfully committed doc_id='{doc_id}' into 7 tables ({len(chunk_records)} chunks).")

            return {
                "document_id": doc_id,
                "status": "COMPLETED",
                "judgment_uuid": judgment_uuid,
                "source_file_uuid": source_file_uuid,
                "chunks_count": len(chunks_data),
                "pages_count": total_pages
            }

        except Exception as e:
            # Transaction rollback on any error
            trans.rollback()
            logger.exception(f"FATAL: Database transaction failed for doc_id='{doc_id}': {e}")

            # Record failure in ingestion_runs if possible
            try:
                with self.engine.begin() as err_conn:
                    err_conn.execute(text("""
                        UPDATE ingestion_runs
                        SET processing_status = 'FAILED', error_message = :msg, finished_at = CURRENT_TIMESTAMP
                        WHERE idempotency_key = :ikey;
                    """), {"msg": str(e)[:1000], "ikey": f"{doc_id}_v1.0"})
            except Exception as log_err:
                logger.error(f"Could not record FAILED status in ingestion_runs: {log_err}")

            if self.db_optional:
                logger.warning(f"DB_OPTIONAL=true: Swallowing error for doc_id='{doc_id}'.")
                return {"document_id": doc_id, "status": "FAILED", "error": str(e)}

            # REMOVE SILENT FALLBACK: Exit non-zero
            raise RuntimeError(f"Database write failed for {doc_id}: {e}") from e

        finally:
            conn.close()

    def load_all_from_output(self, output_dir: str = "output") -> Dict[str, Any]:
        """Scans output directory and loads all documents sequentially in transactions."""
        docs_dir = Path(output_dir) / "documents"
        chunks_dir = Path(output_dir) / "chunks"

        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        doc_files = list(docs_dir.glob("*.json"))
        loaded_count = 0
        total_chunks = 0
        failures = []

        logger.info(f"Scanning {len(doc_files)} document summaries from '{output_dir}'...")

        for doc_file in doc_files:
            doc_id = doc_file.stem
            chunk_file = chunks_dir / f"{doc_id}_chunks.json"

            if not chunk_file.exists():
                logger.warning(f"Missing chunks file for {doc_id}: {chunk_file}")
                failures.append({"document_id": doc_id, "error": "Missing chunks file"})
                continue

            try:
                res = self.load_document_and_chunks(str(doc_file), str(chunk_file))
                if res.get("status") == "COMPLETED":
                    loaded_count += 1
                    total_chunks += res.get("chunks_count", 0)
                else:
                    failures.append({"document_id": doc_id, "error": res.get("error")})
            except Exception as e:
                failures.append({"document_id": doc_id, "error": str(e)})
                if not self.db_optional:
                    raise

        self.print_final_report()

        return {
            "total_documents_found": len(doc_files),
            "documents_loaded": loaded_count,
            "total_chunks_loaded": total_chunks,
            "failed_count": len(failures),
            "failures": failures
        }

    def print_final_report(self):
        """Prints a structured final report of rows inserted/updated per table."""
        print("\n" + "=" * 80)
        print(" DATABASE INGESTION LOADER — FINAL REPORT")
        print("=" * 80)
        print(f"{'Table Name':<25} | {'Rows Inserted':<15} | {'Rows Updated':<15}")
        print("-" * 80)
        for tbl, counts in self.report_metrics.items():
            print(f"{tbl:<25} | {counts['inserted']:<15} | {counts['updated']:<15}")
        print("=" * 80)

    def get_stats(self) -> Dict[str, Any]:
        """Queries row counts across all tables for validation."""
        if not self.engine:
            return {"status": "NO_DATABASE_CONNECTED"}

        with self.engine.connect() as conn:
            j_cnt = conn.execute(text("SELECT COUNT(*) FROM judgments;")).scalar() or 0
            sf_cnt = conn.execute(text("SELECT COUNT(*) FROM source_files;")).scalar() or 0
            c_cnt = conn.execute(text("SELECT COUNT(*) FROM chunks;")).scalar() or 0
            ce_cnt = conn.execute(text("SELECT COUNT(*) FROM chunk_embeddings;")).scalar() or 0
            ir_cnt = conn.execute(text("SELECT COUNT(*) FROM ingestion_runs;")).scalar() or 0
            er_cnt = conn.execute(text("SELECT COUNT(*) FROM extraction_runs;")).scalar() or 0
            sp_cnt = conn.execute(text("SELECT COUNT(*) FROM source_pages;")).scalar() or 0
            em_cnt = conn.execute(text("SELECT COUNT(*) FROM embedding_models;")).scalar() or 0

            # Sample chunk for inspection
            sample = conn.execute(text("""
                SELECT chunk_uuid, page_start, section, substr(content, 1, 100) AS snippet, metadata
                FROM chunks LIMIT 1;
            """)).mappings().first()

            sample_dict = dict(sample) if sample else None
            if sample_dict:
                meta = json.loads(sample_dict.get("metadata") or "{}") if isinstance(sample_dict.get("metadata"), str) else (sample_dict.get("metadata") or {})
                sample_dict["chunk_id"] = meta.get("chunk_id", str(sample_dict.get("chunk_uuid")))

            return {
                "judgments": j_cnt,
                "total_documents": j_cnt,
                "total_judgments": j_cnt,
                "source_files": sf_cnt,
                "total_source_files": sf_cnt,
                "chunks": c_cnt,
                "total_chunks": c_cnt,
                "chunk_embeddings": ce_cnt,
                "ingestion_runs": ir_cnt,
                "extraction_runs": er_cnt,
                "source_pages": sp_cnt,
                "embedding_models": em_cnt,
                "sample_chunk": sample_dict
            }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingestion Database Loader for Supreme Court RAG")
    parser.add_argument("--db", default=None, help="Database connection URL")
    parser.add_argument("--output-dir", default="output", help="Directory containing JSON deliverables")
    parser.add_argument("--vector-mode", choices=["pgvector", "external"], default=None,
                        help="Vector storage mode: 'pgvector' or 'external'")
    parser.add_argument("--dry-run", action="store_true", help="Simulate SQL execution without committing")
    parser.add_argument("--stats", action="store_true", help="Print table counts from database")
    args = parser.parse_args()

    loader = DatabaseLoader(db_url=args.db, vector_mode=args.vector_mode, dry_run=args.dry_run)

    if args.stats:
        stats = loader.get_stats()
        print("\n=== Current Database Verification Counts ===")
        print(json.dumps(stats, indent=2))
        return

    summary = loader.load_all_from_output(args.output_dir)
    print("\nBatch Ingestion Completed:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
