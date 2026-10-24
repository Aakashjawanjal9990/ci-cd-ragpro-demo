"""
Sync Deliverables from Google Cloud Storage into Cloud SQL PostgreSQL (scrag)
Reads pre-computed chunks and embeddings from:
    gs://sc-judgments-rag-prod-data/deliverables/chunks/year=YYYY/
and loads them directly into Cloud SQL PostgreSQL without re-parsing or re-embedding.

Usage:
    python sync_gcs_to_cloudsql.py --db "postgresql://postgres:PASSWORD@34.93.53.176:5432/scrag"
    python sync_gcs_to_cloudsql.py --db "postgresql://postgres:PASSWORD@34.93.53.176:5432/scrag" --year 1950
"""

import os
import sys
import re
import json
import uuid
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("GCSToCloudSQLSync")


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

    # Try direct ISO match first
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", cleaned)
    if m:
        dt_str = m.group(0)
        yr = int(m.group(1))
        if expected_year and (yr < expected_year - 1 or yr > expected_year + 1):
            return None
        return dt_str

    try:
        from dateutil import parser as date_parser
        dt = date_parser.parse(cleaned, fuzzy=True)
        if expected_year and (dt.year < expected_year - 1 or dt.year > expected_year + 1):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


class GCSCloudSQLSyncer:
    def __init__(self, db_url: str, bucket_name: str = "sc-judgments-rag-prod-data", vector_mode: Optional[str] = None):
        self.db_url = db_url
        self.bucket_name = bucket_name
        self.vector_mode = (vector_mode or os.environ.get("VECTOR_MODE", "external")).lower()
        self.engine = create_engine(
            self.db_url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
            pool_pre_ping=True
        )
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.has_chunk_id_col = False
        self.has_embedding_col = False
        self.verify_connection()

    def verify_connection(self):
        """Verifies database connectivity, checks optional columns, and registers embedding model."""
        with self.engine.begin() as conn:
            res = conn.execute(text("SELECT current_database(), current_user;")).fetchone()
            logger.info(f"Connected to Cloud SQL PostgreSQL: DB='{res[0]}', User='{res[1]}'")
            
            # 1. Check if vector extension and optional columns exist
            try:
                insp = inspect(conn)
                table_names = set(insp.get_table_names())
                if "chunks" in table_names:
                    cols = {c["name"] for c in insp.get_columns("chunks")}
                    self.has_chunk_id_col = "chunk_id" in cols
                    self.has_embedding_col = "embedding" in cols and self.vector_mode == "pgvector"
                logger.info(f"Chunks schema capabilities: chunk_id column={self.has_chunk_id_col}, "
                            f"embedding column={self.has_embedding_col} (mode: {self.vector_mode})")
            except Exception as e:
                logger.warning(f"Inspection note: {e}")

            # 2. Seed Vertex AI text-embedding-004 in embedding_models
            try:
                model_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "scj.model.google_vertex.text-embedding-004.v1"))
                conn.execute(text("""
                    INSERT INTO embedding_models (
                        embedding_model_uuid, model_id, provider, version, dimensions, is_active
                    ) VALUES (
                        :m_uuid, 'text-embedding-004', 'google_vertex', 'v1', 768, TRUE
                    )
                    ON CONFLICT (provider, model_id, version) DO NOTHING;
                """), {"m_uuid": model_uuid})
                logger.info("Registered 'text-embedding-004' (768 dimensions) in embedding_models.")
            except Exception as e:
                logger.warning(f"Model registration note: {e}")

    def load_single_chunk_file(self, blob_item: Any) -> Dict[str, Any]:
        """Downloads a single _chunks.json from GCS and inserts into Cloud SQL."""
        try:
            if isinstance(blob_item, str):
                blob = self.bucket.blob(blob_item)
                blob_name = blob_item
                gcs_created_at = None
            else:
                blob = blob_item
                blob_name = blob_item.name
                gcs_created_at = getattr(blob_item, "time_created", None)

            ts_created = gcs_created_at or datetime.utcnow()

            chunks_text = blob.download_as_text(encoding="utf-8")
            chunks_data = json.loads(chunks_text)

            if not chunks_data or not isinstance(chunks_data, list):
                return {"status": "EMPTY", "blob": blob_name, "chunks": 0}

            first_chunk = chunks_data[0]
            doc_id = first_chunk.get("document_id") or Path(blob_name).stem.replace("_chunks", "")
            m_yr = re.search(r"^(\d{4})", doc_id)
            expected_year = int(m_yr.group(1)) if m_yr else None

            case_number = first_chunk.get("case_number")
            if case_number and (case_number.strip().upper() == "S.C.R." or not re.search(r'\d+', case_number)):
                case_number = None

            appellant = first_chunk.get("appellant")
            respondent = first_chunk.get("respondent")
            date_of_judgment = first_chunk.get("date_of_judgment")
            parsed_date = parse_date_safe(date_of_judgment, expected_year)
            judges = first_chunk.get("judges", [])
            
            # Generate deterministic UUIDs
            judgment_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"scj.judgment.{doc_id}"))
            source_file_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"scj.file.{doc_id}"))
            
            with self.engine.begin() as conn:
                # 1. Upsert judgments
                upsert_judgment = text("""
                    INSERT INTO judgments (
                        judgment_uuid, judgment_id, court, case_number, judgment_date,
                        citation_text, legal_status, is_primary
                    ) VALUES (
                        :j_uuid, :doc_id, 'Supreme Court of India', :case_number, :j_date,
                        :citation_text, 'REPORTED', TRUE
                    )
                    ON CONFLICT (judgment_id) DO UPDATE SET
                        case_number = COALESCE(EXCLUDED.case_number, judgments.case_number),
                        judgment_date = COALESCE(EXCLUDED.judgment_date, judgments.judgment_date),
                        citation_text = COALESCE(EXCLUDED.citation_text, judgments.citation_text),
                        updated_at = CURRENT_TIMESTAMP;
                """)
                conn.execute(upsert_judgment, {
                    "j_uuid": judgment_uuid,
                    "doc_id": doc_id,
                    "case_number": case_number,
                    "j_date": parsed_date,
                    "citation_text": f"{appellant} v. {respondent}" if appellant and respondent else doc_id
                })

                # 2. Upsert source_files
                upsert_file = text("""
                    INSERT INTO source_files (
                        source_file_uuid, judgment_uuid, version_number, is_current,
                        original_filename, storage_path, page_count, ingestion_status,
                        pipeline_version, created_at
                    ) VALUES (
                        :s_uuid, :j_uuid, 'v1.0', TRUE,
                        :filename, :storage_path, :page_count, 'COMPLETED',
                        'scj-pipeline-gcs-sync', :created_at
                    )
                    ON CONFLICT (judgment_uuid, version_number) DO UPDATE SET
                        storage_path = EXCLUDED.storage_path,
                        page_count = EXCLUDED.page_count,
                        ingestion_status = 'COMPLETED',
                        updated_at = CURRENT_TIMESTAMP;
                """)
                conn.execute(upsert_file, {
                    "s_uuid": source_file_uuid,
                    "j_uuid": judgment_uuid,
                    "filename": f"{doc_id}.pdf",
                    "storage_path": f"gs://{self.bucket_name}/{blob_name}",
                    "page_count": max((c.get("original_page_number", 1) for c in chunks_data), default=1),
                    "created_at": ts_created
                })

                # 3. Clean previous chunks for this file
                conn.execute(text("DELETE FROM chunks WHERE source_file_uuid = :s_uuid"), {"s_uuid": source_file_uuid})

                # 4. Insert chunks
                # Build dynamic SQL based on available columns
                chunk_columns = [
                    "chunk_uuid", "judgment_uuid", "source_file_uuid",
                    "chunk_index", "page_start", "page_end", "content", "section",
                    "metadata", "content_sha256", "is_active", "created_at"
                ]
                value_placeholders = [
                    ":c_uuid", ":j_uuid", ":s_uuid",
                    ":chunk_index", ":page_start", ":page_end", ":content", ":section",
                    ":metadata::jsonb", ":content_sha256", "TRUE", ":created_at"
                ]

                if self.has_chunk_id_col:
                    chunk_columns.append("chunk_id")
                    value_placeholders.append(":chunk_id")

                if self.has_embedding_col:
                    chunk_columns.append("embedding")
                    value_placeholders.append(":embedding")

                insert_chunk_sql = text(f"""
                    INSERT INTO chunks (
                        {', '.join(chunk_columns)}
                    ) VALUES (
                        {', '.join(value_placeholders)}
                    )
                    ON CONFLICT (source_file_uuid, chunk_index) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP;
                """)

                chunk_records = []
                chunk_embeddings_records = []
                model_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "scj.model.google_vertex.text-embedding-004.v1"))

                for idx, c in enumerate(chunks_data, 0):  # 0-based chunk_index
                    c_id = c.get("chunk_id") or f"{doc_id}_c{idx}"
                    c_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"scj.chunk.{c_id}"))
                    c_hash = hashlib.sha256(c["content"].encode("utf-8")).hexdigest()
                    page_num = c.get("original_page_number", 1)
                    
                    emb_val = c.get("embedding")
                    emb_str = json.dumps(emb_val) if (isinstance(emb_val, list) and self.has_embedding_col) else None

                    meta = {
                        "case_number": case_number,
                        "appellant": appellant,
                        "respondent": respondent,
                        "judges": judges,
                        "date_of_judgment": date_of_judgment,
                        "embedding_created_at": ts_created.isoformat()
                    }
                    if not self.has_chunk_id_col:
                        meta["chunk_id"] = c_id

                    rec = {
                        "c_uuid": c_uuid,
                        "j_uuid": judgment_uuid,
                        "s_uuid": source_file_uuid,
                        "chunk_index": idx,
                        "page_start": page_num,
                        "page_end": page_num,
                        "content": c["content"],
                        "section": c.get("section_heading"),
                        "metadata": json.dumps(meta),
                        "content_sha256": c_hash,
                        "created_at": ts_created
                    }
                    if self.has_chunk_id_col:
                        rec["chunk_id"] = c_id
                    if self.has_embedding_col:
                        rec["embedding"] = emb_str

                    chunk_records.append(rec)

                    # Chunk embeddings tracking
                    chunk_embeddings_records.append({
                        "ce_uuid": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"scj.embed.{c_id}")),
                        "chunk_uuid": c_uuid,
                        "model_uuid": model_uuid,
                        "provider": "google_vertex",
                        "vector_store": "pgvector" if self.has_embedding_col else "external_gcs",
                        "sync_status": "SYNCED",
                        "embedding_ref": f"gs://{self.bucket_name}/{blob_name}#{c_id}",
                        "created_at": ts_created,
                        "last_synced_at": datetime.utcnow()
                    })

                # Execute in batch
                for batch_start in range(0, len(chunk_records), 200):
                    conn.execute(insert_chunk_sql, chunk_records[batch_start:batch_start + 200])

                # 5. Populate chunk_embeddings registry
                if chunk_embeddings_records:
                    insert_ce_sql = text("""
                        INSERT INTO chunk_embeddings (
                            chunk_embedding_uuid, chunk_uuid, embedding_model_uuid,
                            provider, vector_store, sync_status, embedding_ref,
                            created_at, last_synced_at
                        ) VALUES (
                            :ce_uuid, :chunk_uuid, :model_uuid,
                            :provider, :vector_store, :sync_status, :embedding_ref,
                            :created_at, :last_synced_at
                        )
                        ON CONFLICT (chunk_uuid, embedding_model_uuid) DO UPDATE SET
                            sync_status = 'SYNCED',
                            embedding_ref = EXCLUDED.embedding_ref,
                            last_synced_at = EXCLUDED.last_synced_at,
                            updated_at = CURRENT_TIMESTAMP;
                    """)
                    for b_start in range(0, len(chunk_embeddings_records), 200):
                        conn.execute(insert_ce_sql, chunk_embeddings_records[b_start:b_start + 200])

            return {"status": "SUCCESS", "blob": blob_name, "chunks": len(chunks_data)}
        except Exception as e:
            logger.error(f"Error loading {blob_name}: {e}")
            return {"status": "FAILED", "blob": blob_name, "error": str(e)}

    def sync_all(self, year: Optional[int] = None, workers: int = 8):
        """Scans GCS deliverables and syncs into Cloud SQL."""
        prefix = f"deliverables/chunks/year={year}/" if year else "deliverables/chunks/"
        logger.info(f"Listing chunk files from gs://{self.bucket_name}/{prefix}...")

        blobs = [
            b for b in self.storage_client.list_blobs(self.bucket_name, prefix=prefix)
            if b.name.endswith("_chunks.json")
        ]

        logger.info(f"Found {len(blobs)} chunk JSON files in GCS. Starting parallel database sync ({workers} workers)...")

        total_chunks = 0
        successful_docs = 0
        failed_docs = 0
        failed_records = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_blob = {executor.submit(self.load_single_chunk_file, b): b for b in blobs}
            for future in as_completed(future_to_blob):
                res = future.result()
                if res.get("status") == "SUCCESS":
                    successful_docs += 1
                    total_chunks += res.get("chunks", 0)
                else:
                    failed_docs += 1
                    failed_records.append({"blob": res.get("blob"), "error": res.get("error")})

                if (successful_docs + failed_docs) % 50 == 0:
                    logger.info(f"Progress: {successful_docs + failed_docs}/{len(blobs)} documents synced ({total_chunks} chunks loaded)...")

        if failed_records:
            fail_log_path = Path("failed_sync_documents.json")
            with open(fail_log_path, "w", encoding="utf-8") as f:
                json.dump(failed_records, f, indent=2)
            logger.warning(f"Recorded {len(failed_records)} failed documents to {fail_log_path.resolve()}.")

        print("\n" + "=" * 80)
        print(" GCS TO CLOUDSQL SYNC COMPLETE")
        print("=" * 80)
        print(f"  • Total Documents Synced: {successful_docs}")
        print(f"  • Total Chunks Loaded:    {total_chunks}")
        print(f"  • Failed Documents:       {failed_docs}")
        print("=" * 80)

        # Print live database statistics
        with self.engine.connect() as conn:
            j_cnt = conn.execute(text("SELECT COUNT(*) FROM judgments")).scalar()
            c_cnt = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
            e_cnt = conn.execute(text("SELECT COUNT(*) FROM chunk_embeddings")).scalar()
            print(f"\nLive Database Verification in Cloud SQL (scrag):")
            print(f"  • judgments table count:        {j_cnt}")
            print(f"  • chunks table count:           {c_cnt}")
            print(f"  • chunk_embeddings table count: {e_cnt}")
            print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Sync GCS Chunk Deliverables directly into Cloud SQL PostgreSQL")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    default_db = os.environ.get("DATABASE_URL")
    parser.add_argument("--db", default=default_db, help="PostgreSQL connection string (defaults to DATABASE_URL in .env)")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "sc-judgments-rag-prod-data"), help="GCS Bucket Name")
    parser.add_argument("--year", type=int, default=None, help="Sync specific year (e.g. 1950)")
    parser.add_argument("--vector-mode", choices=["pgvector", "external"], default=None, help="Vector storage mode")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker threads (default: 16)")
    args = parser.parse_args()

    if not args.db:
        print("\nERROR: No database connection string provided!")
        print("Either pass --db 'postgresql://...' or set DATABASE_URL in your .env file.\n")
        sys.exit(1)

    syncer = GCSCloudSQLSyncer(db_url=args.db, bucket_name=args.bucket, vector_mode=args.vector_mode)
    syncer.sync_all(year=args.year, workers=args.workers)


if __name__ == "__main__":
    main()


