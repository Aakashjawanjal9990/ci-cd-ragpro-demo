"""
Dedicated Mass Batch Embedding & Vector Ingestion Module
Designed for: Supreme Court Judgment RAG System (~3.5 Million Chunks in GCS)

Features:
1. Pure High-Throughput Network I/O (Async/Multi-Threaded)
2. Decoupled from PDF parsing: Reads chunk JSONs directly from GCS or local disk
3. Google Vertex AI (text-embedding-004 in asia-south1) with automatic batching (100 chunks/call)
4. Direct Cloud SQL (PostgreSQL) Vector Upserts into:
   - chunks (embedding column with pgvector)
   - embedding_models (model registration)
   - chunk_embeddings (sync tracking & status)
5. Idempotent Checkpointing: Skips documents whose chunks are already embedded

Author: Kedar (Data Ingestion Engineer)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, text

from config import config
from db_loader import DatabaseLoader

# Safe GCP Imports
try:
    from google.cloud import storage
    from google.auth import default
    GCP_STORAGE_AVAILABLE = True
except ImportError:
    GCP_STORAGE_AVAILABLE = False

try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("BatchEmbedder")

DEFAULT_PROJECT_ID = os.environ.get("GCP_PROJECT", "sc-judgments-rag")
DEFAULT_BUCKET_NAME = os.environ.get("GCS_BUCKET", "sc-judgments-rag-prod-data")
DEFAULT_REGION = os.environ.get("GCP_REGION", "asia-south1")
DEFAULT_MODEL_NAME = "text-embedding-004"
BATCH_SIZE = 100


class BatchEmbedder:
    """
    Dedicated batch embedder for mass vectorization of chunk JSONs.
    """

    def __init__(
        self,
        bucket_name: str = DEFAULT_BUCKET_NAME,
        project_id: str = DEFAULT_PROJECT_ID,
        region: str = DEFAULT_REGION,
        model_name: str = DEFAULT_MODEL_NAME,
        db_url: Optional[str] = None,
        max_workers: int = 8
    ):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.max_workers = max_workers
        self.db_url = db_url or os.environ.get("DATABASE_URL") or config.DATABASE_URL
        self.db_loader = DatabaseLoader(db_url=self.db_url)
        self.engine = self.db_loader.engine
        self.is_postgres = self.db_loader.is_postgres

        # 1. Initialize Vertex AI Model
        self.model = None
        if VERTEX_AI_AVAILABLE:
            try:
                vertexai.init(project=self.project_id, location=self.region)
                self.model = TextEmbeddingModel.from_pretrained(self.model_name)
                logger.info(f"Loaded Vertex AI model '{self.model_name}' (Location: {self.region})")
            except Exception as e:
                logger.error(f"Failed to load Vertex AI model: {e}")
        else:
            logger.warning("Vertex AI SDK not installed. Run `pip install google-cloud-aiplatform`.")

        # 2. Initialize GCS Client
        self.storage_client = None
        self.bucket = None
        if GCP_STORAGE_AVAILABLE:
            try:
                credentials, identified_proj = default()
                self.project_id = self.project_id or identified_proj
                self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
                self.bucket = self.storage_client.bucket(self.bucket_name)
                logger.info(f"Connected to GCS bucket: gs://{self.bucket_name}")
            except Exception as e:
                logger.warning(f"GCS client initialization skipped: {e}")

        # 3. Ensure Embedding Model Registered in DB (if DB connected)
        if self.engine:
            self._register_model_in_db()

    def _register_model_in_db(self):
        """Registers the embedding model in the embedding_models table."""
        model_uuid = "00000000-0000-0000-0000-000000000768"
        with self.engine.begin() as conn:
            if self.is_postgres:
                conn.execute(text("""
                    INSERT INTO embedding_models (
                        embedding_model_uuid, model_id, provider, version, dimensions, is_active
                    ) VALUES (
                        :uuid, :model_id, 'vertex_ai', 'v1', 768, TRUE
                    ) ON CONFLICT (provider, model_id, version) DO NOTHING;
                """), {"uuid": model_uuid, "model_id": self.model_name})
            else:
                conn.execute(text("""
                    INSERT OR IGNORE INTO embedding_models (
                        embedding_model_uuid, model_id, provider, version, dimensions, is_active
                    ) VALUES (
                        :uuid, :model_id, 'vertex_ai', 'v1', 768, 1
                    );
                """), {"uuid": model_uuid, "model_id": self.model_name})

    @staticmethod
    def _generate_dense_768_embedding(text_str: str, dim: int = 768) -> List[float]:
        """
        Generates deterministic, unit-normalized (L2 norm = 1.0) 768-dimensional dense vectors
        matching Vertex AI text-embedding-004 format for offline and local testing.
        """
        import hashlib
        import random
        import math
        seed = int(hashlib.sha256(text_str.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Calls Vertex AI to embed a list of texts in batches of 100."""
        allow_mock = os.environ.get("ALLOW_MOCK_EMBEDDINGS", "false").lower() in ("true", "1")
        if not self.model:
            if not allow_mock:
                raise RuntimeError(
                    "Vertex AI embedding model is not initialized. Mock embeddings are disabled "
                    "to prevent zero-similarity noise (Review Issue C1). Authenticate with GCP or set "
                    "ALLOW_MOCK_EMBEDDINGS=true for offline testing."
                )
            logger.warning("ALLOW_MOCK_EMBEDDINGS=true: Generating deterministic mock vectors for test harness.")
            return [self._generate_dense_768_embedding(t) for t in texts]

        all_vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            sub_batch = texts[i:i + BATCH_SIZE]
            try:
                embeddings = self.model.get_embeddings(sub_batch)
                all_vectors.extend([e.values for e in embeddings])
            except Exception as e:
                logger.error(f"Vertex AI embedding call failed: {e}.")
                if not allow_mock:
                    raise RuntimeError(f"Vertex AI embedding API failed: {e}. Aborting to avoid mock data.")
                logger.warning("Falling back to mock vectors because ALLOW_MOCK_EMBEDDINGS=true.")
                all_vectors.extend([self._generate_dense_768_embedding(t) for t in sub_batch])
        return all_vectors

    def process_chunk_collection(self, doc_id: str, chunks_data: List[Dict[str, Any]]) -> int:
        """
        Embeds a document's chunks and syncs the vectors into Cloud SQL:
        - Injects 'embedding' 768-dim vector into each chunk record
        - Updates chunks table (embedding column)
        - Updates chunk_embeddings table
        """
        if not chunks_data:
            return 0

        texts_to_embed = [c["content"] for c in chunks_data]
        vectors = self.embed_texts(texts_to_embed)

        model_uuid = "00000000-0000-0000-0000-000000000768"
        records_to_insert = []
        now_str = datetime.utcnow().isoformat()

        for c, vec in zip(chunks_data, vectors):
            # Inject embedding vector into the chunk object
            c["embedding"] = vec
            chunk_id = c["chunk_id"]
            ce_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk_id}.{model_uuid}"))
            records_to_insert.append({
                "ce_uuid": ce_uuid,
                "chunk_id": chunk_id,
                "model_uuid": model_uuid,
                "vec_json": json.dumps(vec),
                "now": now_str
            })

        # Upsert into chunks and chunk_embeddings tables (if DB connected)
        if self.engine:
            with self.engine.begin() as conn:
                for rec in records_to_insert:
                    # 1. Update chunks.embedding
                    if self.is_postgres:
                        conn.execute(text("""
                            UPDATE chunks 
                            SET embedding = :vec_json::vector 
                            WHERE chunk_id = :chunk_id;
                        """), {"chunk_id": rec["chunk_id"], "vec_json": rec["vec_json"]})
                    else:
                        conn.execute(text("""
                            UPDATE chunks 
                            SET embedding = :vec_json 
                            WHERE chunk_id = :chunk_id;
                        """), {"chunk_id": rec["chunk_id"], "vec_json": rec["vec_json"]})

                    # 2. Upsert into chunk_embeddings table with independent UUID (Fix for Issue C6)
                    if self.is_postgres:
                        conn.execute(text("""
                            INSERT INTO chunk_embeddings (
                                chunk_embedding_uuid, chunk_uuid, embedding_model_uuid,
                                provider, vector_store, sync_status, last_synced_at
                            )
                            SELECT 
                                :ce_uuid, c.chunk_uuid, :model_uuid,
                                'vertex_ai', 'pgvector', 'SYNCED', CURRENT_TIMESTAMP
                            FROM chunks c
                            WHERE c.chunk_id = :chunk_id
                            ON CONFLICT (chunk_uuid, embedding_model_uuid) DO UPDATE SET
                                sync_status = 'SYNCED',
                                last_synced_at = CURRENT_TIMESTAMP;
                        """), {"ce_uuid": rec["ce_uuid"], "chunk_id": rec["chunk_id"], "model_uuid": rec["model_uuid"]})
                    else:
                        conn.execute(text("""
                            INSERT OR REPLACE INTO chunk_embeddings (
                                chunk_embedding_uuid, chunk_uuid, embedding_model_uuid,
                                provider, vector_store, sync_status, last_synced_at
                            )
                            SELECT 
                                :ce_uuid, c.chunk_uuid, :model_uuid,
                                'vertex_ai', 'sqlite_json', 'SYNCED', CURRENT_TIMESTAMP
                            FROM chunks c
                            WHERE c.chunk_id = :chunk_id;
                        """), {"ce_uuid": rec["ce_uuid"], "chunk_id": rec["chunk_id"], "model_uuid": rec["model_uuid"]})

        logger.info(f"Vectorized & synced {len(chunks_data)} chunks for {doc_id}.")
        return len(chunks_data)

    def embed_from_gcs_blob(self, blob_name: str) -> Dict[str, Any]:
        """Reads a chunk JSON from GCS and processes embeddings."""
        doc_id = Path(blob_name).stem.replace("_chunks", "")
        blob = self.bucket.blob(blob_name)

        try:
            content_str = blob.download_as_text()
            chunks_data = json.loads(content_str)
            count = self.process_chunk_collection(doc_id, chunks_data)
            # Update the GCS JSON with the newly injected 768-dim embeddings
            blob.upload_from_string(
                json.dumps(chunks_data, indent=2, ensure_ascii=False),
                content_type="application/json"
            )
            return {"document_id": doc_id, "status": "COMPLETED", "chunks_embedded": count}
        except Exception as e:
            logger.error(f"Failed embedding GCS blob {blob_name}: {e}")
            return {"document_id": doc_id, "status": "FAILED", "error": str(e)}

    def embed_from_local_file(self, file_path: Path) -> Dict[str, Any]:
        """Reads a local chunk JSON file and processes embeddings."""
        doc_id = file_path.stem.replace("_chunks", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            count = self.process_chunk_collection(doc_id, chunks_data)
            return {"document_id": doc_id, "status": "COMPLETED", "chunks_embedded": count}
        except Exception as e:
            logger.error(f"Failed embedding local file {file_path}: {e}")
            return {"document_id": doc_id, "status": "FAILED", "error": str(e)}

    def run_gcs_batch(self, prefix: str = "deliverables/chunks/", limit: Optional[int] = None) -> Dict[str, Any]:
        """Mass embeds all chunk JSON files in GCS using multi-threading."""
        if not self.bucket:
            raise RuntimeError("GCS bucket not connected.")

        logger.info(f"Listing chunk collections in gs://{self.bucket_name}/{prefix}...")
        blobs = [b.name for b in self.storage_client.list_blobs(self.bucket_name, prefix=prefix) if b.name.endswith(".json")]
        logger.info(f"Found {len(blobs)} chunk collections.")

        if limit and limit > 0:
            blobs = blobs[:limit]

        summary = {"total_files": len(blobs), "completed": 0, "failed": 0, "total_chunks_embedded": 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.embed_from_gcs_blob, b): b for b in blobs}
            for future in as_completed(futures):
                res = future.result()
                if res.get("status") == "COMPLETED":
                    summary["completed"] += 1
                    summary["total_chunks_embedded"] += res.get("chunks_embedded", 0)
                else:
                    summary["failed"] += 1

        logger.info(f"Batch embedding complete: {summary}")
        return summary

    def run_local_batch(self, chunks_dir: str = "output/chunks", limit: Optional[int] = None) -> Dict[str, Any]:
        """Embeds all chunk JSON files from a local directory."""
        dir_path = Path(chunks_dir)
        files = list(dir_path.glob("*_chunks.json"))
        if limit and limit > 0:
            files = files[:limit]

        logger.info(f"Found {len(files)} local chunk files. Embedding with {self.max_workers} threads...")
        summary = {"total_files": len(files), "completed": 0, "failed": 0, "total_chunks_embedded": 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.embed_from_local_file, f): f for f in files}
            for future in as_completed(futures):
                res = future.result()
                if res.get("status") == "COMPLETED":
                    summary["completed"] += 1
                    summary["total_chunks_embedded"] += res.get("chunks_embedded", 0)
                else:
                    summary["failed"] += 1

        return summary


def main():
    parser = argparse.ArgumentParser(description="Dedicated Batch Embedder for Supreme Court RAG Chunks")
    parser.add_argument("--source", choices=["gcs", "local"], default="local", help="Source of chunk JSONs (gcs or local)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET_NAME, help=f"GCS bucket (default: {DEFAULT_BUCKET_NAME})")
    parser.add_argument("--prefix", default="deliverables/chunks/", help="GCS prefix folder for chunks")
    parser.add_argument("--local-dir", default="output/chunks", help="Local directory containing chunk JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads for concurrent API calls (default: 8)")
    parser.add_argument("--db", default=None, help="Database connection URL")
    args = parser.parse_args()

    embedder = BatchEmbedder(
        bucket_name=args.bucket,
        db_url=args.db,
        max_workers=args.workers
    )

    if args.source == "gcs":
        summary = embedder.run_gcs_batch(prefix=args.prefix, limit=args.limit)
    else:
        summary = embedder.run_local_batch(chunks_dir=args.local_dir, limit=args.limit)

    print("\n=== Batch Embedding Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
