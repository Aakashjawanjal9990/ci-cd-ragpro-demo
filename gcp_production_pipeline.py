"""
Comprehensive Production Ingestion & Embedding Pipeline for Google Cloud Platform (GCP)
Designed for: Supreme Court Judgment RAG System (~43,000 PDFs in GCS)

Key Capabilities:
1. Keyless GCP Authentication (Workload Identity, Colab auth, or GCE/Cloud Run Service Account)
2. GCS-Native Streaming (Reads from gs://sc-judgments-rag-prod-data/raw_pdfs/ without local storage)
3. Pre-Flight File Validation (Rejects 0-byte or corrupted PDFs safely)
4. Dual-Engine Extraction & OCR (PyMuPDF with automatic 200 DPI Tesseract OCR fallback + LlamaParse)
5. Legal Text Cleaning & Metadata Extraction (Fixes broken hyphens, extracts SCR citations, parties, bench)
6. Heading-Preserving Logical Chunking (<512 tokens with deterministic IDs)
7. Optional Batch Embeddings (Google Vertex AI text-embedding-004 in asia-south1)
8. Direct Database Synchronization (Upserts into 9-table PostgreSQL / Cloud SQL schema)
9. Horizontal Scalability (Supports Cloud Run Jobs task array indexing: CLOUD_RUN_TASK_INDEX)

Author: Kedar (Data Ingestion Engineer)
"""

import os
import sys
import json
import uuid
import time
import hashlib
import logging
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

# Internal Pipeline Modules
from config import config
from models import (
    PageMetadata,
    ChunkRecord,
    DownstreamChunkRecord,
    DocumentIngestionResult,
    IngestionStatus
)
from extractor import PDFExtractor, validate_pdf
from cleaner_chunker import PageProcessor
from db_loader import DatabaseLoader

# Safe GCP Imports (Allows running in non-GCP or local dev environments)
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

# Logging Configuration
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("GCPProductionPipeline")

# Default GCP Production Settings
DEFAULT_PROJECT_ID = os.environ.get("GCP_PROJECT", "sc-judgments-rag")
DEFAULT_BUCKET_NAME = os.environ.get("GCS_BUCKET", "sc-judgments-rag-prod-data")
DEFAULT_REGION = os.environ.get("GCP_REGION", "asia-south1")


class GCPProductionPipeline:
    """
    End-to-End GCP Ingestion, Parsing, Chunking, and Embedding Orchestrator.
    """

    def __init__(
        self,
        bucket_name: str = DEFAULT_BUCKET_NAME,
        project_id: str = DEFAULT_PROJECT_ID,
        region: str = DEFAULT_REGION,
        db_url: Optional[str] = None,
        use_llamaparse: bool = False,
        enable_embeddings: bool = False,
        embedding_model_name: str = "text-embedding-004",
        max_workers: int = 4
    ):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.region = region
        self.use_llamaparse = use_llamaparse
        self.enable_embeddings = enable_embeddings
        self.embedding_model_name = embedding_model_name
        self.max_workers = max_workers

        # Database Loader instance (if DB URL provided or available)
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.db_loader = DatabaseLoader(db_url=self.db_url) if self.db_url else None

        # Initialize GCP Storage Client
        self.storage_client = None
        self.bucket = None
        if GCP_STORAGE_AVAILABLE:
            try:
                credentials, identified_project = default()
                self.project_id = self.project_id or identified_project
                self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
                self.bucket = self.storage_client.bucket(self.bucket_name)
                logger.info(f"Initialized GCS Client for bucket: gs://{self.bucket_name} (Project: {self.project_id})")
            except Exception as e:
                logger.warning(f"Could not initialize GCS client: {e}. Will operate in local filesystem mode if requested.")

        # Initialize Vertex AI for Embeddings
        self.embed_model = None
        if self.enable_embeddings:
            if not VERTEX_AI_AVAILABLE:
                logger.warning("Vertex AI SDK not installed. Run `pip install google-cloud-aiplatform` to enable embeddings.")
            else:
                try:
                    vertexai.init(project=self.project_id, location=self.region)
                    self.embed_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
                    logger.info(f"Initialized Vertex AI {self.embedding_model_name} in {self.region}")
                except Exception as e:
                    logger.error(f"Failed to initialize Vertex AI: {e}")

        # Core Extractor
        self.extractor = PDFExtractor(
            ocr_enabled=True,
            use_llamaparse=self.use_llamaparse,
            llama_api_key=config.LLAMA_CLOUD_API_KEY
        )

    @staticmethod
    def derive_document_id(file_name: str) -> str:
        """Derives clean, standardized document ID from filename."""
        base_name = Path(file_name).stem
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """Computes SHA-256 digest of a file for duplicate detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates dense vector embeddings using Google Vertex AI in batches of 100.
        """
        if not self.embed_model:
            return chunks

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["content"] for c in batch]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    embeddings = self.embed_model.get_embeddings(texts)
                    for chunk, emb in zip(batch, embeddings):
                        chunk["embedding"] = emb.values
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + 0.5
                        logger.warning(f"Vertex AI retry {attempt + 1}/{max_retries} after {sleep_time:.1f}s: {e}")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Vertex AI embedding batch permanently failed after {max_retries} attempts: {e}")
        return chunks

    def process_pdf_file(
        self,
        local_pdf_path: str,
        output_dir: Path,
        doc_id: Optional[str] = None
    ) -> Tuple[DocumentIngestionResult, List[Dict[str, Any]]]:
        """
        Processes a local PDF file through validation, extraction, cleaning, and chunking.
        """
        doc_id = doc_id or self.derive_document_id(local_pdf_path)
        file_sha256 = self.compute_sha256(local_pdf_path)

        # 1. Defensive Pre-Flight Check & Page Extraction
        extracted_pages = self.extractor.extract_document_pages(local_pdf_path)
        total_pages = len(extracted_pages)

        pages = []
        ocr_pages = []
        engine_used = "pymupdf"
        doc_metadata = PageMetadata()

        # 2. Page-by-Page Cleaning & Chunking
        for page_num, raw_text, used_ocr, engine_name in extracted_pages:
            if engine_name == "llamaparse":
                engine_used = "llamaparse"
            if used_ocr:
                ocr_pages.append(page_num)

            page_contract = PageProcessor.process_page_rule_based(
                document_id=doc_id,
                page_number=page_num,
                raw_text=raw_text
            )
            pages.append(page_contract)

            # Metadata Roll-up
            pm = page_contract.page_metadata
            if not doc_metadata.case_number and pm.case_number:
                doc_metadata.case_number = pm.case_number
            if not doc_metadata.appellant and pm.appellant:
                doc_metadata.appellant = pm.appellant
            if not doc_metadata.respondent and pm.respondent:
                doc_metadata.respondent = pm.respondent
            if not doc_metadata.date_of_judgment and pm.date_of_judgment:
                doc_metadata.date_of_judgment = pm.date_of_judgment
            if not doc_metadata.judges and pm.judges:
                doc_metadata.judges = pm.judges

        # 3. Format Standardized Downstream Chunk Records
        downstream_chunks: List[Dict[str, Any]] = []
        for page in pages:
            for chunk in page.chunks:
                downstream_chunks.append({
                    "chunk_id": chunk.chunk_id or f"{doc_id}_p{page.original_page_number}_c{chunk.chunk_index}",
                    "document_id": doc_id,
                    "original_page_number": page.original_page_number,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "section_heading": chunk.chunk_metadata.section_heading,
                    "case_number": doc_metadata.case_number or page.page_metadata.case_number,
                    "appellant": doc_metadata.appellant or page.page_metadata.appellant,
                    "respondent": doc_metadata.respondent or page.page_metadata.respondent,
                    "date_of_judgment": doc_metadata.date_of_judgment or page.page_metadata.date_of_judgment,
                    "judges": doc_metadata.judges or page.page_metadata.judges
                })

        # 4. Optional Batch Embedding Generation
        if self.enable_embeddings:
            downstream_chunks = self.generate_embeddings_for_chunks(downstream_chunks)

        # 5. Build Document Summary Result
        result = DocumentIngestionResult(
            document_id=doc_id,
            file_path=local_pdf_path,
            file_sha256=file_sha256,
            total_pages=total_pages,
            total_chunks=len(downstream_chunks),
            status=IngestionStatus.COMPLETED,
            ocr_applied_pages=ocr_pages,
            parser_used=engine_used,
            aggregated_metadata=doc_metadata,
            pages=pages
        )

        return result, downstream_chunks

    def process_gcs_blob(self, blob_name: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Streams a PDF from GCS, processes it in ephemeral memory, and writes deliverables back to GCS.
        """
        if not self.bucket:
            raise RuntimeError("GCS bucket is not initialized.")

        doc_id = self.derive_document_id(blob_name)

        # Preserve Year Partitioning from GCS path (e.g. year=1950/)
        import re
        year_match = re.search(r"year=(\d{4})", blob_name)
        year_folder = f"year={year_match.group(1)}/" if year_match else ""

        chunks_blob_name = f"deliverables/chunks/{year_folder}{doc_id}_chunks.json"
        doc_blob_name = f"deliverables/documents/{year_folder}{doc_id}.json"

        # Check if already processed in GCS
        chunks_blob = self.bucket.blob(chunks_blob_name)
        if not force_reprocess and chunks_blob.exists():
            logger.info(f"Skipping {doc_id}: Deliverables already exist in GCS.")
            return {"document_id": doc_id, "status": "SKIPPED_DUPLICATE"}

        # Ephemeral RAM/disk scratch space
        with tempfile.TemporaryDirectory(prefix="gcp_scj_") as temp_dir:
            temp_pdf = os.path.join(temp_dir, Path(blob_name).name)
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Download PDF from GCS
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(temp_pdf)

            # Process PDF
            try:
                res, chunks = self.process_pdf_file(temp_pdf, output_dir, doc_id=doc_id)

                # Upload Chunks JSON to GCS
                chunks_blob.upload_from_string(
                    json.dumps(chunks, indent=2, ensure_ascii=False),
                    content_type="application/json"
                )

                # Upload Document Summary to GCS
                doc_blob = self.bucket.blob(doc_blob_name)
                doc_blob.upload_from_string(
                    json.dumps(res.model_dump(), indent=2, ensure_ascii=False),
                    content_type="application/json"
                )

                # Direct Database Sync (if DB configured)
                if self.db_loader:
                    local_doc_json = os.path.join(temp_dir, f"{doc_id}.json")
                    local_chunks_json = os.path.join(temp_dir, f"{doc_id}_chunks.json")
                    with open(local_doc_json, "w", encoding="utf-8") as f:
                        json.dump(res.model_dump(), f)
                    with open(local_chunks_json, "w", encoding="utf-8") as f:
                        json.dump(chunks, f)
                    self.db_loader.load_document_and_chunks(local_doc_json, local_chunks_json)

                logger.info(f"Successfully uploaded {doc_id} to GCS: {res.total_pages} pages, {len(chunks)} chunks.")
                return {
                    "document_id": doc_id,
                    "status": "COMPLETED",
                    "total_pages": res.total_pages,
                    "total_chunks": len(chunks)
                }

            except Exception as e:
                logger.error(f"Failed processing GCS PDF {blob_name}: {e}")
                return {"document_id": doc_id, "status": "FAILED", "error": str(e)}

    def run_gcs_batch(
        self,
        prefix: str = "raw_pdfs/",
        year: Optional[int] = None,
        limit: Optional[int] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Executes mass batch processing over PDFs stored in the GCS bucket.
        Supports Hive-partitioned directories (e.g., raw_pdfs/pdfs/year=1950/)
        and Cloud Run Jobs Task Indexing (CLOUD_RUN_TASK_INDEX) for horizontal scaling.
        """
        if not self.bucket:
            raise RuntimeError("GCS bucket is not initialized.")

        # If a specific year is requested, adjust prefix
        effective_prefix = prefix
        if year:
            effective_prefix = f"{prefix.rstrip('/')}/pdfs/year={year}/" if "pdfs" not in prefix else f"{prefix.rstrip('/')}/year={year}/"

        logger.info(f"Listing blobs in gs://{self.bucket_name}/{effective_prefix}...")
        all_blobs = [
            b.name for b in self.storage_client.list_blobs(self.bucket_name, prefix=effective_prefix)
            if b.name.endswith(".pdf")
        ]
        
        # If year filter passed but not captured by prefix, filter explicitly
        if year:
            all_blobs = [b for b in all_blobs if f"year={year}" in b]

        logger.info(f"Found {len(all_blobs)} PDF files in bucket (Prefix: '{effective_prefix}').")

        # Horizontal Sharding for Cloud Run Jobs Array Tasks
        task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
        task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))
        # If a specific year is already passed (entrypoint assigned year to this task),
        # process all PDFs for this year. Only shard if processing an unpartitioned directory.
        if task_count > 1 and not year:
            logger.info(f"Running as Cloud Run Task {task_index + 1}/{task_count}. Sharding work...")
            my_blobs = [b for idx, b in enumerate(all_blobs) if idx % task_count == task_index]
        else:
            my_blobs = all_blobs

        if limit and limit > 0:
            my_blobs = my_blobs[:limit]

        logger.info(f"Worker will process {len(my_blobs)} PDFs with {self.max_workers} worker threads.")

        summary = {
            "total_found": len(all_blobs),
            "assigned_to_worker": len(my_blobs),
            "completed": 0,
            "skipped": 0,
            "failed": 0,
            "total_chunks_created": 0,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None
        }

        # Multi-Threaded Execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_gcs_blob, blob_name, force): blob_name for blob_name in my_blobs}

            for future in as_completed(futures):
                blob_name = futures[future]
                try:
                    result = future.result()
                    status = result.get("status")
                    if status == "COMPLETED":
                        summary["completed"] += 1
                        summary["total_chunks_created"] += result.get("total_chunks", 0)
                    elif status == "SKIPPED_DUPLICATE":
                        summary["skipped"] += 1
                    else:
                        summary["failed"] += 1
                except Exception as e:
                    logger.error(f"Exception processing {blob_name}: {e}")
                    summary["failed"] += 1

        summary["end_time"] = datetime.utcnow().isoformat()
        logger.info(f"Batch execution complete: {summary}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Production Supreme Court Ingestion & Embedding Pipeline (GCP Native)")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: run-gcs
    run_parser = subparsers.add_parser("run-gcs", help="Run ingestion over GCS bucket PDFs")
    run_parser.add_argument("--bucket", default=DEFAULT_BUCKET_NAME, help=f"GCS Bucket Name (default: {DEFAULT_BUCKET_NAME})")
    run_parser.add_argument("--project", default=DEFAULT_PROJECT_ID, help=f"GCP Project ID (default: {DEFAULT_PROJECT_ID})")
    run_parser.add_argument("--prefix", default="raw_pdfs/", help="GCS prefix folder for raw PDFs (default: raw_pdfs/)")
    run_parser.add_argument("--year", type=int, default=None, help="Filter to a specific judgment year (e.g., 1950 or 2023)")
    run_parser.add_argument("--limit", type=int, default=None, help="Max number of PDFs to process")
    run_parser.add_argument("--workers", type=int, default=4, help="Concurrent worker threads (default: 4)")
    run_parser.add_argument("--embed", action="store_true", help="Generate Vertex AI text-embedding-004 vectors")
    run_parser.add_argument("--force", action="store_true", help="Force re-processing of duplicates")
    run_parser.add_argument("--use-llamaparse", action="store_true", help="Enable LlamaParse cloud parsing")
    run_parser.add_argument("--db", default=None, help="Database connection URL (PostgreSQL Cloud SQL / SQLite)")

    # Command: sync-db
    sync_parser = subparsers.add_parser("sync-db", help="Sync deliverables from local or GCS into PostgreSQL / SQLite")
    sync_parser.add_argument("--db", default=None, help="Database connection URL")
    sync_parser.add_argument("--output-dir", default="output", help="Directory containing JSON deliverables")
    sync_parser.add_argument("--stats", action="store_true", help="Display 9-table row counts")

    args = parser.parse_args()

    if args.command == "run-gcs":
        pipeline = GCPProductionPipeline(
            bucket_name=args.bucket,
            project_id=args.project,
            db_url=args.db,
            use_llamaparse=args.use_llamaparse,
            enable_embeddings=args.embed,
            max_workers=args.workers
        )
        summary = pipeline.run_gcs_batch(
            prefix=args.prefix,
            year=args.year,
            limit=args.limit,
            force=args.force
        )
        print("\n=== GCP Batch Run Summary ===")
        print(json.dumps(summary, indent=2))

    elif args.command == "sync-db":
        loader = DatabaseLoader(db_url=args.db)
        if args.stats:
            stats = loader.get_stats()
            print("\n=== Database Statistics (9 Tables) ===")
            print(json.dumps(stats, indent=2, default=str))
        else:
            summary = loader.load_all_from_output(args.output_dir)
            print("\n=== Database Load Summary ===")
            print(json.dumps(summary, indent=2))
            print(json.dumps(loader.get_stats(), indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

