"""
Cloud Run Jobs Master Entrypoint for 1950-2026 Supreme Court Judgments
Scales horizontally across 77 years of PDFs stored in GCS.

Environment Variables automatically provided by Cloud Run Jobs:
- CLOUD_RUN_TASK_INDEX: 0-based index of this specific task execution (0..N-1)
- CLOUD_RUN_TASK_COUNT: Total number of tasks configured for the job (e.g. 77)

Features:
1. Dynamic Year Sharding: Distributes years 1950-2026 evenly across task workers.
   - If task_count == 77: Task 0 = 1950, Task 1 = 1951, ..., Task 76 = 2026.
   - If task_count == 10: Tasks share years via round-robin allocation.
2. Direct Partition Access: Worker streams directly from raw_pdfs/pdfs/year=YYYY/
   (No need to list 43,000 files across the entire bucket).
3. Resilient Error Isolation: Failed PDFs are logged to extraction_runs in Cloud SQL;
   one failed PDF never halts the rest of the year.
4. Two-Stage Execution: Supports parsing only, embedding only, or full end-to-end.

Author: Kedar (Data Ingestion & Pipeline Engineer)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

from gcp_production_pipeline import GCPProductionPipeline
from batch_embedder import BatchEmbedder
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Task %(process)d] %(name)s - %(message)s"
)
logger = logging.getLogger("CloudRunJobWorker")


def get_assigned_years(start_year: int, end_year: int, task_index: int, task_count: int) -> List[int]:
    """Distributes the target year range evenly across Cloud Run tasks."""
    all_years = list(range(start_year, end_year + 1))
    if task_count <= 1:
        return all_years
    return [y for idx, y in enumerate(all_years) if idx % task_count == task_index]


def main():
    parser = argparse.ArgumentParser(description="Cloud Run Jobs Orchestrator for 1950-2026 Supreme Court Judgments")
    parser.add_argument("--start-year", type=int, default=1950, help="Starting judgment year (default: 1950)")
    parser.add_argument("--end-year", type=int, default=2026, help="Ending judgment year (default: 2026)")
    parser.add_argument("--mode", choices=["ingest", "embed", "full"], default="full",
                        help="Execution mode: 'ingest' (parse+chunk), 'embed' (vectorize), or 'full' (both)")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "sc-judgments-rag-prod-data"),
                        help="GCS Bucket Name")
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT", "sc-judgments-rag"),
                        help="Google Cloud Project ID")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "asia-south1"),
                        help="GCP Region (default: asia-south1)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CONCURRENT_WORKERS", "4")),
                        help="Worker threads per container instance (default: 4)")
    parser.add_argument("--limit-per-year", type=int, default=None,
                        help="Optional limit on PDFs processed per year (for testing/canary runs)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing of already completed PDFs")
    parser.add_argument("--db", default=os.environ.get("DATABASE_URL"),
                        help="Database connection URL (PostgreSQL Cloud SQL / SQLite)")
    args = parser.parse_args()

    # Detect Cloud Run Jobs Task Environment
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))
    task_attempt = os.environ.get("CLOUD_RUN_TASK_ATTEMPT", "0")

    print("=" * 80)
    print(" SUPREME COURT RAG — CLOUD RUN JOB WORKER INITIALIZED")
    print("=" * 80)
    print(f"  • Cloud Run Task Index:   {task_index} (Attempt: {task_attempt})")
    print(f"  • Total Parallel Tasks:   {task_count}")
    print(f"  • Execution Mode:         {args.mode.upper()}")
    print(f"  • GCP Project:            {args.project}")
    print(f"  • GCS Bucket:             gs://{args.bucket}")
    print(f"  • Region:                 {args.region}")
    print(f"  • Database Connected:     {bool(args.db)}")
    print(f"  • Threads per Worker:     {args.workers}")

    # Determine assigned years for this specific task
    assigned_years = get_assigned_years(args.start_year, args.end_year, task_index, task_count)
    print(f"  • Assigned Years ({len(assigned_years)}):    {assigned_years}")
    print("=" * 80)

    overall_summary = {
        "task_index": task_index,
        "task_count": task_count,
        "mode": args.mode,
        "assigned_years": assigned_years,
        "years_processed": 0,
        "total_pdfs_completed": 0,
        "total_pdfs_skipped": 0,
        "total_pdfs_failed": 0,
        "total_chunks_created": 0,
        "total_chunks_embedded": 0,
        "start_time": datetime.utcnow().isoformat() + "Z",
        "end_time": None
    }

    # Initialize Pipeline Instances
    ingestion_pipeline = None
    batch_embedder = None

    if args.mode in ["ingest", "full"]:
        try:
            ingestion_pipeline = GCPProductionPipeline(
                bucket_name=args.bucket,
                project_id=args.project,
                region=args.region,
                db_url=args.db,
                enable_embeddings=(args.mode == "full"),
                max_workers=args.workers
            )
        except Exception as e:
            logger.error(f"Failed to initialize GCPProductionPipeline: {e}")
            import traceback
            traceback.print_exc()

    if args.mode in ["embed", "full"]:
        try:
            batch_embedder = BatchEmbedder(
                bucket_name=args.bucket,
                project_id=args.project,
                region=args.region,
                db_url=args.db,
                max_workers=args.workers
            )
        except Exception as e:
            logger.error(f"Failed to initialize BatchEmbedder: {e}")
            import traceback
            traceback.print_exc()

    # Process each assigned year
    for year in assigned_years:
        logger.info(f"--- [Task {task_index}] Starting Processing for Year {year} ---")

        try:
            # 1. Ingestion / Parsing Stage
            if ingestion_pipeline:
                # GCS Hive partitioned prefix: raw_pdfs/pdfs/year=YYYY/
                year_summary = ingestion_pipeline.run_gcs_batch(
                    prefix="raw_pdfs/pdfs/",
                    year=year,
                    limit=args.limit_per_year,
                    force=args.force
                )
                overall_summary["total_pdfs_completed"] += year_summary.get("completed", 0)
                overall_summary["total_pdfs_skipped"] += year_summary.get("skipped", 0)
                overall_summary["total_pdfs_failed"] += year_summary.get("failed", 0)
                overall_summary["total_chunks_created"] += year_summary.get("total_chunks_created", 0)
                logger.info(f"Year {year} Ingestion Complete: {year_summary}")

            # 2. Embedding Stage (if separate mode)
            if args.mode == "embed" and batch_embedder:
                embed_prefix = f"deliverables/chunks/year={year}/"
                logger.info(f"Running batch embedding for gs://{args.bucket}/{embed_prefix}...")
                embed_summary = batch_embedder.run_gcs_batch(prefix=embed_prefix)
                overall_summary["total_chunks_embedded"] += embed_summary.get("total_chunks_embedded", 0)
                logger.info(f"Year {year} Embedding Complete: {embed_summary}")

            overall_summary["years_processed"] += 1
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            import traceback
            traceback.print_exc()

    overall_summary["end_time"] = datetime.utcnow().isoformat() + "Z"

    print("\n" + "=" * 80)
    print(f" [Task {task_index}] CLOUD RUN JOB EXECUTION COMPLETED")
    print("=" * 80)
    print(json.dumps(overall_summary, indent=2))

    sys.exit(0)


if __name__ == "__main__":
    main()

