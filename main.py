"""
Main CLI Entry Point for Kedar's Data Ingestion Workflow
Handles PDF ingestion, LlamaParse integration, sample generation, single-page JSON contracts, and batch runs.
Supports both modern and historical Indian Supreme Court judgment PDFs (e.g. 1950 SCR collection).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from models import PageIngestionContract
from cleaner_chunker import PageProcessor
from extractor import LlamaParseExtractor, DEFAULT_LLAMA_CLOUD_API_KEY
from pipeline import IngestionPipeline
from sample_generator import generate_sample_judgment_pdf
from db_loader import DatabaseLoader, DEFAULT_SQLITE_URL

DEFAULT_1950_DIR = "data/sample_pdfs/year=1950/english"


def run_single_page(document_id: str, page_number: int, text_content: str):
    """Processes a single page of text and outputs the exact JSON contract."""
    contract = PageProcessor.process_page_rule_based(
        document_id=document_id,
        page_number=page_number,
        raw_text=text_content
    )
    # Output strictly the required JSON contract
    contract_json = json.dumps(contract.to_contract_dict(), indent=2, ensure_ascii=False)
    print(contract_json)
    return contract_json


def run_sample_pipeline(use_llamaparse: bool = False, llama_api_key: Optional[str] = None):
    """Generates sample Indian Supreme Court PDFs and processes them through the pipeline."""
    sample_dir = Path("data/sample_pdfs")
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_pdf_1 = str(sample_dir / "INSC_2023_CA_1042.pdf")
    print(f"Generating realistic sample judgment PDF: {sample_pdf_1}")
    generate_sample_judgment_pdf(sample_pdf_1)

    print(f"\nInitializing IngestionPipeline (LlamaParse: {use_llamaparse})...")
    pipeline = IngestionPipeline(
        output_dir="output",
        state_file="ingestion_state.json",
        report_file="reports/status_error_report.json",
        enable_ocr=True,
        use_llamaparse=use_llamaparse,
        llama_api_key=llama_api_key
    )

    print(f"\nProcessing {sample_pdf_1}...")
    res = pipeline.process_document(sample_pdf_1, force_reprocess=True)

    print("\n--- Ingestion Result ---")
    print(f"Document ID: {res.document_id}")
    print(f"Status: {res.status}")
    print(f"Engine Used: {res.parser_used}")
    print(f"Total Pages: {res.total_pages}")
    print(f"Total Chunks: {res.total_chunks}")
    print(f"Case Number: {res.aggregated_metadata.case_number}")
    print(f"Appellant: {res.aggregated_metadata.appellant}")
    print(f"Respondent: {res.aggregated_metadata.respondent}")
    print(f"Date: {res.aggregated_metadata.date_of_judgment}")
    print(f"Judges: {res.aggregated_metadata.judges}")

    print("\n--- Verifying Duplicate Detection ---")
    dup_res = pipeline.process_document(sample_pdf_1)
    print(f"Duplicate Run Status: {dup_res.status} (Expected: SKIPPED_DUPLICATE)")

    print("\nDeliverables successfully generated:")
    print(" - Chunks for Anuradha (DB) & Pratik (Retrieval): output/chunks/INSC_2023_CA_1042_chunks.json")
    print(" - Document record: output/documents/INSC_2023_CA_1042.json")
    print(" - Ingestion state: ingestion_state.json")
    print(" - Failure/status report: reports/status_error_report.json")


def run_1950_batch(
    directory: str = DEFAULT_1950_DIR,
    limit: Optional[int] = 3,
    force: bool = False,
    use_llamaparse: bool = False,
    llama_key: Optional[str] = None
):
    """Processes historical 1950 Supreme Court judgment PDFs."""
    print(f"=== Supreme Court 1950 Judgments Ingestion ===")
    print(f"Directory: {directory}")
    print(f"Limit: {limit or 'All files'}")
    print(f"Engine: {'LlamaParse Cloud' if use_llamaparse else 'Local PyMuPDF'}")
    print(f"Force Reprocess: {force}\n")

    pipeline = IngestionPipeline(
        output_dir="output",
        state_file="ingestion_state.json",
        report_file="reports/status_error_report.json",
        enable_ocr=True,
        use_llamaparse=use_llamaparse,
        llama_api_key=llama_key
    )

    summary = pipeline.batch_process(
        input_dir_or_files=directory,
        pattern="*.pdf",
        limit=limit,
        force_reprocess=force
    )

    print("\n=== Ingestion Summary ===")
    print(json.dumps(summary, indent=2))
    print("\nDeliverables stored:")
    print(" - Chunks for DB & Retrieval: output/chunks/")
    print(" - Document records: output/documents/")
    print(" - Checkpoint state: ingestion_state.json")
    return summary


def test_llamaparse_connectivity(api_key: Optional[str] = None):
    """Tests LlamaParse parser availability and API key connectivity."""
    key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY") or DEFAULT_LLAMA_CLOUD_API_KEY
    print(f"Testing LlamaParse with API key: {key[:8]}...{key[-4:]}")
    extractor = LlamaParseExtractor(api_key=key)

    if not extractor.is_available():
        print("LlamaParse package is not installed or importable.")
        return False

    sample_pdf = "data/sample_pdfs/INSC_2023_CA_1042.pdf"
    if not os.path.exists(sample_pdf):
        generate_sample_judgment_pdf(sample_pdf)

    try:
        pages = extractor.parse_pdf_pages(sample_pdf)
        print(f"Successfully connected to LlamaParse! Parsed {len(pages)} pages.")
        for p_num, text in pages:
            preview = text[:80].replace("\n", " ")
            print(f" - Page {p_num}: {preview}...")
        return True
    except Exception as e:
        print(f"LlamaParse test encountered an error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Supreme Court Judgment Ingestion Pipeline (Kedar's Module)")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Command: run-sample
    sample_parser = subparsers.add_parser("run-sample", help="Generate sample PDFs and run ingestion pipeline")
    sample_parser.add_argument("--use-llamaparse", action="store_true", help="Enable LlamaParse cloud parser")
    sample_parser.add_argument("--llama-key", default=DEFAULT_LLAMA_CLOUD_API_KEY, help="LlamaParse Cloud API Key")

    # Command: test-1950 (Convenience command for testing the 1950 PDFs folder)
    t1950_parser = subparsers.add_parser("test-1950", help="Test batch ingestion on 1950 English judgments")
    t1950_parser.add_argument("--dir", default=DEFAULT_1950_DIR, help=f"Path to 1950 PDFs directory (default: {DEFAULT_1950_DIR})")
    t1950_parser.add_argument("--limit", type=int, default=3, help="Max number of files to process (default: 3, set to 0 for all)")
    t1950_parser.add_argument("--force", action="store_true", help="Force reprocessing even if already in state")
    t1950_parser.add_argument("--use-llamaparse", action="store_true", help="Enable LlamaParse cloud parser")
    t1950_parser.add_argument("--llama-key", default=DEFAULT_LLAMA_CLOUD_API_KEY, help="LlamaParse Cloud API Key")

    # Command: test-llamaparse
    test_lp = subparsers.add_parser("test-llamaparse", help="Test LlamaParse API connectivity")
    test_lp.add_argument("--key", default=DEFAULT_LLAMA_CLOUD_API_KEY, help="LlamaParse Cloud API Key")

    # Command: process-page
    page_parser = subparsers.add_parser("process-page", help="Process a single raw text page and output JSON contract")
    page_parser.add_argument("--doc-id", required=True, help="Source Document ID")
    page_parser.add_argument("--page", type=int, required=True, help="Original Page Number")
    page_parser.add_argument("--text", help="Raw text string or path to text file")

    # Command: process-pdf
    pdf_parser = subparsers.add_parser("process-pdf", help="Ingest a single PDF file")
    pdf_parser.add_argument("--file", required=True, help="Path to judgment PDF")
    pdf_parser.add_argument("--force", action="store_true", help="Force reprocessing even if duplicate")
    pdf_parser.add_argument("--use-llamaparse", action="store_true", help="Enable LlamaParse cloud parser")
    pdf_parser.add_argument("--llama-key", default=DEFAULT_LLAMA_CLOUD_API_KEY, help="LlamaParse Cloud API Key")

    # Command: batch
    batch_parser = subparsers.add_parser("batch", help="Batch process a directory of PDFs")
    batch_parser.add_argument("--dir", default=DEFAULT_1950_DIR, help=f"Directory containing PDF files (default: {DEFAULT_1950_DIR})")
    batch_parser.add_argument("--pattern", default="*.pdf", help="File pattern (default: *.pdf)")
    batch_parser.add_argument("--limit", type=int, default=None, help="Max number of files to process")
    batch_parser.add_argument("--force", action="store_true", help="Force reprocessing even if duplicate")
    batch_parser.add_argument("--use-llamaparse", action="store_true", help="Enable LlamaParse cloud parser")
    batch_parser.add_argument("--llama-key", default=DEFAULT_LLAMA_CLOUD_API_KEY, help="LlamaParse Cloud API Key")

    # Command: load-db
    load_db_parser = subparsers.add_parser("load-db", help="Load ingested deliverables from output/ into Database")
    load_db_parser.add_argument("--db", default=None, help=f"Database connection URL (default: {DEFAULT_SQLITE_URL} or env DATABASE_URL)")
    load_db_parser.add_argument("--output-dir", default="output", help="Directory containing JSON deliverables (default: output)")
    load_db_parser.add_argument("--doc-id", default=None, help="Specific document ID to load")
    load_db_parser.add_argument("--stats", action="store_true", help="Display current database table statistics")

    args = parser.parse_args()

    if args.command == "test-1950":
        limit_val = None if args.limit == 0 else args.limit
        run_1950_batch(
            directory=args.dir,
            limit=limit_val,
            force=args.force,
            use_llamaparse=args.use_llamaparse,
            llama_key=args.llama_key
        )
    elif args.command == "run-sample":
        run_sample_pipeline(
            use_llamaparse=args.use_llamaparse,
            llama_api_key=args.llama_key
        )
    elif args.command == "test-llamaparse":
        test_llamaparse_connectivity(api_key=args.key)
    elif args.command == "process-page":
        raw_text = args.text or ""
        if os.path.exists(raw_text):
            with open(raw_text, "r", encoding="utf-8") as f:
                raw_text = f.read()
        run_single_page(args.doc_id, args.page, raw_text)
    elif args.command == "process-pdf":
        pipeline = IngestionPipeline(
            use_llamaparse=args.use_llamaparse,
            llama_api_key=args.llama_key
        )
        res = pipeline.process_document(args.file, force_reprocess=args.force)
        print(f"Processed {res.document_id} with status {res.status} using {res.parser_used}")
    elif args.command == "batch":
        pipeline = IngestionPipeline(
            use_llamaparse=args.use_llamaparse,
            llama_api_key=args.llama_key
        )
        summary = pipeline.batch_process(
            input_dir_or_files=args.dir,
            pattern=args.pattern,
            limit=args.limit,
            force_reprocess=args.force
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "load-db":
        loader = DatabaseLoader(db_url=args.db)
        if args.stats:
            stats = loader.get_stats()
            print("\n=== Database Statistics ===")
            print(json.dumps(stats, indent=2, default=str))
        elif args.doc_id:
            doc_path = Path(args.output_dir) / "documents" / f"{args.doc_id}.json"
            chunk_path = Path(args.output_dir) / "chunks" / f"{args.doc_id}_chunks.json"
            res = loader.load_document_and_chunks(str(doc_path), str(chunk_path))
            print(json.dumps(res, indent=2))
        else:
            summary = loader.load_all_from_output(args.output_dir)
            print("\n=== Database Load Summary ===")
            print(json.dumps(summary, indent=2))
            stats = loader.get_stats()
            print("\n=== Database Statistics ===")
            print(json.dumps(stats, indent=2, default=str))
    else:
        # Default behavior: run test on 1950 directory
        run_1950_batch()


if __name__ == "__main__":
    main()
