"""
Batch Ingestion Pipeline Module
Implements batch processing, retries, duplicate detection, resumable processing, and error reporting.
Supports LlamaParse cloud parsing with automatic fallback to local PyMuPDF / OCR.
Delivers chunk & metadata records formatted for Anuradha (Database) and Pratik (Retrieval).
"""

import os
import sys
import glob
import json
import hashlib
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import fitz  # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import (
    PageMetadata,
    ChunkRecord,
    PageIngestionContract,
    DownstreamChunkRecord,
    DocumentIngestionResult,
    IngestionStatus
)
from extractor import PDFExtractor
from cleaner_chunker import PageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("IngestionPipeline")


class IngestionPipeline:
    """Production-grade ingestion pipeline for Indian Supreme Court judgments."""

    def __init__(
        self,
        output_dir: str = "output",
        state_file: str = "ingestion_state.json",
        report_file: str = "reports/status_error_report.json",
        enable_ocr: bool = True,
        use_llamaparse: bool = False,
        llama_api_key: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.chunks_dir = self.output_dir / "chunks"
        self.documents_dir = self.output_dir / "documents"
        self.reports_dir = Path(report_file).parent
        self.state_file = Path(state_file)
        self.report_file = Path(report_file)

        # Create necessary directories
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = PDFExtractor(
            ocr_enabled=enable_ocr,
            use_llamaparse=use_llamaparse,
            llama_api_key=llama_api_key
        )
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Loads resumable ingestion state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}. Starting fresh state.")
        return {
            "processed_hashes": {},   # sha256 -> document_id
            "documents": {}           # document_id -> {status, completed_pages, total_pages, last_updated}
        }

    def _save_state(self):
        """Persists ingestion state for crash recovery."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """Computes SHA-256 hash of a file for duplicate detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def derive_document_id(file_path: str) -> str:
        """Derives a stable document ID from the filename."""
        base_name = Path(file_path).stem
        # Clean special chars to maintain clean ID convention
        doc_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)
        return doc_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _extract_pages_with_retry(self, file_path: str) -> List[tuple[int, str, bool, str]]:
        """Extracts pages with retry on transient network/IO issues."""
        return self.extractor.extract_document_pages(file_path)

    def process_document(self, file_path: str, force_reprocess: bool = False) -> DocumentIngestionResult:
        """
        Processes a single judgment PDF end-to-end:
        - Detects duplicates using SHA-256
        - Supports LlamaParse or local PyMuPDF extraction
        - Extracts text/OCR per page
        - Cleans and creates page-aware chunks
        - Aggregates document-level metadata (parties, case number, bench)
        - Writes deliverables for Anuradha (DB) & Pratik (Retrieval)
        """
        file_sha256 = self.compute_sha256(file_path)
        document_id = self.derive_document_id(file_path)

        # 1. Duplicate Detection Check
        if not force_reprocess and file_sha256 in self.state["processed_hashes"]:
            existing_doc_id = self.state["processed_hashes"][file_sha256]
            logger.info(f"Skipping duplicate file: {file_path} (Matches existing doc: {existing_doc_id})")
            return DocumentIngestionResult(
                document_id=existing_doc_id,
                file_path=file_path,
                file_sha256=file_sha256,
                total_pages=0,
                total_chunks=0,
                status=IngestionStatus.SKIPPED_DUPLICATE
            )

        logger.info(f"Starting ingestion for document: {document_id} [{file_path}]")
        self.state["documents"][document_id] = {
            "file_path": file_path,
            "sha256": file_sha256,
            "status": IngestionStatus.PROCESSING,
            "last_updated": datetime.utcnow().isoformat()
        }
        self._save_state()

        try:
            # Extract all pages (via LlamaParse or PyMuPDF/OCR)
            extracted_pages = self._extract_pages_with_retry(file_path)
            total_pages = len(extracted_pages)

            pages: List[PageIngestionContract] = []
            ocr_pages: List[int] = []
            engine_used = "pymupdf"

            # Aggregated document metadata
            doc_metadata = PageMetadata()

            for page_num, raw_text, used_ocr, engine_name in extracted_pages:
                if engine_name == "llamaparse":
                    engine_used = "llamaparse"
                if used_ocr:
                    ocr_pages.append(page_num)

                # Process page: clean text, extract metadata, chunk
                page_contract = PageProcessor.process_page_rule_based(
                    document_id=document_id,
                    page_number=page_num,
                    raw_text=raw_text
                )
                pages.append(page_contract)

                # Roll-up metadata
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

            # Format standardized downstream chunk records for Anuradha (Database) & Pratik (Retrieval)
            downstream_chunks: List[Dict[str, Any]] = []
            for page in pages:
                for chunk in page.chunks:
                    downstream_record = DownstreamChunkRecord(
                        chunk_id=chunk.chunk_id or f"{document_id}_p{page.original_page_number}_c{chunk.chunk_index}",
                        document_id=document_id,
                        original_page_number=page.original_page_number,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        section_heading=chunk.chunk_metadata.section_heading,
                        case_number=doc_metadata.case_number or page.page_metadata.case_number,
                        appellant=doc_metadata.appellant or page.page_metadata.appellant,
                        respondent=doc_metadata.respondent or page.page_metadata.respondent,
                        date_of_judgment=doc_metadata.date_of_judgment or page.page_metadata.date_of_judgment,
                        judges=doc_metadata.judges or page.page_metadata.judges
                    )
                    downstream_chunks.append(downstream_record.model_dump())

            # Persist Chunk records for Anuradha (Database) & Pratik (Retrieval)
            chunks_output_path = self.chunks_dir / f"{document_id}_chunks.json"
            with open(chunks_output_path, "w", encoding="utf-8") as f:
                json.dump(downstream_chunks, f, indent=2, ensure_ascii=False)

            # Persist Document summary record
            result = DocumentIngestionResult(
                document_id=document_id,
                file_path=file_path,
                file_sha256=file_sha256,
                total_pages=total_pages,
                total_chunks=len(downstream_chunks),
                status=IngestionStatus.COMPLETED,
                ocr_applied_pages=ocr_pages,
                parser_used=engine_used,
                aggregated_metadata=doc_metadata,
                pages=pages
            )

            doc_output_path = self.documents_dir / f"{document_id}.json"
            with open(doc_output_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

            # Update State
            self.state["processed_hashes"][file_sha256] = document_id
            self.state["documents"][document_id] = {
                "file_path": file_path,
                "sha256": file_sha256,
                "status": IngestionStatus.COMPLETED,
                "parser_used": engine_used,
                "total_pages": total_pages,
                "total_chunks": len(downstream_chunks),
                "last_updated": datetime.utcnow().isoformat()
            }
            self._save_state()

            logger.info(f"Successfully processed {document_id} using {engine_used}: {total_pages} pages, {len(downstream_chunks)} chunks.")
            return result

        except Exception as e:
            err_msg = str(e)
            tb = traceback.format_exc()
            logger.error(f"Failed processing {document_id}: {err_msg}")

            self._record_failure(document_id, file_path, file_sha256, err_msg, tb)

            self.state["documents"][document_id] = {
                "file_path": file_path,
                "sha256": file_sha256,
                "status": IngestionStatus.FAILED,
                "error": err_msg,
                "last_updated": datetime.utcnow().isoformat()
            }
            self._save_state()

            return DocumentIngestionResult(
                document_id=document_id,
                file_path=file_path,
                file_sha256=file_sha256,
                total_pages=0,
                total_chunks=0,
                status=IngestionStatus.FAILED,
                error_message=err_msg
            )

    def _record_failure(self, document_id: str, file_path: str, sha256: str, error: str, traceback_str: str):
        """Appends failure to status_error_report.json."""
        report = []
        if self.report_file.exists():
            try:
                with open(self.report_file, "r", encoding="utf-8") as f:
                    report = json.load(f)
            except Exception:
                report = []

        report.append({
            "timestamp": datetime.utcnow().isoformat(),
            "document_id": document_id,
            "file_path": file_path,
            "sha256": sha256,
            "error": error,
            "traceback": traceback_str
        })

        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def batch_process(
        self,
        input_dir_or_files: str,
        pattern: str = "*.pdf",
        limit: Optional[int] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Executes batch ingestion over a directory of PDFs or file pattern.
        Supports limit, force re-processing, and progress tracking.
        Returns a summary report.
        """
        input_path = Path(input_dir_or_files)
        if input_path.is_dir():
            # If pattern starts with **, use rglob, otherwise glob
            if pattern.startswith("**"):
                pdf_files = sorted([str(p) for p in input_path.rglob(pattern.replace("**/", ""))])
            else:
                pdf_files = sorted([str(p) for p in input_path.glob(pattern)])
        else:
            pdf_files = sorted(glob.glob(input_dir_or_files))

        total_discovered = len(pdf_files)
        if limit and limit > 0:
            pdf_files = pdf_files[:limit]

        logger.info(f"Found {total_discovered} PDF files (Processing {len(pdf_files)}).")
        summary = {
            "total_found": total_discovered,
            "total_processed": len(pdf_files),
            "completed": 0,
            "duplicates_skipped": 0,
            "failed": 0,
            "total_chunks_created": 0,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None
        }

        for idx, pdf_path in enumerate(pdf_files, 1):
            file_name = Path(pdf_path).name
            print(f"[{idx}/{len(pdf_files)}] Ingesting {file_name}...")
            res = self.process_document(pdf_path, force_reprocess=force_reprocess)
            if res.status == IngestionStatus.COMPLETED:
                summary["completed"] += 1
                summary["total_chunks_created"] += res.total_chunks
                print(f"   -> Completed: {res.total_pages} pages, {res.total_chunks} chunks.")
            elif res.status == IngestionStatus.SKIPPED_DUPLICATE:
                summary["duplicates_skipped"] += 1
                print(f"   -> Skipped (Duplicate file).")
            elif res.status == IngestionStatus.FAILED:
                summary["failed"] += 1
                print(f"   -> Failed: {res.error_message}")

        summary["end_time"] = datetime.utcnow().isoformat()
        logger.info(f"Batch Ingestion Complete: {summary}")
        return summary
