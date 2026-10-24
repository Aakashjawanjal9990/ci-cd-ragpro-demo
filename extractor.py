"""
PDF Text, OCR & LlamaParse Extraction Module
Defensively parses selectable text, falls back to local OCR (Tesseract / PyMuPDF) or LlamaParse v2 Cloud.
Hardened for production reliability:
- Pre-flight PDF validation (0-byte detection, magic byte header check, encryption check)
- Strict context manager resource freeing (no file descriptor or native C-memory leaks)
- HTTP connection pooling and resilient timeout management
- Automatic fallback from Cloud parser to local OCR
"""

import os
import io
import time
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import fitz  # PyMuPDF
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import config

logger = logging.getLogger(__name__)

# Minimum character count to consider selectable text valid (below this triggers OCR fallback)
MIN_TEXT_LENGTH_THRESHOLD = config.MIN_TEXT_LENGTH_THRESHOLD
DEFAULT_LLAMA_CLOUD_API_KEY = config.LLAMA_CLOUD_API_KEY


class PDFValidationError(Exception):
    """Raised when a PDF file fails pre-flight validation."""
    pass


class CorruptPDFError(PDFValidationError):
    """Raised when a PDF file is malformed, truncated, or unreadable."""
    pass


class EmptyPDFError(PDFValidationError):
    """Raised when a PDF file has 0 bytes or 0 pages."""
    pass


class EncryptedPDFError(PDFValidationError):
    """Raised when a PDF file is encrypted and password cannot be bypassed."""
    pass


def validate_pdf(file_path: str) -> fitz.Document:
    """
    Performs pre-flight validation on a PDF file:
    1. Verifies file existence and non-zero file size.
    2. Checks for standard PDF magic header (%PDF-).
    3. Validates that PyMuPDF can parse the structure and access pages.
    4. Verifies encryption status.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {file_path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise EmptyPDFError(f"PDF file is 0 bytes (empty/corrupted download): {file_path}")

    # Check magic header
    try:
        with open(path, "rb") as f:
            header = f.read(1024)
            if b"%PDF-" not in header:
                raise CorruptPDFError(f"Missing '%PDF-' header in file: {file_path}")
    except (OSError, IOError) as e:
        raise CorruptPDFError(f"Unable to read file header: {e}")

    # Attempt PyMuPDF open
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise CorruptPDFError(f"PyMuPDF failed to parse document structure: {e}")

    if doc.is_encrypted:
        # Attempt blank password decrypt
        if not doc.authenticate(""):
            doc.close()
            raise EncryptedPDFError(f"PDF is password encrypted: {file_path}")

    if len(doc) == 0:
        doc.close()
        raise EmptyPDFError(f"PDF contains 0 pages: {file_path}")

    return doc


class LlamaParseExtractor:
    """
    Handles cloud-based PDF parsing via LlamaParse v2 API.
    Supports complex legal formatting, multi-column judgments, footnotes, and tables.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.LLAMA_CLOUD_API_KEY
        self.base_url = "https://api.cloud.llamaindex.ai/api/v2/parse"

    def is_available(self) -> bool:
        """Checks if API key is present and valid."""
        return bool(self.api_key and self.api_key.startswith("llx-"))

    def parse_pdf_pages(self, file_path: str, timeout_seconds: int = 120) -> List[Tuple[int, str]]:
        """
        Submits PDF to LlamaParse v2 API, polls for completion, and returns page-by-page markdown.
        Returns:
            List of tuples: (page_number: int, page_markdown: str)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        file_name = Path(file_path).name
        logger.info(f"Uploading {file_name} to LlamaParse v2 API...")

        # Use connection-pooled HTTP client with timeouts
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
            with open(file_path, "rb") as f:
                files = {
                    "file": (file_name, f, "application/pdf")
                }
                upload_config = json.dumps({
                    "tier": "cost_effective",
                    "version": "latest"
                })
                upload_res = client.post(
                    f"{self.base_url}/upload",
                    headers=headers,
                    files=files,
                    data={"configuration": upload_config}
                )

            if upload_res.status_code not in (200, 201):
                raise RuntimeError(f"LlamaParse upload failed ({upload_res.status_code}): {upload_res.text}")

            job_info = upload_res.json()
            job_id = job_info.get("id")
            if not job_id:
                raise RuntimeError(f"No job_id returned by LlamaParse: {upload_res.text}")

            logger.info(f"LlamaParse job {job_id} initiated for {file_name}. Polling status...")

            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                time.sleep(2)
                poll_res = client.get(
                    f"{self.base_url}/{job_id}?expand=markdown",
                    headers=headers
                )

                if poll_res.status_code != 200:
                    logger.warning(f"LlamaParse polling warning ({poll_res.status_code}): {poll_res.text}")
                    continue

                poll_data = poll_res.json()
                status = poll_data.get("job", {}).get("status")

                if status == "COMPLETED":
                    pages_data = poll_data.get("markdown", {}).get("pages", [])
                    results: List[Tuple[int, str]] = []
                    for idx, page_item in enumerate(pages_data):
                        p_num = page_item.get("page_number") or (idx + 1)
                        p_text = page_item.get("markdown") or ""
                        results.append((int(p_num), p_text.strip()))

                    logger.info(f"LlamaParse completed job {job_id} ({len(results)} pages).")
                    return results

                elif status == "FAILED":
                    err = poll_data.get("job", {}).get("error_message") or "Unknown error"
                    raise RuntimeError(f"LlamaParse job {job_id} failed: {err}")

            raise TimeoutError(f"LlamaParse job {job_id} timed out after {timeout_seconds}s")


class PDFExtractor:
    """
    Coordinates page-by-page text extraction with support for:
    - Defensive pre-flight validation
    - LlamaParse cloud parsing
    - Local PyMuPDF selectable text
    - Local OCR fallback (Tesseract / PyMuPDF OCR)
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        use_llamaparse: bool = False,
        llama_api_key: Optional[str] = None
    ):
        self.ocr_enabled = ocr_enabled
        self.use_llamaparse = use_llamaparse
        self.llama_extractor = LlamaParseExtractor(api_key=llama_api_key)

    def extract_document_pages(self, file_path: str) -> List[Tuple[int, str, bool, str]]:
        """
        Extracts all pages from a document defensively.
        Returns:
            List of tuples: (page_num, page_text, used_ocr, engine_name)
        """
        # Step 0: Pre-flight validation
        doc = validate_pdf(file_path)

        try:
            # Step 1: Attempt LlamaParse if requested and available
            if self.use_llamaparse and self.llama_extractor.is_available():
                try:
                    llama_pages = self.llama_extractor.parse_pdf_pages(file_path)
                    if llama_pages:
                        logger.info(f"Extracted {len(llama_pages)} pages via LlamaParse for {file_path}.")
                        return [
                            (p_num, p_text, True, "llamaparse")
                            for p_num, p_text in llama_pages
                        ]
                except Exception as e:
                    logger.warning(f"LlamaParse encountered error: {e}. Falling back to PyMuPDF/OCR.")

            # Step 2: Local PyMuPDF extraction with OCR fallback
            pages_out = []
            for idx in range(len(doc)):
                page_num = idx + 1
                page = doc[idx]
                text, used_ocr = self.extract_page_text(page, page_num)
                pages_out.append((page_num, text, used_ocr, "pymupdf_ocr" if used_ocr else "pymupdf"))

            return pages_out

        finally:
            # Guaranteed C-level memory and file descriptor release
            doc.close()

    def extract_page_text(self, page: fitz.Page, page_num: int) -> Tuple[str, bool]:
        """
        Extracts text from a single PyMuPDF page.
        Returns:
            Tuple of (raw_text, used_ocr: bool)
        """
        # Step 1: Selectable text extraction
        text = page.get_text("text").strip()

        # If selectable text is sufficiently dense and readable, return it
        if len(text) >= MIN_TEXT_LENGTH_THRESHOLD:
            return text, False

        # Step 2: OCR Fallback if selectable text is sparse
        if self.ocr_enabled:
            logger.debug(f"Page {page_num}: text length ({len(text)}) below threshold. Invoking OCR...")
            ocr_text = self._perform_ocr(page)
            if ocr_text and len(ocr_text.strip()) > 0:
                return ocr_text.strip(), True

        return text, False

    def _perform_ocr(self, page: fitz.Page) -> str:
        """
        Performs OCR on page raster image.
        Uses PyMuPDF built-in OCR or pytesseract pixmap fallback.
        Ensures native pixmap memory is promptly freed.
        """
        # Option A: PyMuPDF OCR (if Tesseract data is linked in PyMuPDF build)
        try:
            tp = page.get_textpage_ocr(flags=0, dpi=config.OCR_DPI, full=True)
            text = tp.extractText().strip()
            if text:
                return text
        except Exception:
            pass

        # Option B: pytesseract fallback using page pixmap
        pix = None
        try:
            import pytesseract
            from PIL import Image

            pix = page.get_pixmap(dpi=config.OCR_DPI)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(image)
            return ocr_text
        except ImportError:
            logger.debug("pytesseract or tesseract binary not available.")
        except Exception as e:
            logger.warning(f"OCR execution failed: {e}")
        finally:
            if pix is not None:
                pix = None  # Release pixmap memory

        return ""

    def get_document_page_count(self, file_path: str) -> int:
        """Returns the total number of pages in the PDF defensively."""
        with validate_pdf(file_path) as doc:
            return len(doc)
