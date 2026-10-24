"""
Centralized Configuration & Environment Module
Loads environment variables, validates settings, and provides typed configuration defaults
for local development and production GCP / Cloud Run deployment.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()


class PipelineConfig:
    """Production configuration parameters for the Data Ingestion Pipeline."""

    # Project Roots and Output Directories
    BASE_DIR: Path = Path(__file__).resolve().parent
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))
    CHUNKS_DIR: Path = OUTPUT_DIR / "chunks"
    DOCUMENTS_DIR: Path = OUTPUT_DIR / "documents"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    STATE_FILE: Path = Path(os.getenv("STATE_FILE", str(BASE_DIR / "ingestion_state.json")))
    REPORT_FILE: Path = Path(os.getenv("REPORT_FILE", str(REPORTS_DIR / "status_error_report.json")))

    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'scj_rag.db'}")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    DB_BATCH_SIZE: int = int(os.getenv("DB_BATCH_SIZE", "500"))

    # Concurrency & Worker Settings
    # Default to min(4, CPU count) or 2
    _cpu_count = os.cpu_count() or 2
    DEFAULT_WORKERS: int = max(1, min(4, _cpu_count))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", str(DEFAULT_WORKERS)))

    # Extraction & OCR Settings
    MIN_TEXT_LENGTH_THRESHOLD: int = int(os.getenv("MIN_TEXT_THRESHOLD", "50"))
    OCR_DPI: int = int(os.getenv("OCR_DPI", "200"))
    ENABLE_OCR_DEFAULT: bool = os.getenv("ENABLE_OCR", "true").lower() in ("true", "1", "yes")

    # LlamaParse Cloud Settings
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    # Chunking & Token Constraints (Retriever & DB integration requirements)
    # 1500 chars ~ 350-450 tokens (safe within <512 token retrieval window)
    MAX_CHUNK_CHARS: int = int(os.getenv("MAX_CHUNK_CHARS", "1500"))
    CHUNK_OVERLAP_CHARS: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def ensure_directories(cls):
        """Ensures all necessary persistent runtime directories exist, falling back to /tmp if permissions are restricted."""
        try:
            cls.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
            cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            import tempfile
            tmp_base = Path(tempfile.gettempdir()) / "scj_output"
            cls.OUTPUT_DIR = tmp_base
            cls.CHUNKS_DIR = tmp_base / "chunks"
            cls.DOCUMENTS_DIR = tmp_base / "documents"
            cls.REPORTS_DIR = Path(tempfile.gettempdir()) / "scj_reports"
            try:
                cls.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
                cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
                cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    @classmethod
    def get_masked_llama_key(cls) -> str:
        """Returns masked API key for safe logging."""
        if not cls.LLAMA_CLOUD_API_KEY:
            return "<Not Set>"
        if len(cls.LLAMA_CLOUD_API_KEY) <= 10:
            return "***"
        return f"{cls.LLAMA_CLOUD_API_KEY[:6]}...{cls.LLAMA_CLOUD_API_KEY[-4:]}"


# Global config instance
config = PipelineConfig()
try:
    config.ensure_directories()
except Exception:
    pass

