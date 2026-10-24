"""
Atomic & Concurrency-Safe File I/O Module
Prevents file corruption from system crashes, OOM kills, or power loss,
and ensures process-safe state management in multi-worker environments.
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


def atomic_write_json(file_path: Path | str, data: Any, indent: int = 2) -> None:
    """
    Atomically writes data to a JSON file.
    Writes to a temporary file in the same directory, flushes to disk (fsync),
    and atomically replaces the target path. Readers will never see half-written JSON.
    """
    target = Path(file_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file in the same directory to ensure it is on the same filesystem/drive
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp"
    )

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Force OS buffer to write to physical storage

        # Atomic replacement (atomic in POSIX and Windows NTFS since Python 3.3)
        os.replace(tmp_path, target)
    except Exception as e:
        logger.error(f"Failed atomic write to {target}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def atomic_read_json(file_path: Path | str, default: Any = None) -> Any:
    """
    Safely reads a JSON file with graceful fallback if the file is missing or corrupted.
    """
    path = Path(file_path)
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading JSON from {path}: {e}. Returning default.")
        return default


class StateManager:
    """
    Thread- and process-safe state manager for ingestion checkpoints and error logs.
    Uses FileLock to synchronize writes between concurrent worker processes.
    """

    def __init__(self, state_file: Path | str, report_file: Path | str):
        self.state_file = Path(state_file).resolve()
        self.report_file = Path(report_file).resolve()
        self.state_lock = FileLock(str(self.state_file) + ".lock", timeout=30)
        self.report_lock = FileLock(str(self.report_file) + ".lock", timeout=30)

        # Initialize state file if absent
        if not self.state_file.exists():
            self._init_state()

    def _init_state(self):
        with self.state_lock:
            if not self.state_file.exists():
                initial = {
                    "processed_hashes": {},
                    "documents": {}
                }
                atomic_write_json(self.state_file, initial)

    def load_state(self) -> Dict[str, Any]:
        """Loads state safely with lock."""
        with self.state_lock:
            state = atomic_read_json(self.state_file)
            if not isinstance(state, dict):
                state = {"processed_hashes": {}, "documents": {}}
            if "processed_hashes" not in state:
                state["processed_hashes"] = {}
            if "documents" not in state:
                state["documents"] = {}
            return state

    def is_duplicate(self, file_sha256: str) -> Optional[str]:
        """Checks if hash is already processed (returns existing doc_id if duplicate)."""
        with self.state_lock:
            state = atomic_read_json(self.state_file, default={})
            return state.get("processed_hashes", {}).get(file_sha256)

    def record_success(
        self,
        document_id: str,
        file_sha256: str,
        file_path: str,
        parser_used: str,
        total_pages: int,
        total_chunks: int,
        timestamp: str
    ):
        """Atomically records successful ingestion under file lock."""
        with self.state_lock:
            state = atomic_read_json(self.state_file, default={"processed_hashes": {}, "documents": {}})
            state.setdefault("processed_hashes", {})[file_sha256] = document_id
            state.setdefault("documents", {})[document_id] = {
                "file_path": str(file_path),
                "sha256": file_sha256,
                "status": "COMPLETED",
                "parser_used": parser_used,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "last_updated": timestamp
            }
            atomic_write_json(self.state_file, state)

    def record_failure(
        self,
        document_id: str,
        file_sha256: str,
        file_path: str,
        error_msg: str,
        traceback_str: str,
        timestamp: str
    ):
        """Atomically records failed ingestion in state and appends to status_error_report.json."""
        with self.state_lock:
            state = atomic_read_json(self.state_file, default={"processed_hashes": {}, "documents": {}})
            state.setdefault("documents", {})[document_id] = {
                "file_path": str(file_path),
                "sha256": file_sha256,
                "status": "FAILED",
                "error": error_msg,
                "last_updated": timestamp
            }
            atomic_write_json(self.state_file, state)

        with self.report_lock:
            report: List[Dict[str, Any]] = atomic_read_json(self.report_file, default=[])
            if not isinstance(report, list):
                report = []
            report.append({
                "timestamp": timestamp,
                "document_id": document_id,
                "file_path": str(file_path),
                "sha256": file_sha256,
                "error": error_msg,
                "traceback": traceback_str
            })
            atomic_write_json(self.report_file, report)

