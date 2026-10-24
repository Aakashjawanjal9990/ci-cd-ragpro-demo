"""
Data Ingestion Models & Contracts
Compliant with team roles specification and Anuradha (DB) / Pratik (Retrieval) integration requirements.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PageMetadata(BaseModel):
    """Metadata extracted from a single page of an Indian Supreme Court judgment."""
    case_number: Optional[str] = Field(
        default=None,
        description="Case number, e.g., 'Civil Appeal No. 1042 of 2023' or null if not present"
    )
    appellant: Optional[str] = Field(
        default=None,
        description="Appellant name(s) or null if not present on this page"
    )
    respondent: Optional[str] = Field(
        default=None,
        description="Respondent name(s) or null if not present on this page"
    )
    date_of_judgment: Optional[str] = Field(
        default=None,
        description="Date of judgment or null if not present on this page"
    )
    judges: Optional[List[str]] = Field(
        default=None,
        description="List of judges on the bench, or null if not present on this page"
    )


class ChunkMetadata(BaseModel):
    """Metadata associated with an individual text chunk."""
    section_heading: Optional[str] = Field(
        default=None,
        description="Section heading if the chunk falls under one (e.g. 'Judgment', 'Background')"
    )


class ChunkRecord(BaseModel):
    """An individual chunk conforming to the page extraction contract."""
    chunk_index: int = Field(
        ...,
        description="Sequential 1-based order for this page"
    )
    content: str = Field(
        ...,
        description="Cleaned text of the chunk"
    )
    chunk_metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata,
        description="Chunk-level metadata including section headings"
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Deterministic chunk identifier: {document_id}_p{page_number}_c{chunk_index}"
    )


class PageIngestionContract(BaseModel):
    """The exact single-page JSON extraction contract required by Kedar's specification."""
    document_id: str = Field(
        ...,
        description="Must match the provided Source Document ID"
    )
    original_page_number: int = Field(
        ...,
        description="Must match the provided Page Number (1-based)"
    )
    page_metadata: PageMetadata = Field(
        default_factory=PageMetadata,
        description="Extracted case/bench metadata if present on this page"
    )
    chunks: List[ChunkRecord] = Field(
        default_factory=list,
        description="List of ordered, cleaned text chunks on this page"
    )

    def to_contract_dict(self) -> Dict[str, Any]:
        """Returns the dictionary strictly conforming to the required prompt JSON contract."""
        return {
            "document_id": self.document_id,
            "original_page_number": self.original_page_number,
            "page_metadata": {
                "case_number": self.page_metadata.case_number,
                "appellant": self.page_metadata.appellant,
                "respondent": self.page_metadata.respondent,
                "date_of_judgment": self.page_metadata.date_of_judgment,
                "judges": self.page_metadata.judges
            },
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "chunk_metadata": {
                        "section_heading": c.chunk_metadata.section_heading
                    }
                }
                for c in self.chunks
            ]
        }


class DownstreamChunkRecord(BaseModel):
    """
    Standardized Chunk record consumable by Anuradha (Database Engineer)
    and Pratik (Retrieval Engineer) for indexing and vector storage.
    """
    chunk_id: str = Field(
        ...,
        description="Deterministic chunk identifier: {document_id}_p{page_number}_c{chunk_index}"
    )
    document_id: str = Field(..., description="Stable Source Document ID")
    original_page_number: int = Field(..., description="Original PDF page number")
    chunk_index: int = Field(..., description="Chunk sequence on page")
    content: str = Field(..., description="Cleaned chunk text content")
    section_heading: Optional[str] = Field(default=None, description="Active section heading")
    case_number: Optional[str] = Field(default=None, description="Associated case number")
    appellant: Optional[str] = Field(default=None, description="Appellant")
    respondent: Optional[str] = Field(default=None, description="Respondent")
    date_of_judgment: Optional[str] = Field(default=None, description="Date of judgment")
    judges: Optional[List[str]] = Field(default=None, description="Bench members")


class IngestionStatus(str):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"


class DocumentIngestionResult(BaseModel):
    """Document-level aggregated outcome for persistence and retrieval indexing."""
    document_id: str
    file_path: str
    file_sha256: str
    total_pages: int
    total_chunks: int
    status: str
    ocr_applied_pages: List[int] = Field(default_factory=list)
    parser_used: str = "pymupdf"
    aggregated_metadata: PageMetadata = Field(default_factory=PageMetadata)
    pages: List[PageIngestionContract] = Field(default_factory=list)
    error_message: Optional[str] = None
