"""
Unit and Contract Tests for Data Ingestion Workflow
Validates contract compliance, OCR cleanup, page chunking, metadata extraction, LlamaParse integration, and duplicate handling.
"""

import os
import sys
import json
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    PageIngestionContract,
    PageMetadata,
    ChunkRecord,
    DownstreamChunkRecord,
    IngestionStatus
)
from cleaner_chunker import (
    LegalTextCleaner,
    LegalMetadataExtractor,
    PageChunker,
    PageProcessor
)
from extractor import (
    PDFExtractor,
    LlamaParseExtractor,
    DEFAULT_LLAMA_CLOUD_API_KEY
)
from pipeline import IngestionPipeline
from sample_generator import generate_sample_judgment_pdf
from db_loader import DatabaseLoader


def test_ocr_cleanup():
    raw_ocr = "IN THE SUPREME C0urt OF INDIA\nJ U D G M E N T\nAppe3llant v. Resp0ndent\nrepre- \nsentative"
    cleaned = LegalTextCleaner.clean_text(raw_ocr)
    assert "Court" in cleaned
    assert "JUDGMENT" in cleaned
    assert "Appellant" in cleaned
    assert "Respondent" in cleaned
    assert "representative" in cleaned


def test_page_chunking_and_chunk_ids():
    document_id = "TEST_DOC_001"
    page_number = 1
    sample_text = """JUDGMENT

1. First paragraph of the judgment explaining the facts.

2. Second paragraph detailing statutory provisions."""

    chunks = PageChunker.chunk_page(document_id, page_number, sample_text)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "TEST_DOC_001_p1_c1"
    assert chunks[0].chunk_index == 1
    assert chunks[0].chunk_metadata.section_heading == "JUDGMENT"
    assert chunks[1].chunk_id == "TEST_DOC_001_p1_c2"
    assert chunks[1].chunk_index == 2


def test_metadata_extraction():
    header_text = """
    IN THE SUPREME COURT OF INDIA
    CIVIL APPEAL NO. 1042 OF 2023
    State of Maharashtra ... Appellant(s)
    VERSUS
    ABC Infrastructure Ltd. ... Respondent(s)
    DATED: 24th February, 2023
    Hon'ble Mr. Justice D.Y. Chandrachud, CJI
    """
    metadata = LegalMetadataExtractor.extract_metadata(header_text)
    assert metadata.case_number is not None
    assert "1042" in metadata.case_number
    assert metadata.appellant is not None and "Maharashtra" in metadata.appellant
    assert metadata.respondent is not None and "ABC Infrastructure" in metadata.respondent
    assert metadata.date_of_judgment is not None and "2023" in metadata.date_of_judgment
    assert metadata.judges is not None and len(metadata.judges) > 0


def test_page_ingestion_contract_schema_compliance():
    """Validates that PageIngestionContract strictly complies with Kedar's required JSON structure."""
    contract = PageProcessor.process_page_rule_based(
        document_id="INSC_2023_CA_1042",
        page_number=1,
        raw_text="""CIVIL APPEAL NO. 1042 OF 2023\nUnion of India ... Appellant\nVERSUS\nXYZ Corp ... Respondent\nJUDGMENT\n1. Leave granted."""
    )
    dumped = contract.to_contract_dict()

    # Assert required top-level keys
    assert set(dumped.keys()) == {"document_id", "original_page_number", "page_metadata", "chunks"}
    assert dumped["document_id"] == "INSC_2023_CA_1042"
    assert dumped["original_page_number"] == 1

    # Assert page_metadata keys
    pm = dumped["page_metadata"]
    assert set(pm.keys()) == {"case_number", "appellant", "respondent", "date_of_judgment", "judges"}
    assert pm["case_number"] is not None
    assert pm["appellant"] == "Union of India"
    assert pm["respondent"] == "XYZ Corp"

    # Assert chunks structure
    for chunk in dumped["chunks"]:
        assert set(chunk.keys()) == {"chunk_index", "content", "chunk_metadata"}
        assert set(chunk["chunk_metadata"].keys()) == {"section_heading"}
        assert isinstance(chunk["chunk_index"], int)
        assert isinstance(chunk["content"], str)


def test_llm_prompt_formatting_and_parsing():
    """Tests prompt template rendering and JSON response parsing."""
    prompt = PageProcessor.format_llm_prompt(
        document_id="DOC_999",
        page_number=2,
        raw_page_text="Raw OCR text here"
    )
    assert "DOC_999" in prompt
    assert "Page Number: 2" in prompt
    assert "Raw OCR text here" in prompt

    simulated_llm_json = """
    ```json
    {
      "document_id": "DOC_999",
      "original_page_number": 2,
      "page_metadata": {
        "case_number": "Civil Appeal No. 50 of 2022",
        "appellant": "State of Punjab",
        "respondent": "Ramesh Kumar",
        "date_of_judgment": "10 January 2022",
        "judges": ["Hon'ble Justice A", "Hon'ble Justice B"]
      },
      "chunks": [
        {
          "chunk_index": 1,
          "content": "Cleaned paragraph 1 content.",
          "chunk_metadata": {
            "section_heading": "ARGUMENTS"
          }
        }
      ]
    }
    ```
    """
    parsed = PageProcessor.parse_llm_json_response("DOC_999", 2, simulated_llm_json)
    assert parsed.document_id == "DOC_999"
    assert parsed.original_page_number == 2
    assert parsed.page_metadata.appellant == "State of Punjab"
    assert len(parsed.chunks) == 1
    assert parsed.chunks[0].chunk_id == "DOC_999_p2_c1"


def test_downstream_chunk_record_contract():
    """Validates contract structure delivered to Anuradha (DB) and Pratik (Retrieval)."""
    downstream = DownstreamChunkRecord(
        chunk_id="INSC_2023_CA_1042_p1_c1",
        document_id="INSC_2023_CA_1042",
        original_page_number=1,
        chunk_index=1,
        content="Cleaned text of chunk",
        section_heading="JUDGMENT",
        case_number="CIVIL APPEAL NO. 1042 OF 2023",
        appellant="State of Maharashtra",
        respondent="ABC Infrastructure Ltd.",
        date_of_judgment="24th February, 2023",
        judges=["Hon'ble Mr. Justice D.Y. Chandrachud"]
    )
    d = downstream.model_dump()
    assert d["chunk_id"] == "INSC_2023_CA_1042_p1_c1"
    assert d["original_page_number"] == 1
    assert d["case_number"] == "CIVIL APPEAL NO. 1042 OF 2023"


def test_llamaparse_extractor_configuration():
    """Validates that LlamaParseExtractor properly detects API key and handles configuration."""
    extractor = LlamaParseExtractor()
    assert extractor.api_key == DEFAULT_LLAMA_CLOUD_API_KEY
    if extractor.api_key:
        assert extractor.api_key.startswith("llx-")

    custom_extractor = LlamaParseExtractor(api_key="llx-test-key-12345")
    assert custom_extractor.api_key == "llx-test-key-12345"
    assert custom_extractor.api_key.startswith("llx-")


def test_end_to_end_pipeline(tmp_path):
    # Generate sample PDF
    pdf_path = str(tmp_path / "sample_appeal.pdf")
    generate_sample_judgment_pdf(pdf_path)

    # Run Ingestion
    out_dir = str(tmp_path / "output")
    state_file = str(tmp_path / "state.json")
    report_file = str(tmp_path / "reports" / "error.json")

    pipeline = IngestionPipeline(
        output_dir=out_dir,
        state_file=state_file,
        report_file=report_file
    )

    result = pipeline.process_document(pdf_path)
    assert result.status == IngestionStatus.COMPLETED
    assert result.total_pages == 3
    assert result.total_chunks > 0
    assert result.parser_used in ("pymupdf", "llamaparse")
    assert result.aggregated_metadata.case_number is not None

    # Test Duplicate Detection
    dup_result = pipeline.process_document(pdf_path)
    assert dup_result.status == IngestionStatus.SKIPPED_DUPLICATE


def test_database_loader(tmp_path):
    """Validates that DatabaseLoader successfully initializes schema and bulk loads documents and chunks."""
    # Generate and process sample document
    pdf_path = str(tmp_path / "sample_doc.pdf")
    generate_sample_judgment_pdf(pdf_path)

    out_dir = str(tmp_path / "output")
    pipeline = IngestionPipeline(output_dir=out_dir)
    res = pipeline.process_document(pdf_path)
    assert res.status == IngestionStatus.COMPLETED

    # Test DB Loader with SQLite database in tmp_path
    db_file = tmp_path / "test.db"
    db_url = f"sqlite:///{db_file}"
    loader = DatabaseLoader(db_url=db_url)

    # Bulk load from output dir
    load_summary = loader.load_all_from_output(output_dir=out_dir)
    assert load_summary["documents_loaded"] == 1
    assert load_summary["total_chunks_loaded"] > 0
    assert load_summary["failed_count"] == 0

    # Verify DB stats
    stats = loader.get_stats()
    assert stats["total_documents"] == 1
    assert stats["total_chunks"] == load_summary["total_chunks_loaded"]
    assert stats["sample_chunk"] is not None
    assert "sample_doc" in stats["sample_chunk"]["chunk_id"]
