# Supreme Court Judgment RAG System — Data Ingestion & Embedding Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.25.3-FF6F00)](https://pymupdf.readthedocs.io/)
[![LlamaParse](https://img.shields.io/badge/LlamaParse-0.6.94-7B2CBF)](https://cloud.llamaindex.ai/)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-text--embedding--004-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![PostgreSQL](https://img.shields.io/badge/Cloud%20SQL-PostgreSQL%2016-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-768--dim%20HNSW-blue)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/Tests-9%20Passed-success)](tests/test_ingestion.py)

**Owner / Module**: Kedar (Data Ingestion & Pipeline Engineer)  
**Target Dataset**: ~43,000 Indian Supreme Court judgment PDFs (1950 – Present)  
**GCP Storage Target**: `gs://sc-judgments-rag-prod-data/raw_pdfs/pdfs/year=YYYY/`  
**Repository**: [https://github.com/Kedarlimbalkar/parsing](https://github.com/Kedarlimbalkar/parsing)  

---

## 1. Project Overview

The **Data Ingestion & Embedding Pipeline** is the production data processing engine for the Supreme Court Judgment RAG platform. It processes scanned, historical, and modern digital legal PDFs into clean, page-isolated, section-tagged chunks ($<512$ tokens), generates 768-dimensional dense vector embeddings using Google Cloud Vertex AI `text-embedding-004`, and syncs them into the official 9-table schema in PostgreSQL Cloud SQL (`pgvector`).

### Key Capabilities
- **Decoupled Architecture**: Stage 1 parsing/chunking is decoupled from Stage 2 batch embedding, allowing vector re-indexing without re-parsing 43k PDFs.
- **Dual-Engine Extraction**: High-fidelity parsing via **LlamaParse (v2 REST API)** with automatic zero-downtime failover to local **PyMuPDF (`fitz`)** and 200 DPI Tesseract OCR fallback.
- **Strict Page-Boundary Isolation**: Text is chunked page-by-page. A chunk never crosses page boundaries, guaranteeing 100% verifiable page citations (`original_page_number`).
- **Heading-Preserving Chunking**: Automatically identifies legal sections (`FACTS`, `ISSUES`, `SUBMISSIONS`, `JUDGMENT`, `ORDER`) and tags child paragraphs under `section_heading`.
- **Mass Batch Embedding**: Generates unit-normalized 768-dim embeddings in sub-batches of 100 via Google Vertex AI `text-embedding-004` (in region `asia-south1`).
- **9-Table Cloud SQL Schema**: Unified database persistence supporting SQLite (local testing) and PostgreSQL Cloud SQL with HNSW cosine indexes.

---

## 2. Documentation Directory (`docs/`)

All architectural, security, database, and embedding specifications are neatly organized in the [`docs/`](docs/) directory:

| Document | Description |
| :--- | :--- |
| **[`docs/DATABASE_TABLES_AND_EMBEDDING_SPEC.md`](docs/DATABASE_TABLES_AND_EMBEDDING_SPEC.md)** | **Official 9-table schema (`scrag`), column constraints, chunk size rules, and Vertex AI embedding specs.** |
| **[`docs/END_TO_END_PIPELINE_AND_ARCHITECTURE.md`](docs/END_TO_END_PIPELINE_AND_ARCHITECTURE.md)** | **Master end-to-end specification, work record, approach, and detailed Mermaid architecture diagram.** |
| **[`docs/DATABASE_LOGGING_AND_AUDIT_SPEC.md`](docs/DATABASE_LOGGING_AND_AUDIT_SPEC.md)** | **Database log tables (`ingestion_runs`, `extraction_runs`, `chunk_embeddings`) and audit queries.** |
| **[`docs/HOW_WE_DID_THE_EMBEDDING.md`](docs/HOW_WE_DID_THE_EMBEDDING.md)** | **Complete step-by-step guide on how vector embeddings are generated, normalized, and saved.** |
| **[`docs/EMBEDDING_APPROACH_AND_DATABASE_HANDOFF.md`](docs/EMBEDDING_APPROACH_AND_DATABASE_HANDOFF.md)** | **Database architect handoff spec for Anuradha (DDL, pgvector HNSW indexes, RAM sizing).** |
| **[`docs/DATABASE_INTEGRATION_SPEC.md`](docs/DATABASE_INTEGRATION_SPEC.md)** | Complete 9-table schema definition, cascade foreign keys, and indexes. |
| **[`docs/DATA_INGESTION_ARCHITECTURE.md`](docs/DATA_INGESTION_ARCHITECTURE.md)** | In-depth parsing, OCR cleanup, and chunking technical specification. |
| **[`docs/GCP_ACCESS_AND_TECH_STACK.md`](docs/GCP_ACCESS_AND_TECH_STACK.md)** | Production GCP IAM roles, Cloud Run Jobs sizing, and service credentials. |
| **[`docs/PARSING_AND_CHUNKING_APPROACH.md`](docs/PARSING_AND_CHUNKING_APPROACH.md)** | Legal chunking logic, OCR header stripping, and token constraints. |
| **[`docs/PARSING_AND_CHUNKING_SPEC.md`](docs/PARSING_AND_CHUNKING_SPEC.md)** | Detailed single-page extraction and downstream retrieval contracts. |
| **[`docs/SECURITY_AND_INFRASTRUCTURE_SPEC.md`](docs/SECURITY_AND_INFRASTRUCTURE_SPEC.md)** | Security hardening, non-root containers, Secret Manager, and VPC egress. |
| **[`docs/PROJECT_WORK_SUMMARY.md`](docs/PROJECT_WORK_SUMMARY.md)** | Comprehensive project milestone report and operational summary. |

---

## 3. Quick Start & Execution

### 3.1 Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/Kedarlimbalkar/parsing.git
cd parsing

# Activate Python 3.11 virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Running Automated Tests
```bash
pytest tests/ -v
# ============================== 9 passed in ~7s ==============================
```

### 3.3 Running the Embedding Demonstration (For Anuradha)
```bash
python embedding_demo_for_anuradha.py
```
*Validates 768 dimensions, L2 unit norm ($\|v\|_2 = 1.000000$), semantic cosine queries, and displays database DDL.*

---

## 4. Production GCP Pipelines

### 4.1 Stage 1: Streaming PDF Extraction & Ingestion
Streams raw PDFs directly from GCS (`gs://sc-judgments-rag-prod-data/raw_pdfs/pdfs/year=YYYY/`), parses them, and saves chunk collections:
```bash
# Process all PDFs for a specific year from GCS
python gcp_production_pipeline.py --year 1950 --workers 8

# Process local directory
python gcp_production_pipeline.py --source-dir data/sample_pdfs --workers 4
```

### 4.2 Stage 2: Mass Batch Embedding & pgvector Sync
Vectorizes chunk collections and updates Cloud SQL PostgreSQL:
```bash
# Mass embed from GCS chunks folder
python batch_embedder.py --source gcs --bucket sc-judgments-rag-prod-data --workers 8

# Embed local chunk files and update database
python batch_embedder.py --source local --local-dir output/chunks
```

---

## 5. Repository Structure

```
├── .dockerignore
├── .env.example
├── .gitignore                      # Excludes .env, *.db, checkpoints, and caches
├── Dockerfile                      # Hardened container for Cloud Run Jobs (Python 3.11)
├── README.md                       # Main project documentation
├── requirements.txt                # Production dependencies
│
├── core/
│   ├── atomic_io.py                # Crash-safe atomic file writing (fsync + replace)
│   ├── config.py                   # Centralized configuration (GCP & Local defaults)
│   ├── extractor.py                # Pre-flight PDF validation, PyMuPDF, OCR & LlamaParse
│   ├── cleaner_chunker.py          # Legal section detection, OCR cleaner & <512 token chunker
│   ├── db_loader.py                # 9-table schema database loader (Postgres & SQLite)
│   ├── models.py                   # Strict Pydantic contracts for downstream consumers
│   ├── batch_embedder.py           # High-throughput Vertex AI 768-dim embedder
│   ├── gcp_production_pipeline.py  # Production GCS streaming orchestrator
│   ├── pipeline.py                 # Local batch orchestrator with retry logic
│   └── main.py                     # Unified CLI entrypoint
│
├── docs/                           # Centralized specifications & architecture handoffs
│   ├── HOW_WE_DID_THE_EMBEDDING.md
│   ├── EMBEDDING_APPROACH_AND_DATABASE_HANDOFF.md
│   ├── DATABASE_INTEGRATION_SPEC.md
│   ├── DATA_INGESTION_ARCHITECTURE.md
│   ├── GCP_ACCESS_AND_TECH_STACK.md
│   ├── PARSING_AND_CHUNKING_APPROACH.md
│   ├── PARSING_AND_CHUNKING_SPEC.md
│   ├── PROJECT_WORK_SUMMARY.md
│   └── SECURITY_AND_INFRASTRUCTURE_SPEC.md
│
├── tests/
│   └── test_ingestion.py           # 9 automated unit, extraction & contract tests
│
├── output/
│   ├── chunks/                     # Enriched chunk JSONs containing 768-dim embeddings
│   ├── documents/                  # Document rollup metadata
│   └── sample_embedded_chunk_for_anuradha.json # Dedicated handoff deliverable
│
└── embedding_demo_for_anuradha.py  # Standalone interactive demo script
```

---

## 6. GCP Credentials Required for Production

To run against Google Cloud Platform production resources, the following credentials are required:

1. **Google Cloud Service Account Key** (`sa-key.json` or `GOOGLE_APPLICATION_CREDENTIALS`):
   - **`roles/storage.objectAdmin`** on `gs://sc-judgments-rag-prod-data`
   - **`roles/aiplatform.user`** for Vertex AI `text-embedding-004` API calls in `asia-south1`
   - **`roles/cloudsql.client`** for Cloud SQL connectivity
2. **Cloud SQL PostgreSQL Connection String**:
   - `DATABASE_URL=postgresql://<db_user>:<db_password>@<cloud_sql_ip>:5432/scj_rag`
3. **LlamaParse API Key** (optional, for complex tables):
   - `LLAMA_CLOUD_API_KEY=llx-...` (stored in Google Secret Manager)
