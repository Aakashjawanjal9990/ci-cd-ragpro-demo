-- ==============================================================================
-- Supreme Court Judgment RAG System — Production PostgreSQL & pgvector Schema
-- Cloud SQL PostgreSQL 15 / 16 Specification
-- Scale: ~43,000 Judgments | ~350,000 Pages | ~3,500,000 Chunks
-- Architect Review: Anuradha (Database Architect)
-- Ingestion & Pipeline Engineer: Kedar
-- ==============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";     -- For UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";       -- For fuzzy text search on case numbers & parties
CREATE EXTENSION IF NOT EXISTS "vector";        -- For 768-dim dense embedding vectors

-- ==============================================================================
-- TABLE 1: judgments (Master Judgment Catalog)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS judgments (
    judgment_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    judgment_id TEXT NOT NULL UNIQUE,                                       -- Unique ID, e.g. '1950_1_15_25_EN'
    court TEXT DEFAULT 'Supreme Court of India',
    case_number TEXT,                                                       -- e.g. 'CIVIL APPEAL NO. 1042 OF 2023'
    judgment_date DATE,                                                     -- YYYY-MM-DD
    citation_text TEXT,                                                     -- Primary citation, e.g. '[1950] S.C.R. 15'
    reporter_citations TEXT,                                                -- Other reporter citations (AIR, SCC, SCR)
    legal_status TEXT DEFAULT 'REPORTED',                                   -- 'REPORTED', 'UNREPORTED', 'OVERRULED'
    is_primary BOOLEAN NOT NULL DEFAULT TRUE,
    overruled_by_judgment_uuid UUID REFERENCES judgments(judgment_uuid) ON DELETE SET NULL,
    refresh_interval_days INTEGER,
    last_refreshed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================================
-- TABLE 2: source_files (PDF File Manifest & GCS Storage Tracking)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS source_files (
    source_file_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    judgment_uuid UUID NOT NULL REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
    version_number TEXT NOT NULL DEFAULT 'v1.0',
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    original_filename TEXT NOT NULL,                                        -- e.g. '1950_1_15_25_EN.pdf'
    storage_path TEXT NOT NULL,                                             -- e.g. 'gs://sc-judgments-rag-prod-data/raw_pdfs/pdfs/year=1950/...'
    doc_encryption_files JSONB,
    page_count INTEGER,
    ingestion_status TEXT NOT NULL DEFAULT 'COMPLETED',
    pipeline_version TEXT DEFAULT 'scj-ingestion-v1',
    created_by_user_uuid UUID,
    content_sha256 CHAR(64),                                                -- SHA-256 for duplicate detection
    file_size_bytes BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (judgment_uuid, version_number)
);

-- ==============================================================================
-- TABLE 3: ingestion_runs (Execution Audit & Worker Log)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS ingestion_runs (
    ingestion_run_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file_uuid UUID NOT NULL REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
    pipeline_name TEXT NOT NULL DEFAULT 'scj-data-ingestion-v1',
    processing_status TEXT NOT NULL,                                        -- 'COMPLETED', 'FAILED'
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    idempotency_key TEXT NOT NULL UNIQUE,
    worker_id TEXT,                                                         -- e.g. Cloud Run task ID
    started_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================================
-- TABLE 4: extraction_runs (Parser & OCR Audit Log)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS extraction_runs (
    extraction_run_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file_uuid UUID NOT NULL REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
    ingestion_run_uuid UUID REFERENCES ingestion_runs(ingestion_run_uuid) ON DELETE SET NULL,
    pipeline_version TEXT NOT NULL DEFAULT 'pymupdf',                       -- 'pymupdf', 'tesseract_ocr', 'llamaparse'
    processing_status TEXT NOT NULL DEFAULT 'COMPLETED',
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================================
-- TABLE 5: source_pages (Page-Level Verification & Raw OCR Text)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS source_pages (
    source_page_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file_uuid UUID NOT NULL REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    extraction_method TEXT NOT NULL DEFAULT 'pymupdf',                      -- 'pymupdf', 'tesseract_ocr'
    content TEXT,                                                           -- Raw unchunked page text
    quality_score NUMERIC DEFAULT 0.95,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_file_uuid, page_number)
);

-- ==============================================================================
-- TABLE 6: citations (Extracted Precedents & Legal Cross-References)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS citations (
    citation_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    judgment_uuid UUID NOT NULL REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
    citation_text TEXT NOT NULL,                                            -- e.g. 'AIR 1950 SC 124'
    citation_type TEXT NOT NULL DEFAULT 'REPORTER',
    normalized_citation TEXT,
    source_locator TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================================
-- TABLE 7: embedding_models (Registry for Embedding Foundation Models)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS embedding_models (
    embedding_model_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id TEXT NOT NULL,                                                 -- 'text-embedding-004'
    provider TEXT NOT NULL,                                                 -- 'vertex_ai'
    version TEXT,                                                           -- 'v1'
    dimensions INTEGER NOT NULL DEFAULT 768,                                -- Exactly 768 dimensions
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (provider, model_id, version)
);

-- ==============================================================================
-- TABLE 8: chunks (Core Retrieval Payload & 768-Dim Dense Vectors)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    chunk_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id TEXT UNIQUE NOT NULL,                                          -- Deterministic ID: '1950_1_15_25_EN_p1_c1'
    judgment_uuid UUID NOT NULL REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
    source_file_uuid UUID NOT NULL REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
    extraction_run_uuid UUID REFERENCES extraction_runs(extraction_run_uuid) ON DELETE SET NULL,
    last_seen_ingestion_run_uuid UUID REFERENCES ingestion_runs(ingestion_run_uuid) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,                                           -- Global sequence (1..N) within document
    page_start INTEGER NOT NULL,                                            -- Physical PDF page start
    page_end INTEGER NOT NULL,                                              -- Physical PDF page end
    content TEXT NOT NULL,                                                  -- Clean Markdown text (<512 tokens)
    section TEXT,                                                           -- Heading: 'FACTS', 'JUDGMENT', 'ORDER'
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,                            -- Case no, judges, appellant, respondent
    content_sha256 CHAR(64) NOT NULL,                                       -- Content deduplication hash
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    embedding VECTOR(768),                                                  -- 768-dim dense embedding
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_file_uuid, chunk_index)
);

-- ==============================================================================
-- TABLE 9: chunk_embeddings (Vector Sync & Multi-Store Audit)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_embedding_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_uuid UUID NOT NULL REFERENCES chunks(chunk_uuid) ON DELETE CASCADE,
    embedding_model_uuid UUID REFERENCES embedding_models(embedding_model_uuid) ON DELETE SET NULL,
    provider TEXT NOT NULL DEFAULT 'vertex_ai',
    corpus_resource_name TEXT,
    external_chunk_resource_name TEXT,
    embedding_ref TEXT,
    vector_store TEXT DEFAULT 'pgvector',
    sync_status TEXT NOT NULL DEFAULT 'SYNCED',                             -- 'PENDING', 'SYNCED', 'FAILED'
    last_synced_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chunk_uuid, embedding_model_uuid)
);

-- ==============================================================================
-- PERFORMANCE INDEXES (Optimized for ~3.5 Million Rows)
-- ==============================================================================

-- 1. HNSW Cosine Vector Index (Primary similarity search index)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- 2. Partial HNSW Index (Active chunks only)
CREATE INDEX IF NOT EXISTS idx_chunks_active_embedding 
ON chunks USING hnsw (embedding vector_cosine_ops) 
WHERE is_active = TRUE;

-- 3. B-Tree Indexes for Relational Filters & Foreign Keys
CREATE INDEX IF NOT EXISTS idx_chunks_judgment_uuid ON chunks(judgment_uuid);
CREATE INDEX IF NOT EXISTS idx_chunks_source_file_uuid ON chunks(source_file_uuid);
CREATE INDEX IF NOT EXISTS idx_chunks_page_start ON chunks(page_start);
CREATE INDEX IF NOT EXISTS idx_judgments_judgment_date ON judgments(judgment_date);

-- 4. Trigram GIN Indexes for Fuzzy Text Search on Case Numbers & Citations
CREATE INDEX IF NOT EXISTS idx_judgments_case_number_trgm ON judgments USING gin (case_number gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_judgments_citation_text_trgm ON judgments USING gin (citation_text gin_trgm_ops);

-- 5. Seed Official Vertex AI text-embedding-004 Model
INSERT INTO embedding_models (
    embedding_model_uuid, model_id, provider, version, dimensions, is_active
) VALUES (
    '00000000-0000-0000-0000-000000000768', 'text-embedding-004', 'vertex_ai', 'v1', 768, TRUE
) ON CONFLICT (provider, model_id, version) DO NOTHING;

