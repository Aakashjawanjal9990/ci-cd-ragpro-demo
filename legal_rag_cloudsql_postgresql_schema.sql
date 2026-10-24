-- Legal RAG schema for Google Cloud SQL for PostgreSQL 16.
-- Database: "scrag", Schema: "public"
-- Based on official Mermaid ER diagram (Deployment_Arch.mmd).
BEGIN;

CREATE TABLE users (
 user_uuid UUID PRIMARY KEY, username TEXT NOT NULL UNIQUE, email TEXT NOT NULL UNIQUE,
 password_hash TEXT NOT NULL, full_name TEXT, role TEXT NOT NULL, account_status TEXT NOT NULL,
 last_login_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK (length(btrim(username))>0), CHECK (length(btrim(email))>0)
);

CREATE TABLE judgments (
 judgment_uuid UUID PRIMARY KEY, judgment_id TEXT NOT NULL UNIQUE, court TEXT, case_number TEXT,
 judgment_date DATE, citation_text TEXT, reporter_citations TEXT, legal_status TEXT,
 is_primary BOOLEAN NOT NULL DEFAULT FALSE, overruled_by_judgment_uuid UUID,
 refresh_interval_days INTEGER, last_refreshed_at TIMESTAMPTZ,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK (refresh_interval_days IS NULL OR refresh_interval_days>0),
 CHECK (overruled_by_judgment_uuid IS NULL OR overruled_by_judgment_uuid<>judgment_uuid),
 FOREIGN KEY (overruled_by_judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE SET NULL
);

CREATE TABLE source_files (
 source_file_uuid UUID PRIMARY KEY, judgment_uuid UUID NOT NULL, version_number TEXT NOT NULL,
 is_current BOOLEAN NOT NULL DEFAULT FALSE, original_filename TEXT NOT NULL, storage_path TEXT NOT NULL,
 doc_encryption_files JSONB, page_count INTEGER, ingestion_status TEXT NOT NULL,
 pipeline_version TEXT, created_by_user_uuid UUID, content_sha256 CHAR(64), file_size_bytes BIGINT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
 FOREIGN KEY (created_by_user_uuid) REFERENCES users(user_uuid) ON DELETE SET NULL,
 CHECK (page_count IS NULL OR page_count>=0), CHECK (file_size_bytes IS NULL OR file_size_bytes>=0),
 CHECK (content_sha256 IS NULL OR content_sha256 ~ '^[0-9A-Fa-f]{64}$'),
 CHECK (doc_encryption_files IS NULL OR jsonb_typeof(doc_encryption_files) IN ('object','array')),
 UNIQUE (judgment_uuid, version_number)
);
CREATE UNIQUE INDEX source_files_one_current_per_judgment_uq ON source_files(judgment_uuid) WHERE is_current;
CREATE INDEX source_files_judgment_created_idx ON source_files(judgment_uuid, created_at DESC);
CREATE INDEX source_files_ingestion_status_idx ON source_files(ingestion_status);

CREATE TABLE ingestion_runs (
 ingestion_run_uuid UUID PRIMARY KEY, source_file_uuid UUID NOT NULL, pipeline_name TEXT NOT NULL,
 processing_status TEXT NOT NULL, error_message TEXT, retry_count INTEGER NOT NULL DEFAULT 0,
 max_attempts INTEGER NOT NULL DEFAULT 3, idempotency_key TEXT NOT NULL UNIQUE, worker_id TEXT,
 started_at TIMESTAMPTZ, heartbeat_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
 CHECK (retry_count>=0 AND max_attempts>0 AND retry_count<=max_attempts),
 CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at>=started_at)
);
CREATE INDEX ingestion_runs_source_created_idx ON ingestion_runs(source_file_uuid, created_at DESC);
CREATE INDEX ingestion_runs_status_created_idx ON ingestion_runs(processing_status, created_at);
CREATE INDEX ingestion_runs_active_heartbeat_idx ON ingestion_runs(heartbeat_at) WHERE finished_at IS NULL;

CREATE TABLE extraction_runs (
 extraction_run_uuid UUID PRIMARY KEY, source_file_uuid UUID NOT NULL, ingestion_run_uuid UUID,
 pipeline_version TEXT NOT NULL, processing_status TEXT NOT NULL, error_message TEXT,
 retry_count INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
 started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
 FOREIGN KEY (ingestion_run_uuid) REFERENCES ingestion_runs(ingestion_run_uuid) ON DELETE SET NULL,
 CHECK (retry_count>=0 AND max_attempts>0 AND retry_count<=max_attempts),
 CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at>=started_at)
);
CREATE INDEX extraction_runs_source_created_idx ON extraction_runs(source_file_uuid, created_at DESC);
CREATE INDEX extraction_runs_ingestion_idx ON extraction_runs(ingestion_run_uuid);
CREATE INDEX extraction_runs_status_created_idx ON extraction_runs(processing_status, created_at);

CREATE TABLE source_pages (
 source_page_uuid UUID PRIMARY KEY, source_file_uuid UUID NOT NULL, page_number INTEGER NOT NULL,
 extraction_method TEXT NOT NULL, content TEXT, quality_score NUMERIC,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
 CHECK (page_number>0), CHECK (quality_score IS NULL OR quality_score BETWEEN 0 AND 1),
 UNIQUE (source_file_uuid, page_number)
);
CREATE INDEX source_pages_source_idx ON source_pages(source_file_uuid);

CREATE TABLE categories (
 category_uuid UUID PRIMARY KEY, name TEXT NOT NULL UNIQUE, slug TEXT NOT NULL UNIQUE,
 description TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE judgment_categories (
 judgment_category_uuid UUID PRIMARY KEY, judgment_uuid UUID NOT NULL, category_uuid UUID NOT NULL,
 confidence NUMERIC, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
 FOREIGN KEY (category_uuid) REFERENCES categories(category_uuid) ON DELETE CASCADE,
 CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1), UNIQUE (judgment_uuid, category_uuid)
);
CREATE INDEX judgment_categories_category_idx ON judgment_categories(category_uuid);

CREATE TABLE citations (
 citation_uuid UUID PRIMARY KEY, judgment_uuid UUID NOT NULL, citation_text TEXT NOT NULL,
 citation_type TEXT NOT NULL, normalized_citation TEXT, source_locator TEXT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE CASCADE
);
CREATE INDEX citations_judgment_idx ON citations(judgment_uuid);
CREATE INDEX citations_normalized_idx ON citations(normalized_citation);

CREATE TABLE citation_history (
 citation_history_uuid UUID PRIMARY KEY, judgment_uuid UUID NOT NULL, citation_uuid UUID,
 event_type TEXT NOT NULL, previous_value TEXT, new_value TEXT, change_reason TEXT,
 changed_by_user_uuid UUID, changed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
 FOREIGN KEY (citation_uuid) REFERENCES citations(citation_uuid) ON DELETE SET NULL,
 FOREIGN KEY (changed_by_user_uuid) REFERENCES users(user_uuid) ON DELETE SET NULL
);
CREATE INDEX citation_history_judgment_changed_idx ON citation_history(judgment_uuid, changed_at DESC);
CREATE INDEX citation_history_citation_idx ON citation_history(citation_uuid);

CREATE TABLE chunks (
 chunk_uuid UUID PRIMARY KEY, judgment_uuid UUID NOT NULL, source_file_uuid UUID NOT NULL,
 extraction_run_uuid UUID, last_seen_ingestion_run_uuid UUID, chunk_index INTEGER NOT NULL,
 page_start INTEGER, page_end INTEGER, content TEXT NOT NULL, section TEXT, metadata JSONB,
 content_sha256 CHAR(64), is_active BOOLEAN NOT NULL DEFAULT TRUE,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE CASCADE,
 FOREIGN KEY (source_file_uuid) REFERENCES source_files(source_file_uuid) ON DELETE CASCADE,
 FOREIGN KEY (extraction_run_uuid) REFERENCES extraction_runs(extraction_run_uuid) ON DELETE SET NULL,
 FOREIGN KEY (last_seen_ingestion_run_uuid) REFERENCES ingestion_runs(ingestion_run_uuid) ON DELETE SET NULL,
 CHECK (chunk_index>=0), CHECK (page_start IS NULL OR page_start>0),
 CHECK (page_end IS NULL OR page_end>0),
 CHECK (page_start IS NULL OR page_end IS NULL OR page_end>=page_start),
 CHECK (content_sha256 IS NULL OR content_sha256 ~ '^[0-9A-Fa-f]{64}$'),
 CHECK (metadata IS NULL OR jsonb_typeof(metadata)='object'),
 UNIQUE (source_file_uuid, chunk_index)
);
CREATE INDEX chunks_judgment_active_idx ON chunks(judgment_uuid, is_active);
CREATE INDEX chunks_source_file_idx ON chunks(source_file_uuid);
CREATE INDEX chunks_extraction_run_idx ON chunks(extraction_run_uuid);
CREATE INDEX chunks_last_seen_ingestion_idx ON chunks(last_seen_ingestion_run_uuid);
CREATE INDEX chunks_sha256_idx ON chunks(content_sha256);

CREATE TABLE embedding_models (
 embedding_model_uuid UUID PRIMARY KEY, model_id TEXT NOT NULL, provider TEXT NOT NULL,
 version TEXT, dimensions INTEGER NOT NULL, is_active BOOLEAN NOT NULL DEFAULT TRUE,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK (dimensions>0), UNIQUE (provider, model_id, version)
);

CREATE TABLE chunk_embeddings (
 chunk_embedding_uuid UUID PRIMARY KEY, chunk_uuid UUID NOT NULL, embedding_model_uuid UUID,
 provider TEXT NOT NULL, corpus_resource_name TEXT, external_chunk_resource_name TEXT,
 embedding_ref TEXT, vector_store TEXT, sync_status TEXT NOT NULL, last_synced_at TIMESTAMPTZ,
 last_error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (chunk_uuid) REFERENCES chunks(chunk_uuid) ON DELETE CASCADE,
 FOREIGN KEY (embedding_model_uuid) REFERENCES embedding_models(embedding_model_uuid) ON DELETE SET NULL,
 UNIQUE (chunk_uuid, embedding_model_uuid)
);
CREATE INDEX chunk_embeddings_sync_status_idx ON chunk_embeddings(sync_status);
CREATE INDEX chunk_embeddings_model_idx ON chunk_embeddings(embedding_model_uuid);
CREATE INDEX chunk_embeddings_corpus_idx ON chunk_embeddings(corpus_resource_name);
CREATE INDEX chunk_embeddings_external_chunk_idx ON chunk_embeddings(external_chunk_resource_name);

CREATE TABLE prompt_templates (
 prompt_template_uuid UUID PRIMARY KEY, template_id TEXT NOT NULL, stage TEXT NOT NULL,
 system_instruction TEXT, user_template TEXT NOT NULL, output_format TEXT, input_variables JSONB,
 model_name TEXT, thinking_mode TEXT, token_budget INTEGER, template_version INTEGER NOT NULL,
 is_active BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK (token_budget IS NULL OR token_budget>0), CHECK (template_version>0),
 CHECK (input_variables IS NULL OR jsonb_typeof(input_variables) IN ('object','array')),
 UNIQUE (template_id, stage, template_version)
);
CREATE INDEX prompt_templates_active_stage_idx ON prompt_templates(stage, is_active);

CREATE TABLE user_sessions (
 session_uuid UUID PRIMARY KEY, user_uuid UUID NOT NULL, session_token_hash TEXT NOT NULL UNIQUE,
 login_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, last_activity_at TIMESTAMPTZ,
 logout_at TIMESTAMPTZ, expires_at TIMESTAMPTZ NOT NULL, ip_address INET, user_agent TEXT,
 status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (user_uuid) REFERENCES users(user_uuid) ON DELETE CASCADE,
 CHECK (expires_at>login_at), CHECK (logout_at IS NULL OR logout_at>=login_at)
);
CREATE INDEX user_sessions_user_login_idx ON user_sessions(user_uuid, login_at DESC);
CREATE INDEX user_sessions_status_expiry_idx ON user_sessions(status, expires_at);

CREATE TABLE query_runs (
 query_run_uuid UUID PRIMARY KEY, user_uuid UUID, session_uuid UUID, prompt_template_uuid UUID,
 request_type TEXT NOT NULL, query_text TEXT NOT NULL, response_text TEXT, retrieved_chunks JSONB,
 metadata JSONB, status TEXT NOT NULL, latency_ms INTEGER, model_name TEXT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (user_uuid) REFERENCES users(user_uuid) ON DELETE SET NULL,
 FOREIGN KEY (session_uuid) REFERENCES user_sessions(session_uuid) ON DELETE SET NULL,
 FOREIGN KEY (prompt_template_uuid) REFERENCES prompt_templates(prompt_template_uuid) ON DELETE SET NULL,
 CHECK (latency_ms IS NULL OR latency_ms>=0),
 CHECK (retrieved_chunks IS NULL OR jsonb_typeof(retrieved_chunks)='array'),
 CHECK (metadata IS NULL OR jsonb_typeof(metadata)='object')
);
CREATE INDEX query_runs_user_created_idx ON query_runs(user_uuid, created_at DESC);
CREATE INDEX query_runs_session_created_idx ON query_runs(session_uuid, created_at DESC);
CREATE INDEX query_runs_status_created_idx ON query_runs(status, created_at DESC);
CREATE INDEX query_runs_prompt_template_idx ON query_runs(prompt_template_uuid);

CREATE TABLE citation_checks (
 citation_check_uuid UUID PRIMARY KEY, query_run_uuid UUID NOT NULL, chunk_uuid UUID,
 judgment_uuid UUID, verification_status TEXT NOT NULL, citation_text TEXT,
 verification_details JSONB, checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (query_run_uuid) REFERENCES query_runs(query_run_uuid) ON DELETE CASCADE,
 FOREIGN KEY (chunk_uuid) REFERENCES chunks(chunk_uuid) ON DELETE SET NULL,
 FOREIGN KEY (judgment_uuid) REFERENCES judgments(judgment_uuid) ON DELETE SET NULL,
 CHECK (verification_details IS NULL OR jsonb_typeof(verification_details)='object')
);
CREATE INDEX citation_checks_query_checked_idx ON citation_checks(query_run_uuid, checked_at DESC);
CREATE INDEX citation_checks_chunk_idx ON citation_checks(chunk_uuid);
CREATE INDEX citation_checks_judgment_idx ON citation_checks(judgment_uuid);
CREATE INDEX citation_checks_status_idx ON citation_checks(verification_status);

CREATE TABLE user_activity_logs (
 user_activity_log_uuid UUID PRIMARY KEY, user_uuid UUID, session_uuid UUID, event_type TEXT NOT NULL,
 action TEXT NOT NULL, resource_type TEXT, resource_id TEXT, ip_address INET, user_agent TEXT,
 metadata JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (user_uuid) REFERENCES users(user_uuid) ON DELETE SET NULL,
 FOREIGN KEY (session_uuid) REFERENCES user_sessions(session_uuid) ON DELETE SET NULL,
 CHECK (metadata IS NULL OR jsonb_typeof(metadata)='object')
);
CREATE INDEX user_activity_logs_user_created_idx ON user_activity_logs(user_uuid, created_at DESC);
CREATE INDEX user_activity_logs_session_created_idx ON user_activity_logs(session_uuid, created_at DESC);
CREATE INDEX user_activity_logs_event_created_idx ON user_activity_logs(event_type, created_at DESC);

CREATE TABLE audit_logs (
 audit_log_uuid UUID PRIMARY KEY, user_uuid UUID, session_uuid UUID, action TEXT NOT NULL,
 resource_type TEXT, resource_id TEXT, event_category TEXT NOT NULL, severity TEXT NOT NULL,
 status TEXT NOT NULL, error_message TEXT, metadata JSONB,
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY (user_uuid) REFERENCES users(user_uuid) ON DELETE SET NULL,
 FOREIGN KEY (session_uuid) REFERENCES user_sessions(session_uuid) ON DELETE SET NULL,
 CHECK (metadata IS NULL OR jsonb_typeof(metadata)='object')
);
CREATE INDEX audit_logs_user_created_idx ON audit_logs(user_uuid, created_at DESC);
CREATE INDEX audit_logs_category_created_idx ON audit_logs(event_category, created_at DESC);
CREATE INDEX audit_logs_severity_created_idx ON audit_logs(severity, created_at DESC);
CREATE INDEX audit_logs_resource_idx ON audit_logs(resource_type, resource_id);

CREATE TABLE outbox_events (
 outbox_event_uuid UUID PRIMARY KEY, aggregate_type TEXT NOT NULL, aggregate_uuid UUID NOT NULL,
 event_type TEXT NOT NULL, payload JSONB NOT NULL, delivery_status TEXT NOT NULL,
 attempt_count INTEGER NOT NULL DEFAULT 0, available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 published_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK (jsonb_typeof(payload)='object'), CHECK (attempt_count>=0),
 CHECK (published_at IS NULL OR published_at>=created_at)
);
CREATE INDEX outbox_events_delivery_available_idx ON outbox_events(delivery_status, available_at, created_at);
CREATE INDEX outbox_events_aggregate_idx ON outbox_events(aggregate_type, aggregate_uuid);
CREATE INDEX outbox_events_unpublished_idx ON outbox_events(available_at, created_at) WHERE published_at IS NULL;

COMMIT;
