"""
Hybrid Legal Retriever for Indian Supreme Court RAG System
Combines:
1. Semantic Dense Vector Search (pgvector with HNSW cosine distance)
2. Lexical Keyword Full-Text Search (PostgreSQL tsvector & websearch_to_tsquery)
3. Metadata Pre-Filtering (by Judge, Judgment Date/Year, Case Number, Section)
4. Reciprocal Rank Fusion (RRF) for unified, robust relevance ranking.

Engine: Google Cloud SQL for PostgreSQL 16 (database: "scrag")
Embedding Model: Google Vertex AI text-embedding-004 (768 dimensions)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [HybridRetriever] %(message)s")
logger = logging.getLogger("HybridRetriever")


HYBRID_SEARCH_RRF_SQL = text("""
WITH 
-- 1. Lexical Keyword Search (PostgreSQL FTS)
fts_results AS (
    SELECT 
        c.chunk_uuid,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(c.search_vector, websearch_to_tsquery('english', :query_text)) DESC, 
            c.chunk_uuid DESC
        ) AS rank
    FROM chunks c
    JOIN judgments j ON j.judgment_uuid = c.judgment_uuid
    WHERE c.search_vector @@ websearch_to_tsquery('english', :query_text)
      AND (:judge_filter IS NULL OR c.metadata->>'judges' ILIKE '%' || :judge_filter || '%')
      AND (:year_start IS NULL OR EXTRACT(YEAR FROM j.judgment_date) >= :year_start)
      AND (:year_end IS NULL OR EXTRACT(YEAR FROM j.judgment_date) <= :year_end)
    LIMIT 50
),

-- 2. Semantic Dense Vector Search (pgvector Cosine Distance)
vector_results AS (
    SELECT 
        c.chunk_uuid,
        ROW_NUMBER() OVER (
            ORDER BY c.embedding <=> :query_embedding ASC, 
            c.chunk_uuid DESC
        ) AS rank
    FROM chunks c
    JOIN judgments j ON j.judgment_uuid = c.judgment_uuid
    WHERE c.embedding IS NOT NULL
      AND (:judge_filter IS NULL OR c.metadata->>'judges' ILIKE '%' || :judge_filter || '%')
      AND (:year_start IS NULL OR EXTRACT(YEAR FROM j.judgment_date) >= :year_start)
      AND (:year_end IS NULL OR EXTRACT(YEAR FROM j.judgment_date) <= :year_end)
    ORDER BY c.embedding <=> :query_embedding ASC
    LIMIT 50
),

-- 3. Reciprocal Rank Fusion (RRF with standard smoothing factor k=60)
combined_ranks AS (
    SELECT 
        COALESCE(f.chunk_uuid, v.chunk_uuid) AS chunk_uuid,
        COALESCE(1.0 / (:rrf_k + f.rank), 0.0) + COALESCE(1.0 / (:rrf_k + v.rank), 0.0) AS rrf_score,
        f.rank AS fts_rank,
        v.rank AS vector_rank
    FROM fts_results f
    FULL OUTER JOIN vector_results v ON f.chunk_uuid = v.chunk_uuid
)

-- 4. Final Result Set with case metadata, citations, and scores
SELECT 
    c.chunk_uuid,
    c.chunk_index,
    c.page_start,
    c.page_end,
    c.section,
    c.content,
    c.metadata,
    j.judgment_id,
    j.case_number,
    j.judgment_date,
    j.citation_text,
    j.court,
    cr.rrf_score,
    cr.fts_rank,
    cr.vector_rank
FROM combined_ranks cr
JOIN chunks c ON c.chunk_uuid = cr.chunk_uuid
JOIN judgments j ON j.judgment_uuid = c.judgment_uuid
ORDER BY cr.rrf_score DESC
LIMIT :top_k;
""")


class HybridRetriever:
    """
    Production Hybrid Legal Search Engine for Indian Supreme Court Judgments.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        project_id: Optional[str] = None,
        region: str = "asia-south1",
        embedding_model_name: str = "text-embedding-004"
    ):
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL must be provided or set in the environment.")

        self.project_id = project_id or os.environ.get("GCP_PROJECT", "sc-judgments-rag")
        self.region = region
        self.embedding_model_name = embedding_model_name

        self.engine: Engine = create_engine(
            self.db_url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
            pool_pre_ping=True
        )

        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            try:
                import vertexai
                from vertexai.language_models import TextEmbeddingModel
                vertexai.init(project=self.project_id, location=self.region)
                self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
                logger.info(f"Initialized Vertex AI '{self.embedding_model_name}' in region {self.region}.")
            except Exception as e:
                logger.warning(f"Vertex AI initialization note: {e}")
                self._embedding_model = None
        return self._embedding_model

    def embed_query(self, text_query: str) -> Optional[List[float]]:
        """Generates 768-dimensional float vector for a user query."""
        if not self.embedding_model:
            logger.error("Vertex AI Embedding model is unavailable.")
            return None
        try:
            embeddings = self.embedding_model.get_embeddings([text_query])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None

    def search(
        self,
        query: str,
        judge_filter: Optional[str] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Executes an end-to-end Hybrid Search with RRF and metadata filters.
        """
        if not query or not query.strip():
            return []

        # 1. Generate query vector
        query_vector = self.embed_query(query)
        vector_str = json.dumps(query_vector) if query_vector else None

        # 2. Execute Hybrid SQL in Cloud SQL PostgreSQL
        params = {
            "query_text": query.strip(),
            "query_embedding": vector_str,
            "judge_filter": judge_filter.strip() if judge_filter else None,
            "year_start": year_start,
            "year_end": year_end,
            "top_k": top_k,
            "rrf_k": rrf_k
        }

        with self.engine.connect() as conn:
            result = conn.execute(HYBRID_SEARCH_RRF_SQL, params)
            records = []
            for row in result.fetchall():
                d = dict(row._mapping)
                if isinstance(d.get("metadata"), str):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except Exception:
                        pass
                records.append(d)
            return records

