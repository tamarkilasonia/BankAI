"""Knowledge chunks for RAG: pgvector extension, table and HNSW index.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-01 22:55:01.408075

Notes:
* ``CREATE EXTENSION IF NOT EXISTS vector`` needs superuser rights the first time
  (pgvector is not a "trusted" extension). In Docker the application user is the
  database superuser; on a local PostgreSQL run ``CREATE EXTENSION vector`` once
  as a superuser - afterwards this statement is a no-op.
* The downgrade does not drop the extension: it is database-level infrastructure,
  and dropping it would need superuser rights.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSIONS = 768  # frozen copy: migrations must not change when app constants do


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.String(length=200), nullable=False),
        sa.Column("source", sa.String(length=120), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("section", sa.String(length=200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_chunks")),
        sa.UniqueConstraint(
            "embedding_model", "chunk_id", name="uq_knowledge_chunks_embedding_model_chunk_id"
        ),
    )
    op.create_index(
        "ix_knowledge_chunks_embedding_hnsw",
        "knowledge_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        op.f("ix_knowledge_chunks_embedding_model"), "knowledge_chunks", ["embedding_model"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_knowledge_chunks_embedding_model"), table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_embedding_hnsw", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
