# ml-assistant/api/db.py
import os
import uuid
from typing import List

from sqlalchemy import create_engine, Integer, Text, ForeignKey, text, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy import select

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@db:5432/ml_assistant",
)

# Upewnij się, że wymiar odpowiada modelowi z embed.py (bge-small-en-v1.5 => 384)
EMBED_DIM = 384

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)

    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(EMBED_DIM), nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")

    @staticmethod
    def select_content_by_embedding(q_vec: List[float], limit: int = 6):
        """
        Zwraca SELECT sortujący po cosine distance (pgvector), bez ręcznego CAST.
        """
        return (
            select(Chunk.content)
            .order_by(Chunk.embedding.cosine_distance(q_vec))
            .limit(limit)
        )

    @staticmethod
    def select_contexts_by_embedding(q_vec: List[float], limit: int = 6):
        return (
            select(Document.source, Chunk.chunk_index, Chunk.content)
            .join(Document, Chunk.document_id == Document.id)
            .order_by(Chunk.embedding.cosine_distance(q_vec))
            .limit(limit)
        )


def _create_extensions_and_indexes(conn):
    # Extension
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # Indeks po dokumencie
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (document_id)"))
    # ivfflat dla cosine ops
    conn.execute(
        text(
            """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'idx_chunks_embedding_ivfflat'
            ) THEN
                CREATE INDEX idx_chunks_embedding_ivfflat
                ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            END IF;
        END$$;
        """
        )
    )


def init_db():
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        _create_extensions_and_indexes(conn)
