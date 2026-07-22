from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, UniqueConstraint, Index
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
import uuid
from pgvector.sqlalchemy import HALFVEC

Base = declarative_base()

class Repository(Base):
    __tablename__ = "repositories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    repo_url = Column(String, unique=True, index=True, nullable=False)
    owner = Column(String, nullable=False)
    name = Column(String, nullable=False)
    default_branch = Column(String)
    commit_sha = Column(String) 
    language = Column(String)
    stars = Column(Integer)
    description = Column(Text)
    brief = Column(JSONB)                  # Cached Phase 4 RepoBrief
    quick_summary = Column(Text)            # Fast AI summary
    group_summaries = Column(JSONB)         # Cached Phase 3 group summaries
    group_summaries_sha = Column(String)    # Commit SHA that produced group_summaries
    last_accessed_at = Column(DateTime, default=datetime.utcnow) # For LRU cache eviction
    last_commit_check_at = Column(DateTime, default=datetime.utcnow) # 5-min commit check TTL
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("RepoChunk", back_populates="repository", cascade="all, delete-orphan")
    nodes = relationship("GraphNode", back_populates="repository", cascade="all, delete-orphan")
    edges = relationship("GraphEdge", back_populates="repository", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="repository", cascade="all, delete-orphan")


class RepoChunk(Base):
    __tablename__ = "repo_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String, index=True, nullable=False, default="")
    chunk_index = Column(Integer, default=0, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(HALFVEC(1536)) # OpenAI text-embedding-3-small (storage optimized)
    chunk_metadata = Column(JSONB)

    repository = relationship("Repository", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("repository_id", "file_path", "chunk_index", name="uq_repo_chunk_file_pos"),
        Index("ix_repo_chunks_repo_filepath", "repository_id", "file_path"),
    )


class GraphNode(Base):
    __tablename__ = "graph_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True)
    path = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False) # 'folder' or 'file'
    deep_analysis = Column(Boolean, default=False)

    repository = relationship("Repository", back_populates="nodes")


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String, nullable=False) # 'contains' or 'imports'

    repository = relationship("Repository", back_populates="edges")


class AnalysisJob(Base):
    """
    DB-backed async job state. Survives process restarts / free-tier idle spin-down.
    Jobs older than TTL_HOURS are cleaned up on each new job creation.
    """
    __tablename__ = "analysis_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_url = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending|running|completed|failed
    phase = Column(String, default="queued")                    # queued|embedding|structural_map|search|group_summaries|synthesis
    result = Column(JSONB, nullable=True)                       # RepoBrief JSON on completion
    error = Column(Text, nullable=True)                         # error message on failure
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatSession(Base):
    """
    Multi-turn conversation session bound to a repository and owned by a user_id (client_id).
    """
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, index=True, nullable=False)
    title = Column(String, default="Chat Session")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    repository = relationship("Repository", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """
    Individual message in a chat session.
    """
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


__all__ = [
    "Base", "Repository", "RepoChunk", "GraphNode", "GraphEdge", 
    "AnalysisJob", "ChatSession", "ChatMessage"
]
