from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy import Index
import uuid

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
    last_accessed_at = Column(DateTime, default=datetime.utcnow) # For LRU cache eviction
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    nodes = relationship("GraphNode", back_populates="repository", cascade="all, delete-orphan")
    edges = relationship("GraphEdge", back_populates="repository", cascade="all, delete-orphan")



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

class GithubInstallation(Base):
    __tablename__ = "github_installations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    installation_id = Column(String, unique=True, index=True, nullable=False)
    account_name = Column(String, nullable=False)
    target_type = Column(String, nullable=False) # 'User' or 'Organization'
    repository_selection = Column(String, nullable=False) # 'all' or 'selected'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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


# Indexes for common lookups (created separately from table def for clarity)
__all__ = ["Base", "Repository", "GraphNode", "GraphEdge", "AnalysisJob", "GithubInstallation"]
