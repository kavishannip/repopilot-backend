from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
import uuid
from pgvector.sqlalchemy import Vector

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
    brief = Column(JSONB) # Cached brief
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("RepoChunk", back_populates="repository", cascade="all, delete-orphan")

class RepoChunk(Base):
    __tablename__ = "repo_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(2048)) # Nvidia nemotron-3-embed-1b 
    chunk_metadata = Column(JSONB)

    repository = relationship("Repository", back_populates="chunks")
