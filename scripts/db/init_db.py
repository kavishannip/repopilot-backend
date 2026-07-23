import asyncio
import os
import sys

# Ensure the root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.database import engine
from app.models.schema import Base
from sqlalchemy import text

async def init_db():
    async with engine.begin() as conn:
        print("Enabling pgvector extension if not exists...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
        print("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)
        
        print("Applying column migrations for existing tables...")
        await conn.execute(text("ALTER TABLE repositories ADD COLUMN IF NOT EXISTS last_commit_check_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW();"))
        await conn.execute(text("ALTER TABLE repo_chunks ADD COLUMN IF NOT EXISTS file_path VARCHAR DEFAULT '';"))
        await conn.execute(text("ALTER TABLE repo_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;"))
        
        print("Backfilling file_path from chunk_metadata for existing rows...")
        await conn.execute(text("UPDATE repo_chunks SET file_path = COALESCE(chunk_metadata->>'path', '') WHERE file_path = '' OR file_path IS NULL;"))
        
        print("Cleaning up old duplicate chunks if any...")
        await conn.execute(text("""
            DELETE FROM repo_chunks 
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (PARTITION BY repository_id, file_path, chunk_index ORDER BY id) as rnum 
                    FROM repo_chunks
                ) t WHERE t.rnum > 1
            );
        """))

        print("Adding indexes and unique constraints...")
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_repo_chunks_file_path ON repo_chunks (file_path);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_repo_chunks_repo_filepath ON repo_chunks (repository_id, file_path);"))
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'uq_repo_chunk_file_pos'
                ) THEN
                    ALTER TABLE repo_chunks ADD CONSTRAINT uq_repo_chunk_file_pos UNIQUE (repository_id, file_path, chunk_index);
                END IF;
            END $$;
        """))
        print("Done!")

if __name__ == "__main__":
    asyncio.run(init_db())
