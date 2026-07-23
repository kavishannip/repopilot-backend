import asyncio
import os
import sys

# Ensure the root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.database import engine
from app.models.schema import Base
from sqlalchemy import text
from pgvector.sqlalchemy import HALFVEC

async def migrate():
    async with engine.begin() as conn:
        print("Enabling pgvector extension if not exists...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        print("Altering repo_chunks embedding column to halfvec(1536)...")
        try:
            await conn.execute(text("TRUNCATE TABLE repo_chunks;"))
            await conn.execute(text("ALTER TABLE repo_chunks ALTER COLUMN embedding TYPE halfvec(1536);"))
            print("Successfully migrated embedding to halfvec(1536).")
        except Exception as e:
            print(f"Migration error or already migrated (embedding): {e}")

        print("Adding last_accessed_at column to repositories...")
        try:
            await conn.execute(text("ALTER TABLE repositories ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMP;"))
            await conn.execute(text("UPDATE repositories SET last_accessed_at = updated_at WHERE last_accessed_at IS NULL;"))
            print("Successfully added last_accessed_at.")
        except Exception as e:
            print(f"Migration error or already migrated (last_accessed_at): {e}")

        print("Creating new tables (graph_nodes, graph_edges)...")
        await conn.run_sync(Base.metadata.create_all)
        print("Done!")

if __name__ == "__main__":
    asyncio.run(migrate())
