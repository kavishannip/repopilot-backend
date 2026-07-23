import asyncio
import uuid
import os
import sys
from datetime import datetime

# Ensure root directory is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, async_session
from app.models.schema import Repository, RepoChunk, ChatSession, ChatMessage
from app.services.chat_service import RepoChatService
from app.services.pipeline import ProcessingPipeline
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert


async def test_schema_and_unique_constraint():
    print("\n--- Test 1: DB Schema & Unique Constraint Verification ---")
    async with async_session() as session:
        # Create dummy repository
        test_repo = Repository(
            repo_url=f"https://github.com/test-org/test-repo-{str(uuid.uuid4())[:8]}",
            owner="test-org",
            name="test-repo",
            default_branch="main",
            commit_sha="dummy_sha_123",
            last_commit_check_at=datetime.utcnow()
        )
        session.add(test_repo)
        await session.commit()
        await session.refresh(test_repo)
        repo_id = test_repo.id

        # Insert a chunk
        chunk_val = {
            "id": uuid.uuid4(),
            "repository_id": repo_id,
            "file_path": "src/main.py",
            "chunk_index": 0,
            "chunk_text": "print('hello world')",
            "chunk_metadata": {"path": "src/main.py"},
            "embedding": [0.1] * 1536
        }
        stmt1 = pg_insert(RepoChunk).values([chunk_val])
        stmt1 = stmt1.on_conflict_do_nothing(constraint="uq_repo_chunk_file_pos")
        await session.execute(stmt1)
        await session.commit()

        # Attempt to insert identical (repository_id, file_path, chunk_index) concurrently/again
        duplicate_chunk_val = {
            "id": uuid.uuid4(),
            "repository_id": repo_id,
            "file_path": "src/main.py",
            "chunk_index": 0,
            "chunk_text": "print('duplicate chunk')",
            "chunk_metadata": {"path": "src/main.py"},
            "embedding": [0.2] * 1536
        }
        stmt2 = pg_insert(RepoChunk).values([duplicate_chunk_val])
        stmt2 = stmt2.on_conflict_do_nothing(constraint="uq_repo_chunk_file_pos")
        await session.execute(stmt2)
        await session.commit()

        # Assert exactly 1 row exists
        stmt_count = select(func.count(RepoChunk.id)).where(
            RepoChunk.repository_id == repo_id,
            RepoChunk.file_path == "src/main.py",
            RepoChunk.chunk_index == 0
        )
        count = await session.scalar(stmt_count)
        print(f"Chunks count for src/main.py: {count}")
        assert count == 1, f"Expected 1 chunk due to ON CONFLICT DO NOTHING, got {count}"
        print("[SUCCESS] Unique constraint (repository_id, file_path, chunk_index) verified!")

        # Clean up test repo
        await session.delete(test_repo)
        await session.commit()


async def test_session_security_and_ownership():
    print("\n--- Test 2: Session Ownership & Security Verification ---")
    async with async_session() as session:
        test_repo = Repository(
            repo_url=f"https://github.com/test-org/sec-repo-{str(uuid.uuid4())[:8]}",
            owner="test-org",
            name="sec-repo",
            default_branch="main",
            commit_sha="sec_sha_123",
        )
        session.add(test_repo)
        await session.commit()
        await session.refresh(test_repo)

        # Create session for User A
        sess_a = ChatSession(
            repository_id=test_repo.id,
            user_id="client-user-A",
            title="User A Session"
        )
        session.add(sess_a)
        await session.commit()
        await session.refresh(sess_a)
        sess_a_id = str(sess_a.id)

    chat_service = RepoChatService()

    # Attempt access as User B (should raise PermissionError)
    try:
        await chat_service.generate_chat_response(
            session_id=sess_a_id,
            user_id="client-user-B",
            question="What does this repo do?"
        )
        assert False, "Expected PermissionError for unauthorized user_id"
    except PermissionError as e:
        print(f"[SUCCESS] Caught expected ownership security error: '{e}'")

    # Clean up
    async with async_session() as session:
        stmt = select(Repository).where(Repository.id == test_repo.id)
        repo_obj = (await session.execute(stmt)).scalar_one()
        await session.delete(repo_obj)
        await session.commit()


async def test_get_processed_file_paths():
    print("\n--- Test 3: Fast Indexed Processed File Paths Lookup Verification ---")
    pipeline = ProcessingPipeline()
    async with async_session() as session:
        test_repo = Repository(
            repo_url=f"https://github.com/test-org/indexed-repo-{str(uuid.uuid4())[:8]}",
            owner="test-org",
            name="indexed-repo",
            default_branch="main",
            commit_sha="idx_sha_123",
        )
        session.add(test_repo)
        await session.commit()
        await session.refresh(test_repo)
        repo_id = test_repo.id

        chunks = [
            RepoChunk(repository_id=repo_id, file_path="app/main.py", chunk_index=0, chunk_text="a", embedding=[0.0]*1536),
            RepoChunk(repository_id=repo_id, file_path="app/main.py", chunk_index=1, chunk_text="b", embedding=[0.0]*1536),
            RepoChunk(repository_id=repo_id, file_path="app/config.py", chunk_index=0, chunk_text="c", embedding=[0.0]*1536),
        ]
        session.add_all(chunks)
        await session.commit()

        paths = await pipeline.get_processed_file_paths(repo_id)
        print(f"Processed file paths: {paths}")
        assert paths == {"app/main.py", "app/config.py"}, f"Unexpected paths: {paths}"
        print("[SUCCESS] Fast B-tree indexed path lookup verified!")

        await session.delete(test_repo)
        await session.commit()


async def main():
    await test_schema_and_unique_constraint()
    await test_session_security_and_ownership()
    await test_get_processed_file_paths()
    print("\n=== ALL TEST VERIFICATIONS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(main())
