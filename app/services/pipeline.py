import asyncio
from sqlalchemy.future import select
from sqlalchemy import delete
from typing import List, Dict, Any, Optional

from app.database import async_session
from app.models.schema import Repository, RepoChunk
from app.services.github_client import GitHubClient, RepoSizeExceededError, RepoTooComplexError
from app.services.chunker import CodeChunker
from app.services.embedding import NVIDIAEmbeddingClient
from app.config import settings

import logging
logger = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(self):
        self.github = GitHubClient()
        self.chunker = CodeChunker()
        self.embedder = NVIDIAEmbeddingClient()

    async def process_repo(self, repo_url: str) -> Dict[str, Any]:
        """
        Process a repository: fetch, chunk, embed, store.
        Returns stats including selected_files and cached tree_data for the summarizer
        (avoids a second get_file_tree_structure() call).

        Raises:
            RepoSizeExceededError  — raw repo too large (KB)
            RepoTooComplexError    — important files exceed accuracy budget
        """
        print(f"Fetching metadata for {repo_url}...")
        meta = await self.github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]

        commit_sha = await self._get_branch_commit(repo_url, branch)

        async with async_session() as session:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()

            if repo_obj and repo_obj.commit_sha == commit_sha:
                print("Cache hit! Repository is up to date.")
                stmt_count = select(RepoChunk).where(RepoChunk.repository_id == repo_obj.id)
                res = await session.execute(stmt_count)
                chunks = res.scalars().all()

                selected_paths = set()
                for c in chunks:
                    path = c.chunk_metadata.get("path")
                    if path:
                        selected_paths.add(path)

                return {
                    "status": "cached",
                    "repo_url": repo_url,
                    "commit_sha": commit_sha,
                    "chunks_count": len(chunks),
                    "selected_files": list(selected_paths),
                    "tree_data": None,  # not re-fetched on cache hit; summarizer will fetch its own
                }

            if not repo_obj:
                repo_obj = Repository(
                    repo_url=repo_url,
                    owner=meta["owner"],
                    name=meta["name"],
                    default_branch=branch,
                    commit_sha=commit_sha,
                    language=meta["language"],
                    stars=meta["stars"],
                    description=meta["description"]
                )
                session.add(repo_obj)
            else:
                repo_obj.commit_sha = commit_sha
                repo_obj.language = meta["language"]
                repo_obj.stars = meta["stars"]
                repo_obj.description = meta["description"]

                # Delete stale chunks
                await session.execute(delete(RepoChunk).where(RepoChunk.repository_id == repo_obj.id))

            await session.commit()
            repo_id = repo_obj.id

        # ── Outside DB transaction: heavy I/O ──────────────────────────────

        # Single tree fetch — result shared with summarizer to avoid duplicate API call
        print("Fetching file tree...")
        tree_data = await self.github.get_file_tree_structure(repo_url, branch)
        classification = tree_data["classification"]
        file_sizes = tree_data.get("file_sizes", {})

        # Build ASCII tree for SmartFileSelector prompt
        from app.services.repo_map import RepoMap
        ascii_tree = RepoMap().build_ascii_tree(tree_data["nested_tree"], max_depth=4)

        # Raw file list from the already-fetched tree (pre-prioritized, pre-noise-filtered).
        # This avoids a duplicate GitHub tree request and lets smart selection see the
        # complete analyzable repo instead of a pre-truncated slice.
        files = tree_data.get("files")
        if files is None:
            print("Fetching file list...")
            files = await self.github.list_files(repo_url, branch=branch)

        # Smart file selection — may raise RepoTooComplexError
        from app.services.file_selector import SmartFileSelector
        selector = SmartFileSelector()
        files = await selector.select_files(
            all_files=files,
            classification=classification,
            ascii_tree=ascii_tree,
            file_sizes=file_sizes,
        )

        print(f"Processing {len(files)} selected files...")
        all_chunks = []

        contents = await self.github.get_files_content_batch(
            repo_url,
            [f["path"] for f in files],
            branch=branch,
            max_chars_per_file=settings.max_source_file_chars,
        )
        for f in files:
            content = contents.get(f["path"])
            if content:
                try:
                    all_chunks.extend(self.chunker.chunk_file(f["path"], content))
                except Exception as e:
                    logger.warning("Failed to chunk %s: %r", f["path"], e)

        print(f"Total chunks generated: {len(all_chunks)}")

        # Batch embedding
        embeddings = []
        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            print(f"Embedding {len(texts)} chunks...")
            embeddings = await self.embedder.embed_chunks(texts, input_type="passage")

        # Save chunks
        async with async_session() as session:
            if all_chunks and embeddings:
                print("Saving to database...")
                db_chunks = []
                for i, c in enumerate(all_chunks):
                    clean_text = c["text"].replace("\x00", "")
                    clean_metadata = {
                        k: v.replace("\x00", "") if isinstance(v, str) else v
                        for k, v in c["metadata"].items()
                    }
                    db_chunks.append(
                        RepoChunk(
                            repository_id=repo_id,
                            chunk_text=clean_text,
                            chunk_metadata=clean_metadata,
                            embedding=embeddings[i]
                        )
                    )
                session.add_all(db_chunks)

            await session.commit()
            print("Done processing repository.")

            selected_file_paths = [f["path"] for f in files]
            return {
                "status": "processed",
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "chunks_count": len(all_chunks),
                "selected_files": selected_file_paths,
                "tree_data": tree_data,   # passed to summarizer to skip second fetch
            }

    async def _get_branch_commit(self, repo_url: str, branch: str) -> str:
        owner, repo = self.github._parse_repo_url(repo_url)
        import httpx
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
                headers=self.github.headers
            )
            resp.raise_for_status()
            return resp.json()["sha"]

    async def search_chunks(self, query: str, repo_url: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embed a search query and find most similar chunks for a specific repo.
        """
        query_embedding = await self.embedder.embed_chunks([query], input_type="query")
        if not query_embedding:
            return []

        q_vec = query_embedding[0]

        async with async_session() as session:
            stmt_repo = select(Repository).where(Repository.repo_url == repo_url)
            repo_res = await session.execute(stmt_repo)
            repo_obj = repo_res.scalar_one_or_none()

            if not repo_obj:
                raise ValueError(f"Repository {repo_url} not found in database.")

            stmt = (
                select(RepoChunk, RepoChunk.embedding.cosine_distance(q_vec).label("distance"))
                .where(RepoChunk.repository_id == repo_obj.id)
                .order_by("distance")
                .limit(top_k)
            )

            res = await session.execute(stmt)
            results = []
            for chunk, distance in res:
                results.append({
                    "id": chunk.id,
                    "text": chunk.chunk_text,
                    "metadata": chunk.chunk_metadata,
                    "distance": distance
                })

            return results

    async def search_chunks_by_queries(
        self,
        queries: List[str],
        repo_url: str,
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch-embed multiple search queries and run per-query vector lookups.
        This saves one external embedding request per targeted architecture query.
        """
        if not queries:
            return {}

        query_embeddings = await self.embedder.embed_chunks(queries, input_type="query")
        if not query_embeddings:
            return {query: [] for query in queries}

        async with async_session() as session:
            stmt_repo = select(Repository).where(Repository.repo_url == repo_url)
            repo_res = await session.execute(stmt_repo)
            repo_obj = repo_res.scalar_one_or_none()

            if not repo_obj:
                raise ValueError(f"Repository {repo_url} not found in database.")

            results_by_query: Dict[str, List[Dict[str, Any]]] = {}
            for query, q_vec in zip(queries, query_embeddings):
                stmt = (
                    select(RepoChunk, RepoChunk.embedding.cosine_distance(q_vec).label("distance"))
                    .where(RepoChunk.repository_id == repo_obj.id)
                    .order_by("distance")
                    .limit(top_k)
                )
                res = await session.execute(stmt)
                results_by_query[query] = [
                    {
                        "id": chunk.id,
                        "text": chunk.chunk_text,
                        "metadata": chunk.chunk_metadata,
                        "distance": distance,
                    }
                    for chunk, distance in res
                ]

            return results_by_query

    async def get_repository_chunks(
        self,
        repo_url: str,
        max_text_chars_per_chunk: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Return all stored chunks for a processed repo.
        Used by deterministic static analysis so absence from semantic search does
        not become a false "unused dependency" or README contradiction.
        """
        async with async_session() as session:
            stmt_repo = select(Repository).where(Repository.repo_url == repo_url)
            repo_res = await session.execute(stmt_repo)
            repo_obj = repo_res.scalar_one_or_none()

            if not repo_obj:
                raise ValueError(f"Repository {repo_url} not found in database.")

            stmt = select(RepoChunk).where(RepoChunk.repository_id == repo_obj.id)
            res = await session.execute(stmt)
            chunks = res.scalars().all()

            return [
                {
                    "id": chunk.id,
                    "text": chunk.chunk_text[:max_text_chars_per_chunk],
                    "metadata": chunk.chunk_metadata or {},
                }
                for chunk in chunks
            ]
