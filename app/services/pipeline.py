import asyncio
from sqlalchemy.future import select
from sqlalchemy import delete
from typing import List, Dict, Any, Optional

from app.database import async_session
from app.models.schema import Repository, RepoChunk, GraphNode, GraphEdge
from app.services.github_client import GitHubClient, RepoSizeExceededError, RepoTooComplexError
from app.services.chunker import CodeChunker
from app.services.embedding import NVIDIAEmbeddingClient
from app.config import settings
from datetime import datetime
from sqlalchemy import func

import logging
logger = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(self):
        self.chunker = CodeChunker()
        self.embedder = NVIDIAEmbeddingClient()
        
    async def get_github_client(self, repo_url: str) -> GitHubClient:
        from app.database import async_session
        from sqlalchemy.future import select
        from app.models.schema import GithubInstallation
        
        # We need a dummy client just to parse the URL
        dummy = GitHubClient()
        owner, _ = dummy._parse_repo_url(repo_url)
        
        async with async_session() as session:
            stmt = select(GithubInstallation).where(GithubInstallation.account_name == owner)
            result = await session.execute(stmt)
            install = result.scalar_one_or_none()
            if install:
                from app.services.github_auth import github_auth
                token = await github_auth.get_installation_token(install.installation_id)
                return GitHubClient(installation_token=token)
        return GitHubClient()

    async def process_repo(self, repo_url: str) -> Dict[str, Any]:
        """
        Process a repository: fetch, chunk, embed, store.
        Returns stats including selected_files and cached tree_data for the summarizer
        (avoids a second get_file_tree_structure() call).

        Raises:
            RepoSizeExceededError  — raw repo too large (KB)
            RepoTooComplexError    — important files exceed accuracy budget
        """
        github = await self.get_github_client(repo_url)
        
        print(f"Fetching metadata for {repo_url}...")
        meta = await github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]

        commit_sha = await self._get_branch_commit(github, repo_url, branch)

        async with async_session() as session:
            # --- LRU Cache Eviction ---
            count_stmt = select(func.count(Repository.id))
            repo_count = await session.scalar(count_stmt)
            if repo_count and repo_count > 25:
                logger.info("Repo count > 25, evicting oldest repository to stay under 500MB budget.")
                oldest_stmt = select(Repository).order_by(Repository.last_accessed_at.asc()).limit(1)
                oldest_repo = (await session.execute(oldest_stmt)).scalar_one_or_none()
                if oldest_repo:
                    await session.delete(oldest_repo)
                    await session.commit()

            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()

            if repo_obj and repo_obj.commit_sha == commit_sha:
                print("Cache hit! Repository is up to date.")
                repo_obj.last_accessed_at = datetime.utcnow()
                await session.commit()

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
                    description=meta["description"],
                    last_accessed_at=datetime.utcnow()
                )
                session.add(repo_obj)
            else:
                repo_obj.commit_sha = commit_sha
                repo_obj.language = meta["language"]
                repo_obj.stars = meta["stars"]
                repo_obj.description = meta["description"]
                repo_obj.last_accessed_at = datetime.utcnow()

                # Delete stale chunks and graph
                await session.execute(delete(RepoChunk).where(RepoChunk.repository_id == repo_obj.id))
                await session.execute(delete(GraphNode).where(GraphNode.repository_id == repo_obj.id))
                await session.execute(delete(GraphEdge).where(GraphEdge.repository_id == repo_obj.id))

            await session.commit()
            repo_id = repo_obj.id

        # ── Outside DB transaction: heavy I/O ──────────────────────────────

        # Single tree fetch — result shared with summarizer to avoid duplicate API call
        print("Fetching file tree...")
        tree_data = await github.get_file_tree_structure(repo_url, branch)
        classification = tree_data["classification"]
        file_sizes = tree_data.get("file_sizes", {})

        # Build ASCII tree for SmartFileSelector prompt
        from app.services.repo_map import RepoMap
        ascii_tree = RepoMap().build_ascii_tree(tree_data["nested_tree"], max_depth=4)

        # Raw file list from the already-fetched tree (pre-prioritized, pre-noise-filtered).
        # This avoids a duplicate GitHub tree request and lets smart selection see the
        # complete analyzable repo instead of a pre-truncated slice.
        all_files = tree_data.get("files")
        if all_files is None:
            print("Fetching file list...")
            all_files = await github.list_files(repo_url, branch=branch)

        # Smart file selection — may raise RepoTooComplexError
        from app.services.file_selector import SmartFileSelector
        selector = SmartFileSelector()
        selected_files = await selector.select_files(
            all_files=all_files,
            classification=classification,
            ascii_tree=ascii_tree,
            file_sizes=file_sizes,
        )

        selected_paths = {f["path"] for f in selected_files}

        # --- Structural Scan ---
        print("Running full-repo structural scan...")
        from app.services.structural_scanner import StructuralScanner
        scanner = StructuralScanner()
        scan_files = [
            f for f in all_files 
            if f.get("type", "file") in ("file", "blob") and f["path"].endswith((
                ".py", ".ts", ".tsx", ".js", ".jsx", 
                ".go", ".java", ".kt", ".rs", ".c", ".cpp", ".h", ".hpp", 
                ".cs", ".rb", ".php", ".swift", ".m"
            ))
        ]
        
        scan_contents = await github.get_files_content_batch(
            repo_url,
            [f["path"] for f in scan_files[:3000]],
            branch=branch,
            max_chars_per_file=50_000, # Lightweight fetch for imports
        )
        
        nodes, edges = scanner.scan(repo_id, all_files, scan_contents, selected_paths)

        # Save graph
        async with async_session() as session:
            if nodes:
                session.add_all(nodes)
                await session.flush()
            if edges:
                session.add_all(edges)

            await session.commit()
            print("Done processing repository.")

            selected_file_paths = [f["path"] for f in selected_files]
            return {
                "status": "processed",
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "chunks_count": 0,
                "selected_files": selected_file_paths,
                "tree_data": tree_data,   # passed to summarizer to skip second fetch
            }

    async def _get_branch_commit(self, github: GitHubClient, repo_url: str, branch: str) -> str:
        owner, repo = github._parse_repo_url(repo_url)
        import httpx
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
                headers=github.headers
            )
            resp.raise_for_status()
            return resp.json()["sha"]

    async def search_chunks(self, query: str, repo_url: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
