import asyncio
from sqlalchemy.future import select
from sqlalchemy import delete
from typing import List, Dict, Any, Optional

from app.database import async_session
from app.models.schema import Repository, GraphNode, GraphEdge
from app.services.github_client import GitHubClient, RepoSizeExceededError, RepoTooComplexError
from app.config import settings
from datetime import datetime
from sqlalchemy import func

import logging
logger = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(self):
        pass
        
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

                # Get selected paths from the graph nodes instead of chunks
                stmt_nodes = select(GraphNode).where(GraphNode.repository_id == repo_obj.id, GraphNode.type == "file")
                res = await session.execute(stmt_nodes)
                nodes = res.scalars().all()

                selected_paths = [n.path for n in nodes]

                return {
                    "status": "cached",
                    "repo_url": repo_url,
                    "commit_sha": commit_sha,
                    "chunks_count": 0,
                    "selected_files": selected_paths,
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

                # Delete stale graph
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


