import asyncio
from sqlalchemy.future import select
from sqlalchemy import delete, func, distinct, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from typing import List, Dict, Any, Optional, Set
import uuid

from app.database import async_session
from app.models.schema import Repository, RepoChunk, GraphNode, GraphEdge
from app.services.github_client import GitHubClient, RepoSizeExceededError, RepoTooComplexError
from app.services.chunker import CodeChunker
from app.services.embedding import OpenAIEmbeddingClient
from app.config import settings
from datetime import datetime

import logging
logger = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(self):
        self.github = GitHubClient()
        self.chunker = CodeChunker()
        self.embedder = OpenAIEmbeddingClient()

    async def process_repo(self, repo_url: str) -> Dict[str, Any]:
        """
        Process a repository: fetch, chunk, embed, store.
        Returns stats including selected_files and cached tree_data for the summarizer.
        """
        print(f"Fetching metadata for {repo_url}...")
        meta = await self.github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]

        commit_sha = await self._get_branch_commit(repo_url, branch)

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
                stmt_count = select(RepoChunk).where(RepoChunk.repository_id == repo_obj.id)
                res = await session.execute(stmt_count)
                chunks = res.scalars().all()

                if len(chunks) > 0:
                    print("Cache hit! Repository is up to date.")
                    repo_obj.last_accessed_at = datetime.utcnow()
                    repo_obj.last_commit_check_at = datetime.utcnow()
                    await session.commit()

                selected_paths = set()
                for c in chunks:
                    path = c.file_path or c.chunk_metadata.get("path", "")
                    if path:
                        selected_paths.add(path)

                return {
                    "status": "cached",
                    "repo_url": repo_url,
                    "commit_sha": commit_sha,
                    "chunks_count": len(chunks),
                    "selected_files": list(selected_paths),
                    "tree_data": None,
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
                    last_accessed_at=datetime.utcnow(),
                    last_commit_check_at=datetime.utcnow()
                )
                session.add(repo_obj)
            else:
                repo_obj.commit_sha = commit_sha
                repo_obj.language = meta["language"]
                repo_obj.stars = meta["stars"]
                repo_obj.description = meta["description"]
                repo_obj.last_accessed_at = datetime.utcnow()
                repo_obj.last_commit_check_at = datetime.utcnow()

                # Delete stale chunks and graph
                await session.execute(delete(RepoChunk).where(RepoChunk.repository_id == repo_obj.id))
                await session.execute(delete(GraphNode).where(GraphNode.repository_id == repo_obj.id))
                await session.execute(delete(GraphEdge).where(GraphEdge.repository_id == repo_obj.id))

            await session.commit()
            repo_id = repo_obj.id

        # ── Outside DB transaction: heavy I/O ──────────────────────────────

        print("Fetching file tree...")
        tree_data = await self.github.get_file_tree_structure(repo_url, branch)
        classification = tree_data["classification"]
        file_sizes = tree_data.get("file_sizes", {})

        from app.services.repo_map import RepoMap
        ascii_tree = RepoMap().build_ascii_tree(tree_data["nested_tree"], max_depth=4)

        all_files = tree_data.get("files")
        if all_files is None:
            print("Fetching file list...")
            all_files = await self.github.list_files(repo_url, branch=branch)

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
        nodes, edges = [], []
        if settings.enable_graph_analysis:
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
            
            scan_contents = await self.github.get_files_content_batch(
                repo_url,
                [f["path"] for f in scan_files[:3000]],
                branch=branch,
                max_chars_per_file=50_000,
            )
            
            nodes, edges = scanner.scan(repo_id, all_files, scan_contents, selected_paths)
        else:
            print("Full-repo structural scan SKIPPED (enable_graph_analysis=False)")

        print(f"Processing {len(selected_files)} selected files for deep embedding...")
        all_chunks = []

        contents = await self.github.get_files_content_batch(
            repo_url,
            [f["path"] for f in selected_files],
            branch=branch,
            max_chars_per_file=settings.max_source_file_chars,
        )
        for f in selected_files:
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
            print(f"Embedding {len(texts)} chunks (OpenAI text-embedding-3-small, 1536-dim)...")
            embeddings = await self.embedder.embed_chunks(texts, input_type="passage")

        # Save chunks and graph with ON CONFLICT DO NOTHING
        async with async_session() as session:
            if nodes:
                session.add_all(nodes)
                await session.flush()
            if edges:
                session.add_all(edges)

            if all_chunks and embeddings:
                print("Saving to database with ON CONFLICT DO NOTHING...")
                file_counters = {}
                chunk_values = []
                for i, c in enumerate(all_chunks):
                    clean_text = c["text"].replace("\x00", "")
                    clean_metadata = {
                        k: v.replace("\x00", "") if isinstance(v, str) else v
                        for k, v in c["metadata"].items()
                    }
                    fpath = c["metadata"].get("path", "")
                    c_idx = file_counters.get(fpath, 0)
                    file_counters[fpath] = c_idx + 1

                    chunk_values.append({
                        "id": uuid.uuid4(),
                        "repository_id": repo_id,
                        "file_path": fpath,
                        "chunk_index": c_idx,
                        "chunk_text": clean_text,
                        "chunk_metadata": clean_metadata,
                        "embedding": embeddings[i],
                    })

                stmt = pg_insert(RepoChunk).values(chunk_values)
                stmt = stmt.on_conflict_do_nothing(constraint="uq_repo_chunk_file_pos")
                await session.execute(stmt)

            await session.commit()
            print("Done processing repository.")

            selected_file_paths = [f["path"] for f in selected_files]
            return {
                "status": "processed",
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "chunks_count": len(all_chunks),
                "selected_files": selected_file_paths,
                "tree_data": tree_data,
            }

    async def get_processed_file_paths(self, repository_id: uuid.UUID) -> Set[str]:
        """
        Fast B-tree indexed lookup of file paths already processed into repo_chunks.
        """
        async with async_session() as session:
            stmt = select(distinct(RepoChunk.file_path)).where(RepoChunk.repository_id == repository_id)
            res = await session.execute(stmt)
            return set(res.scalars().all())

    async def check_staleness(self, repo_url: str) -> Dict[str, Any]:
        """
        Check if repository in DB has newer commits on GitHub without performing sync.
        Compares both commit_sha and repo pushed_at timestamp against DB updated_at.
        """
        async with async_session() as session:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            repo_obj = (await session.execute(stmt)).scalar_one_or_none()
            if not repo_obj or not repo_obj.commit_sha:
                return {
                    "is_stale": False,
                    "cached_sha": None,
                    "latest_sha": None,
                    "changed_files_count": 0,
                    "diff_summary": None,
                }

            branch = repo_obj.default_branch or "main"
            try:
                latest_sha = await self._get_branch_commit(repo_url, branch)
                meta = await self.github.get_repo_metadata(repo_url)
                pushed_at_str = meta.get("pushed_at")
            except Exception as e:
                logger.warning(f"Failed to fetch branch commit for staleness check: {e}")
                return {
                    "is_stale": False,
                    "cached_sha": repo_obj.commit_sha,
                    "latest_sha": repo_obj.commit_sha,
                    "changed_files_count": 0,
                    "diff_summary": None,
                }

            # 1. SHA Match Check
            if repo_obj.commit_sha == latest_sha:
                return {
                    "is_stale": False,
                    "cached_sha": repo_obj.commit_sha,
                    "latest_sha": latest_sha,
                    "changed_files_count": 0,
                    "diff_summary": None,
                }

            # 2. Timestamp Comparison: If last analysis time >= GitHub pushed_at, repo is up to date
            if repo_obj.updated_at and pushed_at_str:
                from datetime import datetime, timezone
                try:
                    pushed_dt = datetime.fromisoformat(pushed_at_str.replace("Z", "+00:00"))
                    updated_dt = repo_obj.updated_at
                    if updated_dt.tzinfo is None:
                        updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                    
                    if updated_dt >= pushed_dt:
                        return {
                            "is_stale": False,
                            "cached_sha": repo_obj.commit_sha,
                            "latest_sha": latest_sha,
                            "changed_files_count": 0,
                            "diff_summary": None,
                        }
                except Exception as ex:
                    logger.warning(f"Error parsing pushed_at timestamp: {ex}")

            is_stale = True
            diff_summary = None
            changed_count = 0

            try:
                diff = await self.github.get_commit_diff(repo_url, repo_obj.commit_sha, latest_sha)
                added_cnt = len(diff.get("added", []))
                modified_cnt = len(diff.get("modified", []))
                removed_cnt = len(diff.get("removed", []))
                changed_count = added_cnt + modified_cnt + removed_cnt
                diff_summary = {
                    "added": added_cnt,
                    "modified": modified_cnt,
                    "removed": removed_cnt,
                }
            except Exception as e:
                logger.warning(f"Failed to fetch commit diff for staleness check: {e}")
                changed_count = -1

            return {
                "is_stale": is_stale,
                "cached_sha": repo_obj.commit_sha,
                "latest_sha": latest_sha,
                "changed_files_count": changed_count,
                "diff_summary": diff_summary,
            }

    async def check_and_sync_commit(self, repo_url: str, force: bool = False) -> bool:
        """
        Incremental commit synchronization with 5-min TTL cache check.
        Only processes modified/added/removed files. Unchanged files remain untouched in DB.
        Returns True if new commit sync occurred, False otherwise.
        """
        async with async_session() as session:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            repo_obj = (await session.execute(stmt)).scalar_one_or_none()
            if not repo_obj:
                return False

            now = datetime.utcnow()
            if not force and repo_obj.last_commit_check_at:
                elapsed = (now - repo_obj.last_commit_check_at).total_seconds()
                if elapsed < 300: # 5-min TTL
                    return False

            repo_obj.last_commit_check_at = now
            await session.commit()
            repo_id = repo_obj.id
            old_sha = repo_obj.commit_sha
            branch = repo_obj.default_branch or "main"

        # Check latest branch commit
        latest_sha = await self._get_branch_commit(repo_url, branch)
        if not old_sha or old_sha == latest_sha:
            return False

        print(f"[Incremental Sync] Comparing commits {old_sha[:8]} -> {latest_sha[:8]} for {repo_url}...")
        try:
            diff = await self.github.get_commit_diff(repo_url, old_sha, latest_sha)
        except Exception as e:
            logger.warning(f"Failed to fetch commit diff: {e}. Skipping incremental sync.")
            return False

        removed = diff.get("removed", [])
        modified = diff.get("modified", [])
        added = diff.get("added", [])

        if not removed and not modified and not added:
            async with async_session() as session:
                stmt = select(Repository).where(Repository.id == repo_id)
                repo = (await session.execute(stmt)).scalar_one()
                repo.commit_sha = latest_sha
                await session.commit()
            return False

        # Fan-in sorting if changed files > 150
        changed_targets = modified + added
        if len(changed_targets) > 150:
            logger.info(f"Changed files count ({len(changed_targets)}) > 150 budget. Sorting by fan-in graph score...")
            async with async_session() as session:
                # Count incoming edges per path
                stmt_in = (
                    select(GraphNode.path, func.count(GraphEdge.id).label("fan_in"))
                    .join(GraphEdge, GraphEdge.target_node_id == GraphNode.id)
                    .where(GraphNode.repository_id == repo_id)
                    .group_by(GraphNode.path)
                )
                res_in = await session.execute(stmt_in)
                fan_in_map = {path: cnt for path, cnt in res_in}

            def sort_key(p: str):
                depth = len(p.split("/"))
                score = fan_in_map.get(p, 0)
                return (-score, depth, p)

            changed_targets = sorted(changed_targets, key=sort_key)[:150]
            # Split back into modified and added
            mod_set = set(modified)
            modified = [p for p in changed_targets if p in mod_set]
            added = [p for p in changed_targets if p not in mod_set]

        print(f"[Incremental Sync] Updating {len(removed)} removed, {len(modified)} modified, {len(added)} added files...")

        async with async_session() as session:
            # 1. Delete chunks & nodes for removed files
            if removed:
                await session.execute(
                    delete(RepoChunk).where(
                        RepoChunk.repository_id == repo_id,
                        RepoChunk.file_path.in_(removed)
                    )
                )
                # Find nodes for removed files
                stmt_nodes = select(GraphNode).where(
                    GraphNode.repository_id == repo_id,
                    GraphNode.path.in_(removed)
                )
                rem_nodes = (await session.execute(stmt_nodes)).scalars().all()
                rem_node_ids = [n.id for n in rem_nodes]
                if rem_node_ids:
                    await session.execute(delete(GraphEdge).where(
                        (GraphEdge.source_node_id.in_(rem_node_ids)) | 
                        (GraphEdge.target_node_id.in_(rem_node_ids))
                    ))
                    await session.execute(delete(GraphNode).where(GraphNode.id.in_(rem_node_ids)))

            # 2. Delete chunks for modified files (they will be re-chunked/embedded below)
            if modified:
                await session.execute(
                    delete(RepoChunk).where(
                        RepoChunk.repository_id == repo_id,
                        RepoChunk.file_path.in_(modified)
                    )
                )
                # Surgical graph update for modified files
                stmt_nodes = select(GraphNode).where(
                    GraphNode.repository_id == repo_id,
                    GraphNode.path.in_(modified)
                )
                mod_nodes = (await session.execute(stmt_nodes)).scalars().all()
                mod_node_ids = [n.id for n in mod_nodes]
                if mod_node_ids:
                    await session.execute(delete(GraphEdge).where(
                        (GraphEdge.source_node_id.in_(mod_node_ids)) | 
                        (GraphEdge.target_node_id.in_(mod_node_ids))
                    ))

            await session.commit()

        # Fetch and process modified + added files
        process_paths = modified + added
        if process_paths:
            contents = await self.github.get_files_content_batch(
                repo_url,
                process_paths,
                branch=branch,
                max_chars_per_file=settings.max_source_file_chars,
            )

            new_chunks = []
            for path in process_paths:
                cnt = contents.get(path)
                if cnt:
                    try:
                        new_chunks.extend(self.chunker.chunk_file(path, cnt))
                    except Exception as e:
                        logger.warning(f"Failed to chunk {path}: {e}")

            if new_chunks:
                texts = [c["text"] for c in new_chunks]
                embeddings = await self.embedder.embed_chunks(texts, input_type="passage")

                async with async_session() as session:
                    file_counters = {}
                    chunk_values = []
                    for i, c in enumerate(new_chunks):
                        clean_text = c["text"].replace("\x00", "")
                        clean_metadata = {
                            k: v.replace("\x00", "") if isinstance(v, str) else v
                            for k, v in c["metadata"].items()
                        }
                        fpath = c["metadata"].get("path", "")
                        c_idx = file_counters.get(fpath, 0)
                        file_counters[fpath] = c_idx + 1

                        chunk_values.append({
                            "id": uuid.uuid4(),
                            "repository_id": repo_id,
                            "file_path": fpath,
                            "chunk_index": c_idx,
                            "chunk_text": clean_text,
                            "chunk_metadata": clean_metadata,
                            "embedding": embeddings[i],
                        })

                    stmt = pg_insert(RepoChunk).values(chunk_values)
                    stmt = stmt.on_conflict_do_nothing(constraint="uq_repo_chunk_file_pos")
                    await session.execute(stmt)
                    await session.commit()

        # Update repository commit SHA
        async with async_session() as session:
            stmt = select(Repository).where(Repository.id == repo_id)
            repo = (await session.execute(stmt)).scalar_one()
            repo.commit_sha = latest_sha
            repo.last_commit_check_at = datetime.utcnow()
            await session.commit()

        print(f"[Incremental Sync] Sync complete for {repo_url}!")
        return True

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
                    "file_path": chunk.file_path,
                    "chunk_index": chunk.chunk_index,
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
                        "file_path": chunk.file_path,
                        "chunk_index": chunk.chunk_index,
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
                    "file_path": chunk.file_path,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.chunk_text[:max_text_chars_per_chunk],
                    "metadata": chunk.chunk_metadata or {},
                }
                for chunk in chunks
            ]
