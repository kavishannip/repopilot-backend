import asyncio
from sqlalchemy.future import select
from sqlalchemy import delete
from typing import List, Dict, Any

from app.database import async_session
from app.models.schema import Repository, RepoChunk
from app.services.github_client import GitHubClient, RepoSizeExceededError
from app.services.chunker import CodeChunker
from app.services.embedding import NVIDIAEmbeddingClient

class ProcessingPipeline:
    def __init__(self):
        self.github = GitHubClient()
        self.chunker = CodeChunker()
        self.embedder = NVIDIAEmbeddingClient()

    async def process_repo(self, repo_url: str) -> Dict[str, Any]:
        """
        Process a repository: fetch, chunk, embed, store.
        Returns stats about the process.
        """
        print(f"Fetching metadata for {repo_url}...")
        meta = await self.github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]
        
        # We need the commit SHA of the default branch to check cache
        # For simplicity, if we don't have it directly from meta, we can get it from branch ref
        commit_sha = await self._get_branch_commit(repo_url, branch)
        
        async with async_session() as session:
            # Check cache
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
            
            if repo_obj and repo_obj.commit_sha == commit_sha:
                print("Cache hit! Repository is up to date.")
                # Return counts
                stmt_count = select(RepoChunk).where(RepoChunk.repository_id == repo_obj.id)
                res = await session.execute(stmt_count)
                chunks = res.scalars().all()
                return {
                    "status": "cached",
                    "repo_url": repo_url,
                    "commit_sha": commit_sha,
                    "chunks_count": len(chunks)
                }

            # If it exists but different commit, update. If not exists, create.
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
                await session.flush() # to get ID
            else:
                repo_obj.commit_sha = commit_sha
                repo_obj.language = meta["language"]
                repo_obj.stars = meta["stars"]
                repo_obj.description = meta["description"]
                
                # Delete old chunks
                await session.execute(delete(RepoChunk).where(RepoChunk.repository_id == repo_obj.id))
                await session.flush()

            # Now fetch files
            print("Fetching file list...")
            files = await self.github.list_files(repo_url)
            
            all_chunks = []
            
            print(f"Processing {len(files)} files...")
            
            async def fetch_and_chunk(f):
                try:
                    content = await self.github.get_file_content(repo_url, f["path"])
                    if content:
                        return self.chunker.chunk_file(f["path"], content)
                except Exception as e:
                    print(f"Failed to process {f['path']}: {e}")
                return []
                
            batch_size = 30
            for i in range(0, len(files), batch_size):
                batch = files[i:i+batch_size]
                results = await asyncio.gather(*(fetch_and_chunk(f) for f in batch))
                for file_chunks in results:
                    all_chunks.extend(file_chunks)
                    
            print(f"Total chunks generated: {len(all_chunks)}")
            
            # Batch embedding
            if all_chunks:
                texts = [c["text"] for c in all_chunks]
                print(f"Embedding {len(texts)} chunks...")
                embeddings = await self.embedder.embed_chunks(texts, input_type="passage")
                
                # Save to DB
                print("Saving to database...")
                db_chunks = []
                for i, c in enumerate(all_chunks):
                    clean_text = c["text"].replace("\x00", "")
                    
                    # Clean metadata just in case (as it gets serialized to JSONB)
                    clean_metadata = {}
                    for k, v in c["metadata"].items():
                        if isinstance(v, str):
                            clean_metadata[k] = v.replace("\x00", "")
                        else:
                            clean_metadata[k] = v
                            
                    db_chunks.append(
                        RepoChunk(
                            repository_id=repo_obj.id,
                            chunk_text=clean_text,
                            chunk_metadata=clean_metadata,
                            embedding=embeddings[i]
                        )
                    )
                session.add_all(db_chunks)
                
            await session.commit()
            print("Done processing repository.")
            
            return {
                "status": "processed",
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "chunks_count": len(all_chunks)
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
        # Embed query
        query_embedding = await self.embedder.embed_chunks([query], input_type="query")
        if not query_embedding:
            return []
        
        q_vec = query_embedding[0]
        
        async with async_session() as session:
            # First find the repository id
            stmt_repo = select(Repository).where(Repository.repo_url == repo_url)
            repo_res = await session.execute(stmt_repo)
            repo_obj = repo_res.scalar_one_or_none()
            
            if not repo_obj:
                raise ValueError(f"Repository {repo_url} not found in database.")
                
            # Perform similarity search using pgvector's cosine distance `<=>`
            # We want to order by similarity (ascending cosine distance)
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
