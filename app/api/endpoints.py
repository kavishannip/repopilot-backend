from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
import asyncio
import logging
from typing import Optional, List

from app.database import async_session
from sqlalchemy.future import select
from app.models.schema import Repository
from app.services.summarizer import RepoSummarizer
from app.models.brief import RepoBrief
from app.services.github_client import RepoSizeExceededError, RepoTooComplexError
from app.services.job_manager import job_manager
from app.config import settings

router = APIRouter()
summarizer = RepoSummarizer()
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    repo_url: str


class AnalyzeResponse(BaseModel):
    status: str
    message: str
    data: Optional[RepoBrief] = None


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str           # pending|running|completed|failed
    phase: str            # queued|structural_map|agent_exploration|graph_query|synthesis|done|failed
    result: Optional[RepoBrief] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class GraphNodeResponse(BaseModel):
    id: str
    path: str
    type: str
    deep_analysis: bool

class GraphEdgeResponse(BaseModel):
    id: str
    source_node_id: str
    target_node_id: str
    type: str

class GraphDataResponse(BaseModel):
    nodes: List[GraphNodeResponse]
    edges: List[GraphEdgeResponse]


def is_valid_github_url(url: str) -> bool:
    pattern = r"^https://github\.com/[\w-]+/[\w.-]+/?$"
    return bool(re.match(pattern, url))


async def _run_pipeline(job_id: str, repo_url: str) -> None:
    """
    Background task: runs full analysis pipeline, writes result/error to DB job row.
    """
    try:
        brief = await summarizer.generate_brief(repo_url, job_id=job_id)
        await job_manager.complete_job(job_id, brief.model_dump())
    except RepoSizeExceededError as e:
        await job_manager.fail_job(job_id, f"Repo too large: {e}")
    except RepoTooComplexError as e:
        await job_manager.fail_job(job_id, f"Repo too complex for accurate analysis: {e}")
    except Exception as e:
        logger.exception(f"[Job {job_id[:8]}] pipeline error")
        await job_manager.fail_job(job_id, str(e))


@router.post("/analyze", response_model=JobSubmitResponse)
async def analyze_repo(req: AnalyzeRequest):
    """
    Submit a repo for analysis. Returns a job_id immediately.
    Poll GET /status/{job_id} for progress and result.

    If the repo brief is already cached (same commit SHA, not expired), returns
    the brief directly in the 'result' field of the status endpoint without
    creating a job.
    """
    repo_url = req.repo_url.rstrip("/")

    if not is_valid_github_url(repo_url):
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    # 1. Validate repo + get commit SHA (fast, synchronous-ish)
    try:
        github = await summarizer.pipeline.get_github_client(repo_url)
        meta = await github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]
        commit_sha = await summarizer.pipeline._get_branch_commit(github, repo_url, branch)
    except RepoSizeExceededError as e:
        raise HTTPException(status_code=400, detail=f"Repo exceeds size limit. {e}")
    except RepoTooComplexError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Repository too complex for accurate analysis. {e} "
                "Please try a focused sub-directory or a smaller repository."
            )
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch repo metadata: {e}")

    # 2. Check cache — if valid, return cached job immediately
    async with async_session() as session:
        stmt = select(Repository).where(Repository.repo_url == repo_url)
        result = await session.execute(stmt)
        repo_obj = result.scalar_one_or_none()

        if repo_obj:
            from datetime import datetime
            now = datetime.utcnow()
            expire_days = settings.repo_cache_expire_days

            if repo_obj.updated_at and (now - repo_obj.updated_at).days >= expire_days:
                # Expired — delete and fall through to fresh analysis
                await session.delete(repo_obj)
                await session.commit()
            elif repo_obj.commit_sha == commit_sha and repo_obj.brief:
                # Cache hit — create a synthetic completed job so the caller
                # can use the same polling endpoint consistently
                cached_job_id = await job_manager.create_job(repo_url)
                await job_manager.complete_job(cached_job_id, repo_obj.brief)
                return JobSubmitResponse(
                    job_id=cached_job_id,
                    status="completed",
                    message="Returned from cache — poll /status/{job_id} for result",
                )

    # 3. Create DB job + fire background task
    job_id = await job_manager.create_job(repo_url)
    asyncio.create_task(_run_pipeline(job_id, repo_url))

    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message="Analysis started. Poll GET /api/analyzer/status/{job_id} for progress.",
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Poll analysis job status. Returns phase progress and final result when done.
    """
    job = await job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    result_brief = None
    if job["status"] == "completed" and job.get("result"):
        try:
            result_brief = RepoBrief(**job["result"])
        except Exception as e:
            logger.warning(f"Failed to parse cached brief for job {job_id}: {e}")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        phase=job["phase"],
        result=result_brief,
        error=job.get("error"),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


@router.get("/graph", response_model=GraphDataResponse)
async def get_graph_data(repo_url: str):
    """
    Fetch raw graph data (nodes and edges) for frontend visualization.
    """
    repo_url = repo_url.rstrip("/")
    if not is_valid_github_url(repo_url):
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    async with async_session() as session:
        # Get repository
        stmt = select(Repository).where(Repository.repo_url == repo_url)
        repo_obj = (await session.execute(stmt)).scalar_one_or_none()
        if not repo_obj:
            raise HTTPException(status_code=404, detail="Repository not found or hasn't been analyzed yet.")
            
        from app.models.schema import GraphNode, GraphEdge
        
        # Get nodes
        nodes_stmt = select(GraphNode).where(GraphNode.repository_id == repo_obj.id)
        nodes = (await session.execute(nodes_stmt)).scalars().all()
        
        # Get edges
        edges_stmt = select(GraphEdge).where(GraphEdge.repository_id == repo_obj.id)
        edges = (await session.execute(edges_stmt)).scalars().all()
        
    return GraphDataResponse(
        nodes=[
            GraphNodeResponse(
                id=str(n.id), 
                path=n.path, 
                type=n.type, 
                deep_analysis=n.deep_analysis
            ) for n in nodes
        ],
        edges=[
            GraphEdgeResponse(
                id=str(e.id), 
                source_node_id=str(e.source_node_id), 
                target_node_id=str(e.target_node_id), 
                type=e.type
            ) for e in edges
        ]
    )

