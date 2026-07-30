from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
import asyncio
import json
import logging
from typing import Optional, List
from langchain_core.messages import SystemMessage, HumanMessage

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
    repo_id: str

class QuickSummaryRequest(BaseModel):
    repo_url: str

class QuickSummaryResponse(BaseModel):
    repo_id: str
    repo_url: str
    summary: str
    metadata: dict
    cached: bool = False
    cached_brief: Optional[RepoBrief] = None
    commit_sha: Optional[str] = None


class CheckUpdateRequest(BaseModel):
    repo_url: str


class CheckUpdateResponse(BaseModel):
    is_stale: bool
    cached_sha: Optional[str] = None
    latest_sha: Optional[str] = None
    changed_files_count: int = 0
    diff_summary: Optional[dict] = None


class SyncRequest(BaseModel):
    repo_id: str


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
    phase: str            # queued|embedding|structural_map|search|group_summaries|synthesis|done|failed
    result: Optional[RepoBrief] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ArchitectureDiagramRequest(BaseModel):
    repo_id: str
    force_regenerate: bool = False


class ArchitectureDiagramResponse(BaseModel):
    mermaid_code: str
    diagram_type: str
    cached: bool = False


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


@router.post("/quicksummary", response_model=QuickSummaryResponse)
async def get_quick_summary(req: QuickSummaryRequest):
    """
    Generate a rapid summary of the repository using only metadata and key files.
    Creates the repository in the database if it doesn't exist and returns the repo_id.
    """
    repo_url = req.repo_url.rstrip("/")
    if not is_valid_github_url(repo_url):
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    # Get metadata
    try:
        meta = await summarizer.github.get_repo_metadata(repo_url)
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

    # Check if repo exists and has a quick summary
    async with async_session() as session:
        stmt = select(Repository).where(Repository.repo_url == repo_url)
        result = await session.execute(stmt)
        repo_obj = result.scalar_one_or_none()

        if repo_obj and repo_obj.quick_summary:
            # Cache hit
            cached_brief_obj = None
            if repo_obj.brief:
                try:
                    cached_brief_obj = RepoBrief(**repo_obj.brief)
                except Exception:
                    cached_brief_obj = None

            return QuickSummaryResponse(
                repo_id=str(repo_obj.id),
                repo_url=repo_url,
                summary=repo_obj.quick_summary,
                metadata=meta,
                cached=True,
                cached_brief=cached_brief_obj,
                commit_sha=repo_obj.commit_sha
            )

    # Generate summary
    try:
        summary_text = await summarizer.generate_quick_summary(repo_url)
    except Exception as e:
        logger.exception("Failed to generate quick summary")
        raise HTTPException(status_code=500, detail=f"Failed to generate quick summary: {e}")

    # Save to DB
    async with async_session() as session:
        stmt = select(Repository).where(Repository.repo_url == repo_url)
        result = await session.execute(stmt)
        repo_obj = result.scalar_one_or_none()

        if not repo_obj:
            repo_obj = Repository(
                repo_url=repo_url,
                owner=meta.get("owner", ""),
                name=meta.get("name", ""),
                default_branch=meta.get("default_branch", "main"),
                quick_summary=summary_text,
                language=meta.get("language", ""),
                stars=meta.get("stars", 0),
                description=meta.get("description", "")
            )
            session.add(repo_obj)
        else:
            repo_obj.quick_summary = summary_text

        await session.commit()
        await session.refresh(repo_obj)
        repo_id_str = str(repo_obj.id)
        current_sha = repo_obj.commit_sha

    return QuickSummaryResponse(
        repo_id=repo_id_str,
        repo_url=repo_url,
        summary=summary_text,
        metadata=meta,
        cached=False,
        commit_sha=current_sha
    )


async def _run_sync_pipeline(job_id: str, repo_url: str) -> None:
    """
    Background task: runs incremental sync pipeline + re-synthesizes brief.
    """
    try:
        brief = await summarizer.sync_and_regenerate_brief(repo_url, job_id=job_id)
        await job_manager.complete_job(job_id, brief.model_dump())
    except Exception as e:
        logger.exception(f"[Job {job_id[:8]}] sync pipeline error")
        await job_manager.fail_job(job_id, str(e))


@router.post("/check-update", response_model=CheckUpdateResponse)
async def check_repo_update(req: CheckUpdateRequest):
    """
    Lightweight check if a repository has newer commits on GitHub than what's cached.
    """
    repo_url = req.repo_url.rstrip("/")
    if not is_valid_github_url(repo_url):
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    try:
        res = await summarizer.pipeline.check_staleness(repo_url)
        return CheckUpdateResponse(
            is_stale=res["is_stale"],
            cached_sha=res["cached_sha"],
            latest_sha=res["latest_sha"],
            changed_files_count=res["changed_files_count"],
            diff_summary=res["diff_summary"],
        )
    except Exception as e:
        logger.exception(f"Failed to check updates for {repo_url}")
        raise HTTPException(status_code=500, detail=f"Failed to check updates: {e}")


@router.post("/sync", response_model=JobSubmitResponse)
async def sync_repo(req: SyncRequest):
    """
    Trigger incremental sync for a repository that has updated on GitHub.
    """
    repo_url = None
    async with async_session() as session:
        stmt = select(Repository).where(Repository.id == req.repo_id)
        try:
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
            if not repo_obj:
                raise HTTPException(status_code=404, detail="Repository ID not found.")
            repo_url = repo_obj.repo_url
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Repository ID format.")

    job_id = await job_manager.create_job(repo_url)
    asyncio.create_task(_run_sync_pipeline(job_id, repo_url))

    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message="Incremental synchronization started. Poll status for updates.",
    )


@router.post("/analyze", response_model=JobSubmitResponse)
async def analyze_repo(req: AnalyzeRequest):
    """
    Submit a repo for analysis. Returns a job_id immediately.
    Poll GET /status/{job_id} for progress and result.
    Requires repo_id, which can be obtained from /quicksummary.
    """
    # Fetch repo URL using repo_id
    repo_url = None
    commit_sha = None
    async with async_session() as session:
        stmt = select(Repository).where(Repository.id == req.repo_id)
        try:
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
            if not repo_obj:
                raise HTTPException(status_code=404, detail="Repository ID not found. Call /quicksummary first.")
            repo_url = repo_obj.repo_url
        except Exception: # Handling UUID parse errors or bad format
            raise HTTPException(status_code=400, detail="Invalid Repository ID format.")

    # 1. Validate repo + get commit SHA (fast, synchronous-ish)
    try:
        meta = await summarizer.github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]
        commit_sha = await summarizer.pipeline._get_branch_commit(repo_url, branch)
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
                # We can't actually delete it here because the user just fetched it via quicksummary!
                # Wait, we can just reset brief/chunks, but let's just clear the brief.
                repo_obj.brief = None
                await session.commit()
            elif repo_obj.commit_sha == commit_sha and repo_obj.brief:
                # Cache hit
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


@router.post("/architecture-diagram", response_model=ArchitectureDiagramResponse)
async def generate_architecture_diagram(req: ArchitectureDiagramRequest):
    """
    Generate a high-level Mermaid architecture diagram for a repository.
    Sends analysis brief data to LLM, returns valid Mermaid syntax.
    Result is cached in Repository.brief JSONB under 'architecture_diagram'.
    """
    async with async_session() as session:
        stmt = select(Repository).where(Repository.id == req.repo_id)
        try:
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Repository ID format.")

        if not repo_obj:
            raise HTTPException(status_code=404, detail="Repository not found. Run deep analysis first.")
        if not repo_obj.brief:
            raise HTTPException(status_code=404, detail="No analysis data found. Run deep analysis first.")

        brief_data: dict = repo_obj.brief

        # Return cached diagram if available
        cached_diagram = brief_data.get("architecture_diagram")
        if cached_diagram and isinstance(cached_diagram, dict) and cached_diagram.get("mermaid_code") and not req.force_regenerate:
            return ArchitectureDiagramResponse(
                mermaid_code=cached_diagram["mermaid_code"],
                diagram_type=cached_diagram.get("diagram_type", "graph"),
                cached=True,
            )

    # Build context from brief fields
    architecture_summary = brief_data.get("architecture_summary", "")
    data_flow = brief_data.get("data_flow", "")
    directory_structure = brief_data.get("directory_structure", "")
    summary = brief_data.get("summary", "")

    # Format components
    components = brief_data.get("components", [])
    components_text = ""
    if components:
        parts = []
        for c in components:
            name = c.get("name", "")
            desc = c.get("description", "")
            depends = ", ".join(c.get("depends_on", []))
            techs = ", ".join(c.get("technologies", []))
            parts.append(f"- {name}: {desc} | depends_on: [{depends}] | technologies: [{techs}]")
        components_text = "\n".join(parts)

    # Format tech stack
    tech_stack = brief_data.get("tech_stack", [])
    tech_text = ""
    if tech_stack:
        parts = [f"- {t.get('name', '')} ({t.get('category', '')}): {t.get('purpose', '')}" for t in tech_stack]
        tech_text = "\n".join(parts)

    # Format dependency graph
    dep_graph = brief_data.get("dependency_graph", {})
    dep_text = ""
    if dep_graph:
        parts = [f"- {src} -> [{', '.join(targets)}]" for src, targets in dep_graph.items()]
        dep_text = "\n".join(parts)

    detected_patterns = brief_data.get("detected_patterns", [])
    patterns_text = ", ".join(detected_patterns) if detected_patterns else "Not detected"

    system_prompt = """You are an expert software architect specializing in system visualization.
Your task: generate a high-level Mermaid.js architecture diagram of a software repository.

RULES:
1. Use the Mermaid graph syntax. You may choose `graph TD` (top-down) or `graph LR` (left-right) — pick whichever best fits the architecture shape.
2. Show ONLY core architecture and major services. Keep it ultra-simple and highly readable. Hide minor components.
3. Maximum 12 nodes. Aim for 5-10 nodes for maximum simplicity.
4. Group related nodes using `subgraph` blocks (e.g., subgraph Frontend, subgraph Backend, subgraph External).
5. Add context to node labels using HTML line breaks (`<br/>`). Example: `DB["Database<br/>(Stores user profiles)"]`. This adds detail without cluttering.
6. Use accurate edge labels derived ONLY from the provided context (e.g., if it's MongoDB, don't label the edge SQL). If the protocol is unknown, use generic terms like "calls" or "data".
7. CRITICAL: Do NOT hallucinate technologies. Base all nodes, edge protocols, and database types EXACTLY on the 'TECH STACK' and 'DATA AND REQUEST FLOW' sections below.
8. Apply node styling using `:::` class notation or inline `style` for node types:
   - Frontend nodes: style with fill:#312e81,stroke:#6366f1 (indigo)
   - Backend/API nodes: style with fill:#1e3a5f,stroke:#3b82f6 (blue)  
   - Database nodes: style with fill:#14532d,stroke:#22c55e (green)
   - External service nodes: style with fill:#7c2d12,stroke:#f97316 (orange)
   - Auth/Security nodes: style with fill:#4a1d96,stroke:#a855f7 (purple)
9. RETURN ONLY the Mermaid code — no markdown fences, no explanation, no commentary.
10. The output must be syntactically valid Mermaid that can be rendered directly.
11. Do not use parentheses inside node labels — use brackets [] or quotes only.
12. CRITICAL: For directed edges with labels, use EXACTLY this syntax: `Node1 -->|Label| Node2`. DO NOT use `|>` in arrows (e.g. `-->|Label|>` is INVALID)."""

    user_prompt = f"""Generate a Mermaid architecture diagram for this repository.

=== PROJECT OVERVIEW ===
{summary[:2000]}

=== SYSTEM ARCHITECTURE ===
{architecture_summary[:3000]}

=== DATA AND REQUEST FLOW ===
{data_flow[:2000]}

=== COMPONENTS ===
{components_text[:2000]}

=== COMPONENT DEPENDENCIES ===
{dep_text[:1000]}

=== TECH STACK ===
{tech_text[:1500]}

=== DETECTED PATTERNS ===
{patterns_text}

=== DIRECTORY STRUCTURE (top level) ===
{directory_structure[:1500]}

Now produce the Mermaid diagram code only:"""

    try:
        from app.services.llm import get_fast_llm
        llm = get_fast_llm()
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        mermaid_raw = response.content.strip()
    except Exception as e:
        logger.exception("Failed to generate architecture diagram via LLM")
        raise HTTPException(status_code=500, detail=f"LLM diagram generation failed: {e}")

    # Strip any accidental markdown fences
    mermaid_code = re.sub(r'^```[a-zA-Z]*\n?', '', mermaid_raw, flags=re.MULTILINE)
    mermaid_code = re.sub(r'```$', '', mermaid_code, flags=re.MULTILINE).strip()

    # Sanitize invalid arrow syntax (e.g., -->|HTTP|> becomes -->|HTTP|)
    mermaid_code = re.sub(r'(-->\|[^|]+\|)>', r'\1 ', mermaid_code)
    mermaid_code = re.sub(r'(->\|[^|]+\|)>', r'\1 ', mermaid_code)

    # Detect diagram type from first line
    first_line = mermaid_code.split("\n")[0].strip().lower()
    if first_line.startswith("graph"):
        diagram_type = "graph"
    elif first_line.startswith("flowchart"):
        diagram_type = "flowchart"
    elif first_line.startswith("sequencediagram"):
        diagram_type = "sequence"
    elif first_line.startswith("c4"):
        diagram_type = "c4"
    else:
        diagram_type = "graph"

    # Cache result in Repository.brief JSONB
    async with async_session() as session:
        stmt = select(Repository).where(Repository.id == req.repo_id)
        result = await session.execute(stmt)
        repo_obj = result.scalar_one_or_none()
        if repo_obj and repo_obj.brief:
            updated_brief = dict(repo_obj.brief)
            updated_brief["architecture_diagram"] = {
                "mermaid_code": mermaid_code,
                "diagram_type": diagram_type,
            }
            repo_obj.brief = updated_brief
            await session.commit()

    return ArchitectureDiagramResponse(
        mermaid_code=mermaid_code,
        diagram_type=diagram_type,
        cached=False,
    )

