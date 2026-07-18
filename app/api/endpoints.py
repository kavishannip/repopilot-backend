from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
from typing import Optional

from app.database import async_session
from sqlalchemy.future import select
from app.models.schema import Repository
from app.services.summarizer import RepoSummarizer
from app.models.brief import RepoBrief
from app.services.github_client import RepoSizeExceededError
from app.config import settings

router = APIRouter()
summarizer = RepoSummarizer()


class AnalyzeRequest(BaseModel):
    repo_url: str


class AnalyzeResponse(BaseModel):
    status: str
    message: str
    data: Optional[RepoBrief] = None


def is_valid_github_url(url: str) -> bool:
    pattern = r"^https://github\.com/[\w-]+/[\w.-]+/?$"
    return bool(re.match(pattern, url))


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(req: AnalyzeRequest):
    repo_url = req.repo_url.rstrip("/")

    if not is_valid_github_url(repo_url):
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    # 1. Fetch metadata first to get commit sha
    try:
        meta = await summarizer.github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]
        commit_sha = await summarizer.pipeline._get_branch_commit(repo_url, branch)
    except RepoSizeExceededError as e:
        raise HTTPException(
            status_code=400,
            detail=f"This repo exceeds the current size limit. {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch repo metadata: {str(e)}")

    # 2. Check cache for brief (must match commit_sha and not be expired)
    async with async_session() as session:
        stmt = select(Repository).where(Repository.repo_url == repo_url)
        result = await session.execute(stmt)
        repo_obj = result.scalar_one_or_none()

        if repo_obj:
            from datetime import datetime
            
            # Check if cache is expired
            now = datetime.utcnow()
            expire_days = settings.repo_cache_expire_days
            
            if repo_obj.updated_at and (now - repo_obj.updated_at).days >= expire_days:
                # Cache expired -> delete it from DB and proceed to analyze fresh
                await session.delete(repo_obj)
                await session.commit()
            elif repo_obj.commit_sha == commit_sha and repo_obj.brief:
                # Valid cache hit
                brief = RepoBrief(**repo_obj.brief)
                return AnalyzeResponse(
                    status="success",
                    message="Returned from cache",
                    data=brief,
                )

    # 3. Full multi-pass analysis
    try:
        import asyncio
        timeout = settings.analysis_timeout_seconds
        brief = await asyncio.wait_for(
            summarizer.generate_brief(repo_url),
            timeout=float(timeout),
        )

        # Cache the brief
        async with async_session() as session:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
            if repo_obj:
                repo_obj.brief = brief.model_dump()
                await session.commit()

        return AnalyzeResponse(
            status="success",
            message="Repository analyzed successfully",
            data=brief,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timed out after {settings.analysis_timeout_seconds}s. The repository may be too large or complex."
        )
    except RepoSizeExceededError as e:
        raise HTTPException(
            status_code=400,
            detail=f"This repo exceeds the current file limit — try a smaller repo. {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze repository: {str(e)}")
