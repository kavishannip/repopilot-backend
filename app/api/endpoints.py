from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re
import logging
from typing import Optional, Dict, Any
from app.services.agent_service import GitHubAgentService
from app.models.brief import RepoBrief

router = APIRouter()
agent_service = GitHubAgentService()
logger = logging.getLogger(__name__)

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
    """
    Submit a repo for analysis. This endpoint runs synchronously and returns the complete markdown brief.
    """
    repo_url = req.repo_url.rstrip("/")

    if not is_valid_github_url(repo_url):
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    try:
        result = await agent_service.analyze_repo(repo_url)
        brief = RepoBrief(**result)
        return AnalyzeResponse(
            status="completed",
            message="Analysis completed successfully.",
            data=brief
        )
    except Exception as e:
        logger.exception("Failed to analyze repo")
        raise HTTPException(status_code=500, detail=str(e))

