from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from sqlalchemy.future import select
from app.database import async_session
from app.models.schema import GithubInstallation
from app.services.github_auth import github_auth

router = APIRouter()
logger = logging.getLogger(__name__)

class InstallCallbackRequest(BaseModel):
    installation_id: str
    setup_action: str = "install"

@router.get("/install")
async def install_github_app():
    """Redirects the user to the GitHub App installation page."""
    # Assuming app name is configured or known. For now, returning a general message.
    # The frontend would typically handle this redirect based on the App's public URL.
    return {"message": "Redirect user to GitHub App installation page", "url": "https://github.com/apps/YOUR_APP_NAME/installations/new"}

@router.post("/callback")
async def github_app_callback(req: InstallCallbackRequest):
    """
    Handle the callback after a user installs the GitHub App.
    Stores the installation ID in the database.
    """
    try:
        # Verify the installation ID by trying to get a token
        token = await github_auth.get_installation_token(req.installation_id)
        if not token:
            raise HTTPException(status_code=400, detail="Invalid installation ID")
            
        async with async_session() as session:
            # Check if it already exists
            stmt = select(GithubInstallation).where(GithubInstallation.installation_id == req.installation_id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                return {"message": "Installation already registered"}
                
            new_install = GithubInstallation(
                installation_id=req.installation_id,
                account_name="fetched_later", # Can fetch via GitHub API with token
                target_type="User",
                repository_selection="all"
            )
            session.add(new_install)
            await session.commit()
            
        return {"message": "Successfully registered GitHub App installation"}
    except Exception as e:
        logger.exception("Failed to register GitHub App")
        raise HTTPException(status_code=500, detail=str(e))
