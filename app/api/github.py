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
            
        # Fetch the installation details to get the account name
        details = await github_auth.get_installation_details(req.installation_id)
        account_name = details.get("account", {}).get("login")
        target_type = details.get("account", {}).get("type", "User")
        repository_selection = details.get("repository_selection", "all")

        if not account_name:
            raise HTTPException(status_code=400, detail="Could not determine account name for installation")
            
        async with async_session() as session:
            # Check if it already exists
            stmt = select(GithubInstallation).where(GithubInstallation.installation_id == req.installation_id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing record in case account name changed or repository selection changed
                existing.account_name = account_name
                existing.target_type = target_type
                existing.repository_selection = repository_selection
                await session.commit()
                return {"message": "Installation already registered, updated details"}
                
            new_install = GithubInstallation(
                installation_id=req.installation_id,
                account_name=account_name,
                target_type=target_type,
                repository_selection=repository_selection
            )
            session.add(new_install)
            await session.commit()
            
        return {"message": "Successfully registered GitHub App installation"}
    except Exception as e:
        logger.exception("Failed to register GitHub App")
        raise HTTPException(status_code=500, detail=str(e))
