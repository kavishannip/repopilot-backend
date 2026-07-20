import time
import jwt
import httpx
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class GitHubAppAuth:
    def __init__(self):
        self.app_id = settings.github_app_id
        self.private_key = settings.github_app_private_key
        # The private key might be stored with literal \n instead of actual newlines if from env var
        if self.private_key:
            self.private_key = self.private_key.replace("\\n", "\n")

    def generate_jwt(self) -> str:
        if not self.app_id or not self.private_key:
            raise ValueError("GitHub App ID or Private Key not configured.")
            
        payload = {
            # issued at time, 60 seconds in the past to allow for clock drift
            'iat': int(time.time()) - 60,
            # JWT expiration time (10 minute maximum)
            'exp': int(time.time()) + (10 * 60),
            # GitHub App's identifier
            'iss': self.app_id
        }
        
        encoded_jwt = jwt.encode(payload, self.private_key, algorithm='RS256')
        return encoded_jwt

    async def get_installation_token(self, installation_id: str) -> str:
        """Exchange JWT for a short-lived installation token."""
        encoded_jwt = self.generate_jwt()
        
        headers = {
            "Authorization": f"Bearer {encoded_jwt}",
            "Accept": "application/vnd.github.v3+json",
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://api.github.com/app/installations/{installation_id}/access_tokens",
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data["token"]

    async def get_installation_details(self, installation_id: str) -> dict:
        """Fetch installation details to get account login."""
        encoded_jwt = self.generate_jwt()
        
        headers = {
            "Authorization": f"Bearer {encoded_jwt}",
            "Accept": "application/vnd.github.v3+json",
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://api.github.com/app/installations/{installation_id}",
                headers=headers
            )
            resp.raise_for_status()
            return resp.json()

github_auth = GitHubAppAuth()
