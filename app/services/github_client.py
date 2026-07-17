import httpx
import fnmatch
from urllib.parse import urlparse
from app.config import settings
from typing import List, Dict, Any, Optional

MAX_REPO_SIZE_KB = 100 * 1024  # 100 MB
MAX_FILE_COUNT = 300

NOISE_DIRS = {
    "node_modules", "vendor", "dist", "build", ".git", ".idea", ".vscode",
    "__pycache__", "out", "target", "bin", "obj"
}

NOISE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".mp4", ".mp3",
    ".zip", ".tar", ".gz", ".bin", ".exe", ".dll", ".so", ".dylib",
    ".ttf", ".woff", ".woff2", ".eot", ".svg"
}

LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "poetry.lock", "Cargo.lock", "Gemfile.lock", "mix.lock"
}

class RepoSizeExceededError(Exception):
    pass

class RateLimitExceededError(Exception):
    pass

class GitHubClient:
    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "RepoPilot"
        }
        if settings.github_token:
            self.headers["Authorization"] = f"Bearer {settings.github_token}"

    def _parse_repo_url(self, repo_url: str):
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            return path_parts[0], path_parts[1]
        raise ValueError("Invalid GitHub URL. Must be in format 'https://github.com/owner/repo'")

    def _handle_rate_limit(self, response: httpx.Response):
        if response.status_code == 403 and "rate limit exceeded" in response.text.lower():
            raise RateLimitExceededError("GitHub API rate limit exceeded.")
        response.raise_for_status()

    async def get_repo_metadata(self, repo_url: str) -> Dict[str, Any]:
        owner, repo = self._parse_repo_url(repo_url)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=self.headers
            )
            self._handle_rate_limit(resp)
            data = resp.json()

            size_kb = data.get("size", 0)
            if size_kb > MAX_REPO_SIZE_KB:
                raise RepoSizeExceededError(
                    f"Repo size {size_kb}KB exceeds maximum allowed {MAX_REPO_SIZE_KB}KB."
                )

            return {
                "owner": owner,
                "name": repo,
                "description": data.get("description", ""),
                "stars": data.get("stargazers_count", 0),
                "default_branch": data.get("default_branch", "main"),
                "size_kb": size_kb,
                "language": data.get("language", "")
            }

    async def list_files(self, repo_url: str, branch: str = None) -> List[Dict[str, Any]]:
        owner, repo = self._parse_repo_url(repo_url)
        if not branch:
            meta = await self.get_repo_metadata(repo_url)
            branch = meta["default_branch"]

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
                headers=self.headers
            )
            self._handle_rate_limit(resp)
            data = resp.json()

            tree = data.get("tree", [])
            files = []

            for item in tree:
                if item["type"] == "blob":
                    path = item["path"]
                    if not self._is_noise(path):
                        files.append(item)

            prioritized_files = self._prioritize_files(files)
            return prioritized_files[:MAX_FILE_COUNT]

    def _is_noise(self, path: str) -> bool:
        parts = path.split("/")
        
        # Check directories
        for part in parts[:-1]:
            if part in NOISE_DIRS or part.startswith("."):
                return True

        filename = parts[-1]
        
        # Check extensions
        if any(filename.endswith(ext) for ext in NOISE_EXTENSIONS):
            return True
            
        # Check lockfiles and minified files
        if filename in LOCKFILES:
            return True
        if filename.endswith(".min.js") or filename.endswith(".min.css"):
            return True

        return False

    def _prioritize_files(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        manifest_patterns = ["README*", "package.json", "requirements.txt", "Cargo.toml", "pom.xml", "Dockerfile", "docker-compose*"]
        entry_patterns = ["main.*", "app.*", "index.*", "server.*"]

        def get_score(file_item):
            path = file_item["path"]
            filename = path.split("/")[-1]
            depth = len(path.split("/"))

            score = 100 + depth  # Shallow files get lower (better) score

            for pat in manifest_patterns:
                if fnmatch.fnmatch(filename, pat):
                    return score - 100  # Highest priority

            for pat in entry_patterns:
                if fnmatch.fnmatch(filename, pat):
                    return score - 50   # High priority

            return score

        return sorted(files, key=get_score)

    async def get_file_content(self, repo_url: str, path: str, branch: str = None) -> Optional[str]:
        owner, repo = self._parse_repo_url(repo_url)
        if not branch:
            meta = await self.get_repo_metadata(repo_url)
            branch = meta["default_branch"]

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                headers=self.headers
            )
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                return None
            self._handle_rate_limit(resp)
            return None
