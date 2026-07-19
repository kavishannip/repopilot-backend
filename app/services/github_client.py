import httpx
import fnmatch
from urllib.parse import urlparse
from app.config import settings
from typing import List, Dict, Any, Optional

MAX_REPO_SIZE_KB = 100 * 1024  # 100 MB
MAX_FILE_COUNT = 150  # Reduced for quality: understand each file better

NOISE_DIRS = {
    "node_modules", "vendor", "dist", "build", ".git", ".idea", ".vscode",
    "__pycache__", "out", "target", "bin", "obj", ".next", ".nuxt", ".svelte-kit",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache"
}

NOISE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".mp4", ".mp3",
    ".zip", ".tar", ".gz", ".bin", ".exe", ".dll", ".so", ".dylib",
    ".ttf", ".woff", ".woff2", ".eot", ".svg", ".webp", ".avif",
    ".parquet", ".pkl", ".h5", ".model", ".onnx", ".pb"
}

LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "poetry.lock", "Cargo.lock", "Gemfile.lock", "mix.lock",
    "composer.lock", "Pipfile.lock", "uv.lock"
}

# Files that MUST be read in full — highest priority
CRITICAL_FILES = {
    "README.md", "README.rst", "README.txt", "README",
    "package.json", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "go.mod", "go.sum", "pubspec.yaml", "mix.exs",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", "justfile",
    ".env.example", ".env.sample",
}

# Config files — high value for architecture understanding
CONFIG_FILES = {
    "tsconfig.json", "jsconfig.json",
    "next.config.js", "next.config.mjs", "next.config.ts",
    "vite.config.js", "vite.config.ts", "vite.config.mjs",
    "webpack.config.js", "webpack.config.ts",
    "tailwind.config.js", "tailwind.config.ts",
    "angular.json", "nuxt.config.ts", "nuxt.config.js",
    "svelte.config.js", "remix.config.js",
    "nest-cli.json", "angular.json",
    ".eslintrc.js", ".eslintrc.json", ".eslintrc.yaml", "eslint.config.js",
    ".prettierrc", ".prettierrc.json",
    "jest.config.js", "jest.config.ts", "vitest.config.ts",
    "alembic.ini", "alembic.cfg",
    "pytest.ini", "setup.cfg", "pyproject.toml",
    ".github/workflows",  # CI/CD
    "vercel.json", "netlify.toml", "fly.toml", "render.yaml",
    "kubernetes.yaml", "k8s.yaml", "helm",
}

# Patterns for entry point detection
ENTRY_POINT_PATTERNS = [
    "main.*", "app.*", "index.*", "server.*", "manage.py",
    "wsgi.py", "asgi.py", "cli.*", "run.*", "start.*",
]

# Source file extensions for language detection
LANG_EXTENSIONS: Dict[str, str] = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "JavaScript", ".tsx": "TypeScript", ".go": "Go",
    ".java": "Java", ".kt": "Kotlin", ".scala": "Scala",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#",
    ".cpp": "C++", ".c": "C", ".h": "C/C++",
    ".rs": "Rust", ".swift": "Swift", ".dart": "Dart",
    ".ex": "Elixir", ".exs": "Elixir", ".clj": "Clojure",
    ".hs": "Haskell", ".ml": "OCaml", ".r": "R",
    ".vue": "Vue", ".svelte": "Svelte",
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
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
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
                "language": data.get("language", ""),
                "topics": data.get("topics", []),
                "homepage": data.get("homepage", ""),
                "open_issues": data.get("open_issues_count", 0),
            }

    async def list_files(self, repo_url: str, branch: str = None) -> List[Dict[str, Any]]:
        owner, repo = self._parse_repo_url(repo_url)
        if not branch:
            meta = await self.get_repo_metadata(repo_url)
            branch = meta["default_branch"]

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
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

    async def get_file_tree_structure(self, repo_url: str, branch: str = None) -> Dict[str, Any]:
        """
        Returns a rich structural map of the repository:
        - full nested directory tree
        - file classification buckets
        - per-directory language breakdown
        - all_files list for further processing
        """
        owner, repo = self._parse_repo_url(repo_url)
        if not branch:
            meta = await self.get_repo_metadata(repo_url)
            branch = meta["default_branch"]

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
                headers=self.headers
            )
            self._handle_rate_limit(resp)
            data = resp.json()

        tree_items = data.get("tree", [])

        # Build nested directory tree
        nested_tree: Dict[str, Any] = {}
        all_paths = []
        for item in tree_items:
            path = item["path"]
            item_type = item["type"]  # "blob" or "tree"
            parts = path.split("/")
            current = nested_tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            if item_type == "blob":
                current[parts[-1]] = {"_type": "file", "_size": item.get("size", 0)}
                all_paths.append(path)
            else:
                current.setdefault(parts[-1], {})

        # Classify files
        classification = self.classify_files(all_paths)

        # Per-directory language breakdown
        dir_languages = self.get_directory_languages(all_paths)

        return {
            "nested_tree": nested_tree,
            "classification": classification,
            "directory_languages": dir_languages,
            "total_files": len(all_paths),
        }

    def classify_files(self, paths: List[str]) -> Dict[str, List[str]]:
        """Classify file paths into buckets for smart reading strategy."""
        classification = {
            "critical": [],      # README, manifests — read in full, always
            "entry_points": [],  # main.*, app.*, routes — read in full
            "config": [],        # tsconfig, next.config, etc. — read in full
            "source": [],        # actual code files — chunk + embed
            "docs": [],          # other markdown/txt — skim or embed
            "ci_cd": [],         # .github/workflows, Dockerfile variants
        }

        for path in paths:
            if self._is_noise(path):
                continue
            filename = path.split("/")[-1]
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            # CI/CD
            if ".github/workflows" in path or filename.endswith((".yml", ".yaml")) and (
                "github" in path or "circleci" in path or "travis" in path
            ):
                classification["ci_cd"].append(path)
                continue

            # Critical files (exact filename match)
            if filename in CRITICAL_FILES:
                classification["critical"].append(path)
                continue

            # Config files
            if filename in CONFIG_FILES or any(
                fnmatch.fnmatch(filename, pat) for pat in ["*.config.*", ".eslint*", ".prettier*"]
            ):
                classification["config"].append(path)
                continue

            # Entry points
            if any(fnmatch.fnmatch(filename, pat) for pat in ENTRY_POINT_PATTERNS):
                # Shallower entry points are real entries; deep ones might just be named similarly
                depth = len(path.split("/"))
                if depth <= 4:
                    classification["entry_points"].append(path)
                    continue

            # Route files — common patterns across frameworks
            path_lower = path.lower()
            if any(seg in path_lower for seg in [
                "/routes/", "/route/", "/router/", "/controllers/",
                "/api/", "/pages/api/", "/handlers/", "/views/",
                "/endpoints/", "/resolvers/", "/schema.", "/models/"
            ]):
                classification["entry_points"].append(path)
                continue

            # Docs
            if ext in {".md", ".rst", ".txt", ".adoc"}:
                classification["docs"].append(path)
                continue

            # Source code
            if ext in LANG_EXTENSIONS:
                classification["source"].append(path)

        return classification

    def get_directory_languages(self, paths: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Returns per top-level-directory language file counts.
        e.g. {"src": {"TypeScript": 12, "JavaScript": 3}, "backend": {"Python": 8}}
        """
        dir_langs: Dict[str, Dict[str, int]] = {}
        for path in paths:
            parts = path.split("/")
            top_dir = parts[0] if len(parts) > 1 else "_root"
            filename = parts[-1]
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            lang = LANG_EXTENSIONS.get(ext)
            if lang:
                dir_langs.setdefault(top_dir, {})
                dir_langs[top_dir][lang] = dir_langs[top_dir].get(lang, 0) + 1
        return dir_langs

    def _is_noise(self, path: str) -> bool:
        parts = path.split("/")

        # Check directories
        for part in parts[:-1]:
            if part in NOISE_DIRS or part.startswith("."):
                # Allow .github directory for CI workflows
                if part == ".github":
                    continue
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
        manifest_patterns = [
            "README*", "package.json", "requirements.txt", "pyproject.toml",
            "Cargo.toml", "pom.xml", "go.mod", "Gemfile",
            "Dockerfile", "docker-compose*", "Makefile",
        ]
        entry_patterns = ["main.*", "app.*", "index.*", "server.*", "manage.py", "wsgi.py", "asgi.py"]
        config_patterns = ["tsconfig*", "next.config*", "vite.config*", "tailwind.config*", "*.config.*"]

        def get_score(file_item):
            path = file_item["path"]
            filename = path.split("/")[-1]
            depth = len(path.split("/"))
            score = 100 + depth  # Shallower = better base score

            for pat in manifest_patterns:
                if fnmatch.fnmatch(filename, pat):
                    return score - 100  # Highest priority

            for pat in entry_patterns:
                if fnmatch.fnmatch(filename, pat):
                    return score - 50

            for pat in config_patterns:
                if fnmatch.fnmatch(filename, pat):
                    return score - 40

            # Route/model/controller files get boosted
            path_lower = path.lower()
            if any(seg in path_lower for seg in [
                "/routes/", "/api/", "/controllers/", "/models/", "/pages/api/"
            ]):
                return score - 20

            return score

        return sorted(files, key=get_score)

    async def get_file_content(self, repo_url: str, path: str, branch: str = None) -> Optional[str]:
        owner, repo = self._parse_repo_url(repo_url)
        if not branch:
            meta = await self.get_repo_metadata(repo_url)
            branch = meta["default_branch"]

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
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

    async def get_files_content_batch(
        self,
        repo_url: str,
        paths: List[str],
        branch: str = None,
        max_chars_per_file: int = 8000,
    ) -> Dict[str, str]:
        """
        Fetch multiple files concurrently. Returns {path: content} dict.
        Truncates each file at max_chars_per_file to stay within LLM context budget.
        """
        import asyncio
        owner, repo = self._parse_repo_url(repo_url)
        if not branch:
            meta = await self.get_repo_metadata(repo_url)
            branch = meta["default_branch"]

        async def _fetch(path: str) -> tuple[str, Optional[str]]:
            try:
                content = await self.get_file_content(repo_url, path, branch)
                if content and len(content) > max_chars_per_file:
                    content = content[:max_chars_per_file] + "\n... [truncated]"
                return path, content
            except Exception as e:
                print(f"Failed to fetch {path}: {e}")
                return path, None

        results = await asyncio.gather(*(_fetch(p) for p in paths))
        return {path: content for path, content in results if content}
