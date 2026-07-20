import httpx
import fnmatch
import logging
from urllib.parse import urlparse
from app.config import settings
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

MAX_REPO_SIZE_KB = 100 * 1024  # 100 MB
MAX_FILE_COUNT = 150  # Hard accuracy budget — do not exceed

NOISE_DIRS = {
    "node_modules", "vendor", "dist", "build", ".git", ".hg", ".svn", ".idea", ".vscode",
    "__pycache__", "out", "target", "bin", "obj", ".next", ".nuxt", ".svelte-kit",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".pytype",
    # Python
    ".venv", "venv", "env", ".tox", ".eggs", "*.egg-info", ".ipynb_checkpoints",
    "site-packages", ".pdm-build", ".pdm-python", "__pypackages__",
    # JS/TS tooling & bundlers
    ".cache", ".parcel-cache", ".turbo", ".yarn", ".pnpm-store", "bower_components",
    ".docusaurus", "storybook-static", ".vercel", ".netlify", ".wrangler",
    ".astro", ".output", ".angular", ".expo", ".expo-shared", ".nx",
    ".yarn/cache", ".yarn/unplugged", ".yarn/install-state.gz", ".pnp.cjs",
    "web-build", ".rollup.cache", ".webpack",
    # Mobile
    "Pods", "DerivedData", "ios/Pods", ".gradle", ".dart_tool",
    "fastlane/report.xml", ".flutter-plugins", ".flutter-plugins-dependencies",
    "android/.gradle", "android/app/build", "ios/build", ".symlinks",
    "Carthage", ".swiftpm", "build.xcarchive",
    # JVM / .NET / other build systems
    ".settings", ".vs", "packages", "bld", "cmake-build-debug", "cmake-build-release",
    ".gradle-cache", "gradle-app.setting", ".kotlin", "out/production", "out/test",
    "bin/Debug", "bin/Release", "obj/Debug", "obj/Release", ".vs/config",
    # Infra / IaC
    ".terraform", ".terragrunt-cache", ".serverless", "cdk.out", ".sam", ".pulumi",
    ".vagrant", "ansible/.cache",
    # Test/coverage/logs
    "htmlcov", ".nyc_output", ".sass-cache", "logs", "tmp", "temp", ".hypothesis",
    "allure-results", "test-results", "playwright-report", ".playwright",
    # Rust/Go/Elixir
    "_build", "deps", ".cargo", "Cargo.lock.bak",
    # Data/ML
    "wandb", "mlruns", ".dvc/cache", ".dvc/tmp", "lightning_logs", "checkpoints",
    ".neptune", "outputs/multirun", ".cometml-runs",
    # Ruby / PHP
    ".bundle", "spec/reports", "public/assets", "public/packs",
    # Docs generators
    "_site", ".jekyll-cache", ".jekyll-metadata", "site", "public",
    # Game engines
    "Library", "Temp", "Logs", "obj", "Build", "Builds", ".godot", "ProjectSettings/Backups",
    # Misc
    ".direnv", ".devenv", "__snapshots__", ".husky/_", ".changeset",
}

NOISE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".mp4", ".mp3",
    ".zip", ".tar", ".gz", ".bin", ".exe", ".dll", ".so", ".dylib",
    ".ttf", ".woff", ".woff2", ".eot", ".svg", ".webp", ".avif",
    ".parquet", ".pkl", ".h5", ".model", ".onnx", ".pb",
    # More images/design
    ".bmp", ".tiff", ".heic", ".psd", ".ai", ".sketch", ".fig", ".otf",
    ".xcf", ".indd", ".eps", ".raw", ".cr2", ".nef", ".dng",
    # More media
    ".mov", ".avi", ".mkv", ".wav", ".flac", ".ogg", ".m4a", ".webm",
    ".m4v", ".3gp", ".aac", ".wma", ".opus", ".mpg", ".mpeg",
    # More archives/binaries
    ".7z", ".rar", ".iso", ".dmg", ".apk", ".aab", ".ipa", ".jar", ".war",
    ".class", ".pyc", ".pyo", ".o", ".a", ".lib", ".node", ".wasm",
    ".xz", ".bz2", ".zst", ".lz4", ".cab", ".msi", ".deb", ".rpm", ".snap", ".appimage",
    # DBs, logs, generated
    ".db", ".sqlite", ".sqlite3", ".log", ".map", ".mdb", ".accdb",
    ".min.js", ".min.css", ".bundle.js", ".chunk.js",
    # ML/data artifacts
    ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".weights",
    ".gguf", ".ggml", ".tflite", ".mlmodel", ".coreml", ".engine", ".trt",
    ".arrow", ".feather", ".orc", ".avro",
    # Fonts / design (extra)
    ".woff2", ".pfb", ".pfm",
    # Game engine binary/asset
    ".unity", ".uasset", ".umap", ".pak", ".fbx", ".blend", ".gltf.bin",
    ".ttc", ".dat", ".sav",
    # Certs / keys (binary forms, still noise for text scanning)
    ".p12", ".pfx", ".keystore", ".jks",
    # Office binaries (usually not source)
    ".xlsx", ".xls", ".pptx", ".ppt", ".docx", ".doc",
}

LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb", "bun.lock",
    "poetry.lock", "Cargo.lock", "Gemfile.lock", "mix.lock",
    "composer.lock", "Pipfile.lock", "uv.lock",
    # More ecosystems
    "Podfile.lock", "packages.lock.json", "flake.lock", "shard.lock",
    "gradle.lockfile", "conan.lock", "go.sum",
    "poetry.toml", "pdm.lock", "npm-shrinkwrap.json",
    "Package.resolved",  # Swift PM
    "renv.lock",  # R
    "cabal.project.freeze",  # Haskell
    "gems.locked",  # Ruby (bundler alt)
    "elm.json",  # not strictly a lockfile but pinned deps
    "vcpkg.json", "vcpkg-configuration.json",
    "deno.lock",
}

# Files that MUST be read in full — highest priority
CRITICAL_FILES = {
    "README.md", "README.rst", "README.txt", "README", "README.adoc",
    "package.json", "requirements.txt", "requirements-dev.txt", "pyproject.toml",
    "setup.py", "setup.cfg", "Pipfile", "environment.yml", "environment.yaml",
    "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
    "settings.gradle.kts",
    "Gemfile", "go.mod", "go.sum", "pubspec.yaml", "mix.exs",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml", "docker-compose.override.yml",
    "Makefile", "justfile", "Taskfile.yml",
    ".env.example", ".env.sample", ".env.template",
    # More language/ecosystem manifests
    "build.sbt", "project.clj", "deps.edn", "rebar.config", "dune-project", "dune",
    "CMakeLists.txt", "meson.build", "BUILD.bazel", "WORKSPACE", "WORKSPACE.bazel",
    "Rakefile", "Vagrantfile", "Procfile", "*.csproj", "*.sln", "*.fsproj", "*.vbproj",
    "Package.swift", "shard.yml", "*.rockspec",
    "conanfile.txt", "conanfile.py", "vcpkg.json",
    "dub.json", "dub.sdl",  # D language
    "stack.yaml", "*.cabal",  # Haskell
    "Project.toml", "Manifest.toml",  # Julia
    "rebar.lock", "mix.exs",
    # Mobile manifests
    "AndroidManifest.xml", "Info.plist", "Podfile", "app.json", "app.config.js",
    "build.xcconfig", "*.xcodeproj", "*.xcworkspace",
    # Version pinning / runtime hints
    ".nvmrc", ".ruby-version", ".python-version", ".tool-versions", ".node-version",
    ".java-version", ".swift-version", "runtime.txt",
    # Dockerfile variants
    "Dockerfile.dev", "Dockerfile.prod", "Dockerfile.test", "Containerfile",
    # License/ownership (cheap, useful context)
    "LICENSE", "LICENSE.md", "LICENSE.txt", "CODEOWNERS", "NOTICE", "COPYING",
    # CMS / framework markers
    "wp-config.php", "composer.json", "artisan", ".htaccess",
    "manage.py", "config/routes.rb", "next.config.js",
    # Monorepo root markers
    "lerna.json", "nx.json", "turbo.json", "pnpm-workspace.yaml", "rush.json",
    # Docs / changelog
    "CHANGELOG.md", "CONTRIBUTING.md",
}

# Config files — high value for architecture understanding
CONFIG_FILES = {
    "tsconfig.json", "jsconfig.json", "tsconfig.base.json",
    "next.config.js", "next.config.mjs", "next.config.ts",
    "vite.config.js", "vite.config.ts", "vite.config.mjs",
    "webpack.config.js", "webpack.config.ts",
    "tailwind.config.js", "tailwind.config.ts", "tailwind.config.mjs",
    "angular.json", "nuxt.config.ts", "nuxt.config.js",
    "svelte.config.js", "remix.config.js", "react-router.config.ts",
    "nest-cli.json",
    ".eslintrc.js", ".eslintrc.json", ".eslintrc.yaml", "eslint.config.js", "eslint.config.mjs",
    ".prettierrc", ".prettierrc.json", ".prettierrc.js", "biome.json", "biome.jsonc",
    "jest.config.js", "jest.config.ts", "vitest.config.ts", "ava.config.js",
    "alembic.ini", "alembic.cfg",
    "pytest.ini", "tox.ini", "noxfile.py", ".flake8", "ruff.toml", "mypy.ini",
    ".github/workflows",  # CI/CD
    "vercel.json", "netlify.toml", "fly.toml", "render.yaml", "railway.json",
    "kubernetes.yaml", "k8s.yaml", "helm", "Chart.yaml", "values.yaml", "skaffold.yaml",
    # More frontend framework configs
    "astro.config.mjs", "astro.config.ts", "gatsby-config.js", "quasar.config.js", "vue.config.js",
    "babel.config.js", ".babelrc", ".babelrc.json", "postcss.config.js", "rollup.config.js",
    "esbuild.config.js", "tsup.config.ts", "metro.config.js", "vue.config.ts",
    "unocss.config.ts", "windi.config.ts", "solid.config.ts", "qwik.config.ts",
    # Monorepo tooling
    "turbo.json", "nx.json", "lerna.json", "rush.json",
    "pnpm-workspace.yaml", ".yarnrc.yml", ".yarnrc",
    # Mobile / cross-platform
    "tauri.conf.json", "capacitor.config.json", "capacitor.config.ts", "ionic.config.json",
    "app.config.js", "expo.json", "eas.json", "metro.config.js", "react-native.config.js",
    # Testing / QA
    "cypress.config.js", "cypress.config.ts", "playwright.config.ts", "karma.conf.js",
    ".stylelintrc", "commitlint.config.js", ".huskyrc", "lint-staged.config.js",
    "wdio.conf.js", "nightwatch.conf.js", "codecept.conf.js",
    # CI/CD (beyond GitHub Actions)
    ".circleci/config.yml", ".gitlab-ci.yml", "azure-pipelines.yml",
    "Jenkinsfile", "buildspec.yml", "cloudbuild.yaml", "bitbucket-pipelines.yml",
    ".drone.yml", "appveyor.yml", ".travis.yml", "codemagic.yaml",
    # Infra as Code / cloud
    "*.tf", "*.tfvars", "terraform.tfvars", "serverless.yml", "sam-template.yaml",
    "template.yaml", "cdk.json", "Pulumi.yaml", "Pulumi.dev.yaml", "amplify.yml", "firebase.json",
    "ansible.cfg", "playbook.yml", "inventory.ini", "*.bicep", "cloudformation.yaml",
    # Data / ML configs
    "dvc.yaml", "params.yaml", "mlflow.yaml", "dbt_project.yml", "profiles.yml",
    "great_expectations.yml", "feature_store.yaml", "kubeflow.yaml",
    "hydra.yaml", ".hydra",
    # Databases / ORMs
    "prisma/schema.prisma", "schema.prisma", "knexfile.js", "sequelize.config.js",
    "typeorm.config.ts", "drizzle.config.ts", "liquibase.properties",
    # CMS
    "wp-config-sample.php", "gatsby-node.js",
    # Linters/formatters (other langs)
    ".rubocop.yml", ".solhint.json", "rustfmt.toml", ".clang-format", "checkstyle.xml",
    "phpcs.xml", "phpstan.neon", ".php-cs-fixer.php",
    # Editor/env
    ".editorconfig", ".nvmrc",
}

# Patterns for entry point detection
ENTRY_POINT_PATTERNS = [
    "main.*", "app.*", "index.*", "server.*", "manage.py",
    "wsgi.py", "asgi.py", "cli.*", "run.*", "start.*",
    # More entry conventions
    "__main__.py", "handler.*", "worker.*", "daemon.*", "bootstrap.*",
    "program.*", "Program.cs", "Startup.cs",
    "mod.rs", "lib.rs",
    "router.*", "routes.*",
    # Additional common entrypoints
    "entrypoint.*", "entry.*", "init.*", "setup.*", "launch.*",
    "Main.java", "Main.kt", "Application.java", "Application.kt",
    "Program.fs", "Application.cs",
    "AppDelegate.*", "SceneDelegate.*", "MainActivity.*", "MainApplication.*",
    "gateway.*", "api.*", "web.*", "service.*", "lambda_function.py", "function.py",
    "index.php", "web.php", "console.php", "artisan",
    "manage.rb", "config.ru",
]

# Source file extensions for language detection
LANG_EXTENSIONS: Dict[str, str] = {
    ".py": "Python", ".pyi": "Python", ".pyw": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript",
    ".jsx": "JavaScript", ".tsx": "TypeScript", ".go": "Go",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin", ".scala": "Scala", ".sc": "Scala",
    ".rb": "Ruby", ".rake": "Ruby", ".gemspec": "Ruby",
    ".php": "PHP", ".phtml": "PHP", ".cs": "C#",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++", ".hxx": "C++",
    ".c": "C", ".h": "C/C++",
    ".rs": "Rust", ".swift": "Swift", ".dart": "Dart",
    ".ex": "Elixir", ".exs": "Elixir", ".clj": "Clojure", ".cljs": "ClojureScript", ".cljc": "Clojure",
    ".hs": "Haskell", ".lhs": "Haskell", ".ml": "OCaml", ".mli": "OCaml", ".r": "R", ".R": "R",
    ".vue": "Vue", ".svelte": "Svelte",
    # More languages/ecosystems
    ".m": "Objective-C", ".mm": "Objective-C++",
    ".sql": "SQL", ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".fish": "Shell",
    ".ps1": "PowerShell", ".psm1": "PowerShell",
    ".lua": "Lua", ".pl": "Perl", ".pm": "Perl", ".erl": "Erlang", ".hrl": "Erlang",
    ".fs": "F#", ".fsx": "F#", ".fsi": "F#",
    ".jl": "Julia", ".nim": "Nim", ".nims": "Nim", ".zig": "Zig",
    ".sol": "Solidity", ".vy": "Vyper",
    ".groovy": "Groovy", ".gvy": "Groovy", ".gd": "GDScript", ".gdshader": "Godot Shader",
    ".razor": "Razor", ".cshtml": "Razor", ".vb": "VB.NET",
    ".twig": "Twig", ".astro": "Astro",
    ".proto": "Protocol Buffers", ".graphql": "GraphQL", ".gql": "GraphQL",
    ".cr": "Crystal", ".elm": "Elm", ".re": "ReScript", ".res": "ReScript",
    ".purs": "PureScript", ".f90": "Fortran", ".f95": "Fortran", ".for": "Fortran",
    ".asm": "Assembly", ".s": "Assembly", ".nasm": "Assembly",
    ".tf": "Terraform", ".hcl": "HCL",
    # Additional systems/functional/niche languages
    ".d": "D", ".v": "V", ".odin": "Odin", ".idr": "Idris", ".agda": "Agda",
    ".ada": "Ada", ".adb": "Ada", ".ads": "Ada",
    ".cob": "COBOL", ".cbl": "COBOL",
    ".pas": "Pascal", ".pp": "Pascal", ".dpr": "Delphi",
    ".rkt": "Racket", ".scm": "Scheme", ".ss": "Scheme",
    ".lisp": "Common Lisp", ".lsp": "Common Lisp", ".cl": "Common Lisp",
    ".prolog": "Prolog", ".pro": "Prolog",
    ".matlab": "MATLAB", ".mat": "MATLAB",
    ".sas": "SAS", ".do": "Stata",
    ".vhd": "VHDL", ".vhdl": "VHDL", ".v_hdl": "Verilog", ".sv": "SystemVerilog",
    ".apex": "Apex", ".cls": "Apex/ObjC-header",
    ".tcl": "Tcl", ".awk": "AWK",
    ".coffee": "CoffeeScript", ".litcoffee": "CoffeeScript",
    ".yaml": "YAML", ".yml": "YAML", ".json": "JSON", ".jsonc": "JSON with Comments",
    ".toml": "TOML", ".ini": "INI", ".xml": "XML",
    ".md": "Markdown", ".mdx": "MDX", ".rst": "reStructuredText",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS", ".sass": "Sass",
    ".less": "Less", ".styl": "Stylus",
    ".ipynb": "Jupyter Notebook",
    ".dockerfile": "Dockerfile",
    ".wasm.wat": "WebAssembly Text", ".wat": "WebAssembly Text",
    ".move": "Move", ".cairo": "Cairo", ".circom": "Circom",
    ".purs": "PureScript",
    ".bal": "Ballerina", ".mojo": "Mojo", "": "Unknown",
}

# Well-known project-type markers: presence of any of these strongly signals
# the overall project category, useful for high-level classification before
# deciding which language/framework heuristics to apply next.
PROJECT_TYPE_MARKERS: Dict[str, list] = {
    "Node/JS": ["package.json", "node_modules", "yarn.lock", "pnpm-lock.yaml"],
    "Python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
    "Rust": ["Cargo.toml", "Cargo.lock"],
    "Go": ["go.mod", "go.sum"],
    "Java/Kotlin (Gradle)": ["build.gradle", "build.gradle.kts", "settings.gradle"],
    "Java/Kotlin (Maven)": ["pom.xml"],
    "Ruby": ["Gemfile", "Gemfile.lock", ".ruby-version"],
    "PHP": ["composer.json", "composer.lock"],
    ".NET": ["*.csproj", "*.sln", "*.fsproj"],
    "Swift/iOS": ["Package.swift", "*.xcodeproj", "Podfile"],
    "Flutter/Dart": ["pubspec.yaml"],
    "Elixir": ["mix.exs"],
    "Haskell": ["stack.yaml", "*.cabal"],
    "Julia": ["Project.toml", "Manifest.toml"],
    "C/C++ (CMake)": ["CMakeLists.txt"],
    "C/C++ (Bazel)": ["WORKSPACE", "BUILD.bazel"],
    "Terraform/IaC": ["*.tf", "terraform.tfvars"],
    "Docker": ["Dockerfile", "docker-compose.yml"],
    "Kubernetes/Helm": ["Chart.yaml", "values.yaml", "kubernetes.yaml"],
    "Monorepo": ["turbo.json", "nx.json", "lerna.json", "pnpm-workspace.yaml", "rush.json"],
    "Data/ML": ["dvc.yaml", "mlflow.yaml", "dbt_project.yml", "environment.yml"],
    "Unity": ["Library", "ProjectSettings", "Assets", "*.unity"],
    "Godot": [".godot", "project.godot"],
    "WordPress": ["wp-config.php", "wp-content"],
    "Laravel": ["artisan", "composer.json"],
    "Rails": ["config/routes.rb", "Gemfile", "app/controllers"],
    "Django": ["manage.py", "wsgi.py"],
    "Next.js": ["next.config.js", "next.config.mjs", "next.config.ts"],
    "NestJS": ["nest-cli.json"],
}


class RepoSizeExceededError(Exception):
    pass


class RepoTooComplexError(Exception):
    """Raised when selected important files exceed MAX_FILE_COUNT budget.
    We refuse to analyze rather than truncate and lose accuracy.
    """
    pass


class RateLimitExceededError(Exception):
    pass


class GitHubClient:
    def __init__(self, installation_token: str = None):
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "RepoPilot"
        }
        if installation_token:
            self.headers["Authorization"] = f"Bearer {installation_token}"
        elif settings.github_token:
            self.headers["Authorization"] = f"Bearer {settings.github_token}"
        self._raw_download_semaphore = None

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

            parent = data.get("parent")
            return {
                "owner": owner,
                "name": repo,
                "full_name": data.get("full_name", f"{owner}/{repo}"),
                "description": data.get("description", ""),
                "stars": data.get("stargazers_count", 0),
                "watchers": data.get("subscribers_count", 0),
                "default_branch": data.get("default_branch", "main"),
                "size_kb": size_kb,
                "language": data.get("language", ""),
                "topics": data.get("topics", []),
                "homepage": data.get("homepage", ""),
                "open_issues": data.get("open_issues_count", 0),
                "license": data.get("license", {}).get("name", "None") if data.get("license") else "None",
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "pushed_at": data.get("pushed_at", ""),
                "forks": data.get("forks_count", 0),
                "archived": data.get("archived", False),
                "visibility": data.get("visibility", "public"),
                "is_fork": data.get("fork", False),
                "parent_repo": parent.get("full_name") if parent else None,
                "network_count": data.get("network_count", 0),
                "has_wiki": data.get("has_wiki", False),
                "has_discussions": data.get("has_discussions", False),
            }

    async def get_repo_enriched_metadata(self, repo_url: str) -> Dict[str, Any]:
        owner, repo = self._parse_repo_url(repo_url)
        
        async def fetch_json(url):
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                resp = await client.get(url, headers=self.headers)
                if resp.status_code == 200:
                    return resp.json()
                return {}
                
        async def fetch_contributors_count():
            # Get contributors count from Link header for per_page=1
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                resp = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=1",
                    headers=self.headers
                )
                if resp.status_code == 200:
                    link_header = resp.headers.get("Link", "")
                    if link_header:
                        import re
                        match = re.search(r'page=(\d+)>; rel="last"', link_header)
                        if match:
                            return int(match.group(1))
                    # If no link header, just count the elements
                    return len(resp.json())
                return 0

        # Run concurrently
        import asyncio
        meta_task = self.get_repo_metadata(repo_url)
        langs_task = fetch_json(f"https://api.github.com/repos/{owner}/{repo}/languages")
        release_task = fetch_json(f"https://api.github.com/repos/{owner}/{repo}/releases/latest")
        contrib_task = fetch_contributors_count()
        
        meta, langs, release, contribs = await asyncio.gather(
            meta_task, langs_task, release_task, contrib_task, return_exceptions=True
        )
        
        # Handle exceptions if any task failed
        if isinstance(meta, Exception):
            raise meta
        
        langs_data = langs if not isinstance(langs, Exception) else {}
        release_data = release if not isinstance(release, Exception) else {}
        contribs_data = contribs if not isinstance(contribs, Exception) else 0

        # Calculate language percentages
        total_bytes = sum(langs_data.values()) if langs_data else 0
        language_breakdown = {}
        if total_bytes > 0:
            language_breakdown = {
                lang: round((bytes_cnt / total_bytes) * 100, 1)
                for lang, bytes_cnt in langs_data.items()
            }

        return {
            "meta": meta,
            "language_breakdown": language_breakdown,
            "latest_release": release_data.get("tag_name", "None"),
            "contributors_count": contribs_data
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

            return self._prioritize_files(files)

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

        # Build nested directory tree + collect sizes
        nested_tree: Dict[str, Any] = {}
        all_paths = []
        file_sizes: Dict[str, int] = {}  # path -> bytes
        for item in tree_items:
            path = item["path"]
            item_type = item["type"]  # "blob" or "tree"
            parts = path.split("/")
            current = nested_tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            if item_type == "blob":
                size = item.get("size", 0)
                current[parts[-1]] = {"_type": "file", "_size": size}
                all_paths.append(path)
                file_sizes[path] = size
            else:
                current.setdefault(parts[-1], {})

        # Classify files (with glob support for CRITICAL_FILES patterns)
        classification = self.classify_files(all_paths)

        # Per-directory language breakdown
        dir_languages = self.get_directory_languages(all_paths)

        # Detect project types from marker files
        detected_project_types = self._detect_project_types(all_paths)

        filtered_files = [
            item for item in tree_items
            if item.get("type") == "blob" and not self._is_noise(item.get("path", ""))
        ]

        return {
            "nested_tree": nested_tree,
            "classification": classification,
            "directory_languages": dir_languages,
            "file_sizes": file_sizes,
            "files": self._prioritize_files(filtered_files),
            "total_files": len(all_paths),
            "total_analyzable_files": len(filtered_files),
            "detected_project_types": detected_project_types,
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

        # Separate CRITICAL_FILES into exact names vs glob patterns
        critical_exact = {f for f in CRITICAL_FILES if "*" not in f and "?" not in f}
        critical_globs = [f for f in CRITICAL_FILES if "*" in f or "?" in f]
        config_exact = {f for f in CONFIG_FILES if "*" not in f and "?" not in f}
        config_globs = [f for f in CONFIG_FILES if "*" in f or "?" in f]

        for path in paths:
            if self._is_noise(path):
                continue
            filename = path.split("/")[-1]
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

            # CI/CD
            if ".github/workflows" in path or (
                filename.endswith((".yml", ".yaml")) and
                any(k in path for k in ["github", "circleci", "travis", "gitlab", "azure-pipeline",
                                         "bitbucket", "drone", "appveyor", "codemagic", "buildspec"])
            ):
                classification["ci_cd"].append(path)
                continue

            # Critical files — exact match OR glob match
            if filename in critical_exact or any(fnmatch.fnmatch(filename, pat) for pat in critical_globs):
                classification["critical"].append(path)
                continue

            # Config files — exact match OR glob match
            if (
                filename in config_exact
                or any(fnmatch.fnmatch(filename, pat) for pat in config_globs)
                or any(fnmatch.fnmatch(filename, pat) for pat in ["*.config.*", ".eslint*", ".prettier*"])
            ):
                classification["config"].append(path)
                continue

            # Entry points
            if any(fnmatch.fnmatch(filename, pat) for pat in ENTRY_POINT_PATTERNS):
                depth = len(path.split("/"))
                if depth <= 4:
                    classification["entry_points"].append(path)
                    continue

            # Route / model / controller files
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

    def _detect_project_types(self, paths: List[str]) -> List[str]:
        """
        Identify project types from marker files present in the repo.
        Returns e.g. ["Python", "Django", "Docker", "Next.js"]
        """
        path_set = set(paths)
        filenames = {p.split("/")[-1] for p in paths}
        detected = []
        for project_type, markers in PROJECT_TYPE_MARKERS.items():
            for marker in markers:
                # Exact filename match
                if marker in filenames:
                    detected.append(project_type)
                    break
                # Glob match against filenames
                if "*" in marker and any(fnmatch.fnmatch(f, marker) for f in filenames):
                    detected.append(project_type)
                    break
                # Directory presence match (e.g. "app/controllers")
                if "/" in marker and any(marker in p for p in paths):
                    detected.append(project_type)
                    break
        return detected

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
        import asyncio
        if self._raw_download_semaphore is None:
            self._raw_download_semaphore = asyncio.Semaphore(settings.github_raw_concurrency)
            
        owner, repo = self._parse_repo_url(repo_url)
        if not branch:
            meta = await self.get_repo_metadata(repo_url)
            branch = meta["default_branch"]

        async with self._raw_download_semaphore:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                    headers=self.headers
                )
                if resp.status_code == 200:
                    return resp.text
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 429 or resp.status_code == 403:
                    print(f"Rate limited by raw.githubusercontent.com for {path}. Waiting 5s...")
                    await asyncio.sleep(5)
                    # Retry once
                    resp = await client.get(
                        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                        headers=self.headers
                    )
                    if resp.status_code == 200:
                        return resp.text
                    return None
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
        if self._raw_download_semaphore is None:
            self._raw_download_semaphore = asyncio.Semaphore(settings.github_raw_concurrency)

        async def _fetch(client: httpx.AsyncClient, path: str) -> tuple[str, Optional[str]]:
            try:
                async with self._raw_download_semaphore:
                    resp = await client.get(
                        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                        headers=self.headers,
                    )
                    if resp.status_code == 200:
                        content = resp.text
                    elif resp.status_code in {403, 429}:
                        retry_after = int(resp.headers.get("Retry-After", "5"))
                        logger.warning(
                            "Rate limited by raw.githubusercontent.com for %s. Waiting %ss...",
                            path,
                            retry_after,
                        )
                        await asyncio.sleep(retry_after)
                        resp = await client.get(
                            f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                            headers=self.headers,
                        )
                        content = resp.text if resp.status_code == 200 else None
                    else:
                        content = None
                if content and len(content) > max_chars_per_file:
                    content = content[:max_chars_per_file] + "\n... [truncated]"
                return path, content
            except Exception as e:
                print(f"Failed to fetch {path}: {e}")
                return path, None

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            results = await asyncio.gather(*(_fetch(client, p) for p in paths))
        return {path: content for path, content in results if content}
