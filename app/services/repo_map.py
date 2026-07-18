"""
RepoMap: Builds a structural understanding of a repository.
Parses manifests, detects frameworks, generates directory ASCII tree,
and extracts dependency information — all without embedding/LLM.
"""

import json
import re
from typing import Dict, List, Any, Optional


# Framework detection rules: config file presence → framework name + version source
FRAMEWORK_SIGNALS: List[Dict[str, Any]] = [
    # JS/TS Frameworks
    {"file": "next.config.js",      "framework": "Next.js",       "dep": "next"},
    {"file": "next.config.mjs",     "framework": "Next.js",       "dep": "next"},
    {"file": "next.config.ts",      "framework": "Next.js",       "dep": "next"},
    {"file": "nuxt.config.ts",      "framework": "Nuxt.js",       "dep": "nuxt"},
    {"file": "nuxt.config.js",      "framework": "Nuxt.js",       "dep": "nuxt"},
    {"file": "vite.config.ts",      "framework": "Vite",          "dep": "vite"},
    {"file": "vite.config.js",      "framework": "Vite",          "dep": "vite"},
    {"file": "angular.json",        "framework": "Angular",       "dep": "@angular/core"},
    {"file": "svelte.config.js",    "framework": "SvelteKit",     "dep": "@sveltejs/kit"},
    {"file": "remix.config.js",     "framework": "Remix",         "dep": "@remix-run/react"},
    {"file": "nest-cli.json",       "framework": "NestJS",        "dep": "@nestjs/core"},
    {"file": "expo.json",           "framework": "Expo",          "dep": "expo"},
    {"file": "metro.config.js",     "framework": "React Native",  "dep": "react-native"},
    # Python Frameworks
    {"file": "manage.py",           "framework": "Django",        "dep": "django"},
    {"file": "alembic.ini",         "framework": "SQLAlchemy/Alembic", "dep": "sqlalchemy"},
    {"file": "alembic.cfg",         "framework": "SQLAlchemy/Alembic", "dep": "sqlalchemy"},
    # Build tools
    {"file": "tailwind.config.js",  "framework": "Tailwind CSS",  "dep": "tailwindcss"},
    {"file": "tailwind.config.ts",  "framework": "Tailwind CSS",  "dep": "tailwindcss"},
    # Deployment
    {"file": "vercel.json",         "framework": "Vercel deployment", "dep": None},
    {"file": "netlify.toml",        "framework": "Netlify deployment", "dep": None},
    {"file": "fly.toml",            "framework": "Fly.io deployment", "dep": None},
    {"file": "render.yaml",         "framework": "Render deployment", "dep": None},
    {"file": "Dockerfile",          "framework": "Docker",        "dep": None},
    {"file": "docker-compose.yml",  "framework": "Docker Compose","dep": None},
    {"file": "docker-compose.yaml", "framework": "Docker Compose","dep": None},
]

# Key dependency → technology mapping for package.json / requirements.txt
KNOWN_DEPS: Dict[str, str] = {
    # JS
    "react": "React", "react-dom": "React DOM",
    "next": "Next.js", "nuxt": "Nuxt.js",
    "vue": "Vue.js", "svelte": "@sveltejs/kit",
    "express": "Express.js", "fastify": "Fastify",
    "@nestjs/core": "NestJS", "koa": "Koa.js",
    "hono": "Hono", "elysia": "Elysia",
    "typescript": "TypeScript",
    "tailwindcss": "Tailwind CSS",
    "framer-motion": "Framer Motion", "motion": "Motion (Framer)",
    "redux": "Redux", "react-redux": "React Redux",
    "zustand": "Zustand", "jotai": "Jotai", "recoil": "Recoil",
    "@clerk/nextjs": "Clerk Auth", "@clerk/react": "Clerk Auth",
    "next-auth": "NextAuth.js", "@auth/core": "Auth.js",
    "mongoose": "Mongoose ODM", "prisma": "Prisma ORM",
    "@prisma/client": "Prisma Client", "drizzle-orm": "Drizzle ORM",
    "openai": "OpenAI SDK", "@anthropic-ai/sdk": "Anthropic SDK",
    "firebase": "Firebase", "@supabase/supabase-js": "Supabase",
    "socket.io": "Socket.io", "ws": "WebSocket (ws)",
    "graphql": "GraphQL", "apollo-server": "Apollo Server",
    "@apollo/client": "Apollo Client",
    "stripe": "Stripe", "resend": "Resend",
    "zod": "Zod (validation)", "yup": "Yup (validation)",
    "axios": "Axios", "swr": "SWR", "@tanstack/react-query": "TanStack Query",
    "styled-components": "Styled Components",
    # Python
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django",
    "starlette": "Starlette", "litestar": "Litestar",
    "sqlalchemy": "SQLAlchemy", "alembic": "Alembic",
    "pydantic": "Pydantic", "pydantic-settings": "Pydantic Settings",
    "langchain": "LangChain", "langchain-groq": "LangChain Groq",
    "langchain-openai": "LangChain OpenAI",
    "openai": "OpenAI SDK", "anthropic": "Anthropic SDK",
    "celery": "Celery", "redis": "Redis",
    "httpx": "HTTPX", "aiohttp": "aiohttp", "requests": "Requests",
    "pymongo": "PyMongo", "motor": "Motor (async MongoDB)",
    "asyncpg": "asyncpg", "psycopg2": "psycopg2",
    "pytest": "pytest", "uvicorn": "Uvicorn", "gunicorn": "Gunicorn",
    "pgvector": "pgvector",
    # Mobile / Other
    "expo": "Expo", "react-native": "React Native",
}


class RepoMap:
    """
    Builds a structural map of a repository without LLM calls.
    Used to provide structural context to the summarizer.
    """

    def build_ascii_tree(self, nested_tree: Dict[str, Any], prefix: str = "", max_depth: int = 4, current_depth: int = 0) -> str:
        """Generates an ASCII directory tree string from nested tree dict."""
        if current_depth >= max_depth:
            return ""
        lines = []
        items = sorted(nested_tree.items(), key=lambda x: (isinstance(x[1], dict) and "_type" not in x[1], x[0]))
        for i, (name, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            is_file = isinstance(value, dict) and value.get("_type") == "file"

            if is_file:
                lines.append(f"{prefix}{connector}{name}")
            else:
                lines.append(f"{prefix}{connector}{name}/")
                extension = "    " if is_last else "│   "
                subtree = {k: v for k, v in value.items() if k != "_type" and k != "_size"}
                if subtree:
                    lines.append(self.build_ascii_tree(subtree, prefix + extension, max_depth, current_depth + 1))

        return "\n".join(filter(None, lines))

    def parse_package_json(self, content: str) -> Dict[str, Any]:
        """Extract structured info from package.json."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {}

        all_deps = {}
        all_deps.update(data.get("dependencies", {}))
        all_deps.update(data.get("devDependencies", {}))

        known_tech = {}
        for dep, version in all_deps.items():
            tech_name = KNOWN_DEPS.get(dep)
            if tech_name:
                known_tech[tech_name] = version.lstrip("^~>=")

        return {
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "version": data.get("version", ""),
            "scripts": data.get("scripts", {}),
            "all_dependencies": all_deps,
            "identified_technologies": known_tech,
            "main": data.get("main", ""),
            "type": data.get("type", ""),  # "module" vs "commonjs"
            "engines": data.get("engines", {}),
        }

    def parse_requirements_txt(self, content: str) -> Dict[str, Any]:
        """Extract structured info from requirements.txt."""
        deps = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-r"):
                continue
            # Handle: package==1.0, package>=1.0, package~=1.0, package
            match = re.match(r"^([a-zA-Z0-9_\-\[\]]+)([>=<!~^]+.*)?$", line)
            if match:
                name = match.group(1).split("[")[0].lower()
                version = match.group(2) or ""
                deps[name] = version.strip()

        known_tech = {}
        for dep, version in deps.items():
            tech_name = KNOWN_DEPS.get(dep)
            if tech_name:
                clean_version = version.lstrip(">=~^<!=")
                known_tech[tech_name] = clean_version

        return {
            "all_dependencies": deps,
            "identified_technologies": known_tech,
        }

    def parse_pyproject_toml(self, content: str) -> Dict[str, Any]:
        """Extract basic info from pyproject.toml (basic TOML parsing, no toml lib needed)."""
        deps = {}
        identified = {}

        # Extract dependencies section (handles poetry and PEP 518 style)
        in_deps = False
        for line in content.splitlines():
            stripped = line.strip()
            if "[tool.poetry.dependencies]" in stripped or "[project]" in stripped:
                in_deps = True
                continue
            if stripped.startswith("[") and in_deps:
                in_deps = False
            if in_deps and "=" in stripped and not stripped.startswith("#"):
                parts = stripped.split("=", 1)
                name = parts[0].strip().strip('"').lower()
                if name not in {"python", "name", "version", "description", "readme"}:
                    version = parts[1].strip().strip('"').strip("'")
                    deps[name] = version
                    tech_name = KNOWN_DEPS.get(name)
                    if tech_name:
                        identified[tech_name] = version

        return {
            "all_dependencies": deps,
            "identified_technologies": identified,
        }

    def detect_frameworks(
        self,
        classification: Dict[str, List[str]],
        parsed_manifests: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Cross-references config file presence and parsed dependency lists
        to produce a confident framework detection map with versions.
        Returns {framework_name: version_or_detected}.
        """
        detected: Dict[str, str] = {}
        all_file_paths = set(
            classification.get("critical", []) +
            classification.get("config", []) +
            classification.get("entry_points", []) +
            classification.get("ci_cd", [])
        )
        filenames = {p.split("/")[-1] for p in all_file_paths}

        # Config-file-based signals
        for signal in FRAMEWORK_SIGNALS:
            if signal["file"] in filenames:
                framework = signal["framework"]
                dep_key = signal["dep"]
                # Try to get version from parsed manifest
                version = ""
                if dep_key:
                    for manifest_data in parsed_manifests.values():
                        all_deps = manifest_data.get("all_dependencies", {})
                        if dep_key in all_deps:
                            version = all_deps[dep_key].lstrip("^~>=<!")
                            break
                detected[framework] = version or "detected"

        # Dependency-based signals (from manifest parsing)
        for manifest_data in parsed_manifests.values():
            for tech, version in manifest_data.get("identified_technologies", {}).items():
                if tech not in detected:
                    detected[tech] = version

        return detected

    def build_component_names(
        self,
        classification: Dict[str, List[str]],
        dir_languages: Dict[str, Dict[str, int]],
    ) -> List[Dict[str, str]]:
        """
        Infer architectural components from directory structure.
        Returns list of {name, directory, primary_language, estimated_role}.
        """
        components = []
        # Top-level source directories often correspond to components
        seen_dirs = set()
        all_source = (
            classification.get("source", []) +
            classification.get("entry_points", [])
        )

        for path in all_source:
            parts = path.split("/")
            if len(parts) >= 2:
                top_dir = parts[0]
                if top_dir not in seen_dirs:
                    seen_dirs.add(top_dir)
                    langs = dir_languages.get(top_dir, {})
                    primary_lang = max(langs, key=langs.get) if langs else "Unknown"
                    role = self._infer_directory_role(top_dir, langs)
                    components.append({
                        "name": top_dir,
                        "directory": f"/{top_dir}",
                        "primary_language": primary_lang,
                        "estimated_role": role,
                    })

        return components

    def _infer_directory_role(self, dir_name: str, langs: Dict[str, int]) -> str:
        name_lower = dir_name.lower()
        role_signals = {
            "frontend": ["src", "client", "frontend", "web", "ui", "app", "pages", "components", "views"],
            "backend": ["server", "backend", "api", "services", "handlers", "controllers"],
            "database": ["db", "database", "migrations", "models", "schema"],
            "tests": ["tests", "test", "__tests__", "spec", "specs"],
            "infrastructure": ["infra", "infrastructure", "terraform", "k8s", "kubernetes", "helm", "deploy"],
            "documentation": ["docs", "doc", "documentation", "wiki"],
            "scripts/tooling": ["scripts", "tools", "utils", "helpers", "bin"],
            "configuration": ["config", "configs", "settings"],
            "shared/lib": ["lib", "libs", "shared", "common", "packages", "core"],
        }
        for role, signals in role_signals.items():
            if name_lower in signals:
                return role
        return "source code"

    def format_dependencies_for_llm(self, parsed_manifests: Dict[str, Any]) -> str:
        """Format parsed manifest data into clean text for LLM context."""
        lines = []
        for filename, data in parsed_manifests.items():
            lines.append(f"=== {filename} ===")
            if data.get("name"):
                lines.append(f"Project: {data['name']} v{data.get('version', '')}")
            if data.get("description"):
                lines.append(f"Description: {data['description']}")
            if data.get("scripts"):
                key_scripts = {k: v for k, v in data["scripts"].items()
                               if k in {"dev", "build", "start", "test", "lint"}}
                if key_scripts:
                    lines.append(f"Scripts: {json.dumps(key_scripts)}")
            if data.get("identified_technologies"):
                lines.append(f"Identified technologies:")
                for tech, ver in data["identified_technologies"].items():
                    lines.append(f"  - {tech}: {ver}")
            if data.get("all_dependencies"):
                dep_names = list(data["all_dependencies"].keys())[:30]
                lines.append(f"All dependencies ({len(data['all_dependencies'])} total): {', '.join(dep_names)}")
            lines.append("")

        return "\n".join(lines)
