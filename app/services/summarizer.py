"""
Multi-pass repository summarizer.

Pass 0: Ensure repo is embedded (NVIDIA embeddings) — delegated to ProcessingPipeline
Pass 1: Structural map + full reads of key files (no LLM)
Pass 2: Targeted similarity search (no LLM)
Pass 3: Parallel file-group summaries — Groq llama-3.1-8b-instant (fast, configurable)
Pass 4: Final synthesis — NVIDIA nemotron reasoning LLM (thinking ON, 1M ctx)

Intermediate caching:
  - Pass 0 embeddings: cached in DB keyed by commit_sha (handled by ProcessingPipeline)
  - Pass 3 group summaries: cached in repositories.group_summaries keyed by commit_sha
  - Pass 4 brief: cached in repositories.brief keyed by commit_sha

Fallback:
  - If Pass 4 fails: serve degraded "lite brief" from Pass 1 + Pass 3 (no LLM)
"""

import asyncio
import json
import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError
from app.models.schema import Repository
from app.models.brief import RepoBrief, LLMRepoBrief, ComponentInfo, TechItem, FileReference
from app.services.llm import get_fast_llm, get_reasoning_llm
from app.services.pipeline import ProcessingPipeline
from app.services.repo_map import RepoMap
from app.services.github_client import GitHubClient
from app.config import settings
from app.database import async_session
from sqlalchemy.future import select

logger = logging.getLogger(__name__)

# Max chars to read per individual critical/entry-point file
MAX_CRITICAL_FILE_CHARS = settings.max_file_content_chars

# How many source chunks to retrieve per targeted query
CHUNKS_PER_QUERY = 5

# Max total context chars fed to each group summary call (fast LLM)
MAX_GROUP_CONTEXT_CHARS = 15_000

# Parallel group summary buckets
MAX_SOURCE_GROUPS = 10

# Context caps for final synthesis
SYNTHESIS_CRITICAL_FILES_CHARS = 40_000
SYNTHESIS_SUMMARIES_CHARS = 20_000
SYNTHESIS_SEARCH_CHARS = 20_000

TECH_EVIDENCE_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    "mongodb": [
        ("mongodb", "MongoDB driver import/reference"),
        ("mongoose", "Mongoose ODM import/reference"),
        ("mongoclient", "MongoClient usage"),
        ("mongoose.connect", "Mongoose connection call"),
        ("mongodb://", "MongoDB connection URI"),
        ("mongodb+srv://", "MongoDB Atlas connection URI"),
        ("pymongo", "PyMongo import/reference"),
        ("motor.motor_asyncio", "Motor async MongoDB import/reference"),
    ],
    "redis": [("redis", "Redis import/reference"), ("redis://", "Redis connection URI")],
    "postgres": [
        ("postgres", "Postgres reference"),
        ("postgresql://", "Postgres connection URI"),
        ("psycopg2", "psycopg2 import/reference"),
        ("asyncpg", "asyncpg import/reference"),
        ("pgvector", "pgvector import/reference"),
    ],
    "mysql": [("mysql", "MySQL import/reference"), ("mysql://", "MySQL connection URI")],
    "sqlite": [("sqlite", "SQLite import/reference"), ("sqlite://", "SQLite connection URI")],
}


def _repair_json_string(json_str: str) -> str:
    """Fix common LLM JSON syntax issues like trailing commas or unescaped tabs."""
    cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)
    cleaned = cleaned.replace('\t', '  ')
    return cleaned


def _extract_json_payload(raw_content: Any) -> Dict[str, Any]:
    """
    Robust multi-strategy JSON extractor for LLM outputs.
    Handles thinking CoT blocks (<think>...</think>), markdown code blocks,
    non-greedy brace scanning, and minor JSON syntax repairs.
    """
    if isinstance(raw_content, list):
        parts = []
        for block in raw_content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        text = "\n".join(parts)
    elif not isinstance(raw_content, str):
        text = str(raw_content)
    else:
        text = raw_content

    # 0. Strip CoT reasoning tags if present
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text_clean:
        text_clean = text

    # 1. Check for markdown code blocks ```json ... ``` or ``` ... ```
    code_block_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text_clean, re.DOTALL)
    for block in code_block_matches:
        try:
            return json.loads(block)
        except Exception:
            repaired = _repair_json_string(block)
            try:
                return json.loads(repaired)
            except Exception:
                continue

    # 2. Balanced brace depth scanning for outer JSON object
    start_idx = text_clean.find('{')
    while start_idx != -1:
        depth = 0
        in_string = False
        escape = False
        end_idx = -1
        for i in range(start_idx, len(text_clean)):
            char = text_clean[i]
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
        if end_idx != -1:
            candidate = text_clean[start_idx:end_idx + 1]
            try:
                return json.loads(candidate)
            except Exception:
                repaired = _repair_json_string(candidate)
                try:
                    return json.loads(repaired)
                except Exception:
                    pass
        start_idx = text_clean.find('{', start_idx + 1)

    # 3. Fallback greedy regex search
    greedy_match = re.search(r'\{.*\}', text_clean, re.DOTALL)
    if greedy_match:
        candidate = greedy_match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            repaired = _repair_json_string(candidate)
            try:
                return json.loads(repaired)
            except Exception as e:
                raise ValueError(f"Failed to parse LLM JSON candidate: {e}")

    raise ValueError("No valid JSON object found in LLM output")


# ─────────────────────────────────────────────────────────────────────────────
# Static analysis helpers (deterministic — injected as ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_dependency_usage(
    parsed_manifests: Dict[str, Any],
    full_file_contents: Dict[str, str],
    search_context: Dict[str, List[Dict[str, Any]]],
    all_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Cross-reference manifest deps against observed imports in analyzed code.
    Returns {dep_name: {status, evidence}}.

    Import data sources:
    1. full_file_contents — regex scan for import/require/from
    2. search_context chunk metadata — chunker.py already extracts imports
    3. all stored selected chunks — prevents retrieval misses from becoming false issues
    """
    # Collect all declared deps
    declared: Dict[str, str] = {}  # dep_name -> version
    for manifest_data in parsed_manifests.values():
        declared.update(manifest_data.get("all_dependencies", {}))

    if not declared:
        return {}

    imported_raw: set = set()
    dep_evidence: Dict[str, List[str]] = {}

    # Regex patterns: Python, JS/TS
    import_patterns = [
        r"^\s*from\s+([\w.\-@/]+)\s+import\b",                           # Python
        r"^\s*import\s+([\w.\-@/]+)\b",                                  # Python / simple JS default
        r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",                # JS/TS static
        r"^\s*import\s+['\"]([^'\"]+)['\"]",                             # JS/TS side-effect import
        r"require\(\s*['\"]([^'\"]+)['\"]\s*\)",                         # CommonJS
        r"await\s+import\(\s*['\"]([^'\"]+)['\"]\s*\)",                  # Dynamic import
    ]

    def normalize_module_name(raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("@"):
            parts = raw.split("/")
            return "/".join(parts[:2]).lower() if len(parts) >= 2 else raw.lower()
        return raw.split("/")[0].split(".")[0].lower()

    def extract_module_names(text: str) -> set:
        found = set()
        for line in text.splitlines():
            for pat in import_patterns:
                m = re.search(pat, line)
                if not m:
                    continue
                for g in m.groups():
                    if not g:
                        continue
                    root = normalize_module_name(g)
                    if root and len(root) > 1 and not root.startswith("."):
                        found.add(root)
        return found

    def add_imports_from_text(text: str, source_label: str) -> None:
        for module_name in extract_module_names(text):
            imported_raw.add(module_name)
            dep_evidence.setdefault(module_name, [])
            if source_label not in dep_evidence[module_name]:
                dep_evidence[module_name].append(source_label)

    for path, content in full_file_contents.items():
        add_imports_from_text(content, path)

    # Also pull from chunk metadata (chunker already extracted imports)
    for chunks in search_context.values():
        for chunk in chunks:
            path = chunk.get("metadata", {}).get("path", "unknown")
            add_imports_from_text(chunk.get("text", ""), path)
            for imp_line in chunk.get("metadata", {}).get("imports", []):
                add_imports_from_text(imp_line, path)

    for chunk in all_chunks or []:
        metadata = chunk.get("metadata", {})
        path = metadata.get("path", "unknown")
        add_imports_from_text(chunk.get("text", ""), path)
        for imp_line in metadata.get("imports", []):
            add_imports_from_text(imp_line, path)

    # Match declared deps against imported
    result = {}
    for dep, version in declared.items():
        dep_key = normalize_module_name(dep)
        dep_compact = dep_key.replace("-", "").replace("_", "")
        matching_imports = [
            imp for imp in imported_raw
            if dep_key == imp
            or dep_compact == imp.replace("-", "").replace("_", "")
            or dep_key in imp
            or imp in dep_key
        ]
        evidence = []
        for imp in matching_imports:
            evidence.extend(dep_evidence.get(imp, []))
        result[dep] = {
            "status": "observed_used" if matching_imports else "not_observed_in_selected_code",
            "version": version,
            "matched_imports": sorted(set(matching_imports)),
            "evidence": sorted(set(evidence))[:8],
        }

    return result


def _build_positive_tech_evidence(
    all_chunks: List[Dict[str, Any]],
    full_file_contents: Dict[str, str],
) -> Dict[str, List[str]]:
    """Find positive technology usage evidence across selected code chunks."""
    evidence: Dict[str, List[str]] = {}

    def scan_text(text: str, source_label: str) -> None:
        text_lower = text.lower()
        for tech, patterns in TECH_EVIDENCE_PATTERNS.items():
            for needle, description in patterns:
                if needle in text_lower:
                    value = f"{source_label}: {description}"
                    evidence.setdefault(tech, [])
                    if value not in evidence[tech]:
                        evidence[tech].append(value)

    for path, content in full_file_contents.items():
        scan_text(content, path)

    for chunk in all_chunks:
        metadata = chunk.get("metadata", {})
        path = metadata.get("path", "unknown")
        start = metadata.get("start_line")
        label = f"{path}:L{start}" if start else path
        scan_text(chunk.get("text", ""), label)
        for import_line in metadata.get("imports", []):
            scan_text(import_line, label)

    return {tech: refs[:10] for tech, refs in evidence.items()}


def _analyze_env_vars(
    full_file_contents: Dict[str, str],
    search_context: Dict[str, List[Dict[str, Any]]],
    all_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """
    Grep code for env var reads. Returns {VAR_NAME: [file_path, ...]}.

    Patterns:
    - Python: os.getenv("X"), os.environ["X"], os.environ.get("X"), settings.X
    - JS/TS: process.env.X, process.env["X"]
    - Generic: ${VAR_NAME} in yaml/toml
    """
    patterns = [
        r'os\.getenv\(["\'](\w+)["\']\)',
        r'os\.environ\[["\'](\w+)["\']\]',
        r'os\.environ\.get\(["\'](\w+)["\']\)',
        r'process\.env\.([A-Z_][A-Z0-9_]+)',
        r'process\.env\[["\']([A-Z_][A-Z0-9_]+)["\']\]',
        r'\$\{([A-Z_][A-Z0-9_]+)\}',
    ]

    env_vars: Dict[str, List[str]] = {}

    def scan_text(text: str, source_label: str) -> None:
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            for pat in patterns:
                for m in re.finditer(pat, line):
                    var = m.group(1)
                    key = f"{source_label}:L{lineno}"
                    env_vars.setdefault(var, [])
                    if key not in env_vars[var]:
                        env_vars[var].append(key)

    for file_path, content in full_file_contents.items():
        scan_text(content, file_path)

    for chunks in search_context.values():
        for chunk in chunks:
            path = chunk["metadata"].get("path", "unknown")
            start = chunk["metadata"].get("start_line", "")
            label = f"{path}:L{start}" if start else path
            scan_text(chunk["text"], label)

    for chunk in all_chunks or []:
        path = chunk.get("metadata", {}).get("path", "unknown")
        start = chunk.get("metadata", {}).get("start_line", "")
        label = f"{path}:L{start}" if start else path
        scan_text(chunk.get("text", ""), label)

    return env_vars


def _format_dep_usage(dep_usage: Dict[str, Dict[str, Any]]) -> str:
    """Format dependency usage diff for LLM prompt injection."""
    if not dep_usage:
        return "No manifest found or no dependencies declared."

    used = sorted(k for k, v in dep_usage.items() if v.get("status") == "observed_used")
    not_observed = sorted(k for k, v in dep_usage.items() if v.get("status") == "not_observed_in_selected_code")

    lines = []
    if used:
        lines.append(f"OBSERVED USED in analyzed code ({len(used)}):")
        for dep in used[:40]:
            evidence = ", ".join(dep_usage[dep].get("evidence", [])[:3])
            lines.append(f"  - {dep}: {evidence or 'import/reference found'}")
    if not_observed:
        lines.append(
            f"NOT OBSERVED in selected analyzed code ({len(not_observed)}), "
            "but this is NOT proof of an unused dependency and MUST NOT be flagged as an issue by itself:"
        )
        lines.append("  - " + ", ".join(not_observed[:40]))
    return "\n".join(lines)


def _format_tech_evidence(tech_evidence: Dict[str, List[str]]) -> str:
    if not tech_evidence:
        return "No known database/cache technology usage evidence found in selected code chunks."
    lines = []
    for tech, refs in sorted(tech_evidence.items()):
        lines.append(f"  - {tech}:")
        for ref in refs[:6]:
            lines.append(f"      {ref}")
    return "\n".join(lines)


def _format_env_vars(env_vars: Dict[str, List[str]]) -> str:
    """Format env var grep results for LLM prompt injection."""
    if not env_vars:
        return "No env var reads detected in code."
    lines = []
    for var, locations in sorted(env_vars.items()):
        loc_str = ", ".join(locations[:3])
        if len(locations) > 3:
            loc_str += f" (+{len(locations)-3} more)"
        lines.append(f"  {var}: {loc_str}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main summarizer
# ─────────────────────────────────────────────────────────────────────────────

class RepoSummarizer:
    def __init__(self):
        # Phase 3: Groq (fast, configurable model)
        self.fast_llm = get_fast_llm()
        # Phase 4: NVIDIA NIM (reasoning, structured output)
        self.reasoning_llm = get_reasoning_llm()
        self.pipeline = ProcessingPipeline()
        self.repo_map_builder = RepoMap()
        self.github = GitHubClient()

    async def generate_quick_summary(self, repo_url: str) -> str:
        """
        Fast AI summary of the repo based on metadata, README, and package files.
        Uses Groq (fast_llm) to produce a rapid response.
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        import fnmatch

        print(f"\n[Quick Summary] Starting for {repo_url}...")
        meta = await self.github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]

        tree_data = await self.github.get_file_tree_structure(repo_url, branch)
        classification = tree_data["classification"]

        files_to_read = []
        # Get README
        readme_files = [f for f in classification["critical"] if f.lower().split("/")[-1].startswith("readme")]
        if readme_files:
            files_to_read.append(readme_files[0])
            
        # Get package manifests
        manifest_names = {
    # JavaScript / TypeScript / Node
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "bun.lockb", "tsconfig.json", "deno.json", "deno.jsonc",

    # Python
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Pipfile", "Pipfile.lock", "poetry.lock", "environment.yml",
    "tox.ini", "conda.yaml",

    # Go
    "go.mod", "go.sum",

    # Java / Kotlin / JVM
    "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
    "settings.gradle.kts", "ivy.xml",

    # Rust
    "Cargo.toml", "Cargo.lock",

    # Ruby
    "Gemfile", "Gemfile.lock", "*.gemspec",

    # PHP
    "composer.json", "composer.lock",

    # C# / .NET
    "*.csproj", "*.fsproj", "*.vbproj", "*.sln", "packages.config",
    "nuget.config", "Directory.Build.props",

    # C / C++
    "CMakeLists.txt", "Makefile", "makefile", "meson.build",
    "conanfile.txt", "conanfile.py", "vcpkg.json",

    # Swift / Objective-C
    "Package.swift", "Podfile", "Podfile.lock", "*.xcodeproj",
    "*.xcworkspace",

    # Dart / Flutter
    "pubspec.yaml", "pubspec.lock",

    # Elixir / Erlang
    "mix.exs", "mix.lock", "rebar.config",

    # Haskell
    "*.cabal", "stack.yaml", "package.yaml",

    # Scala
    "build.sbt",

    # Perl
    "Makefile.PL", "cpanfile",

    # R
    "DESCRIPTION", "renv.lock",

    # Julia
    "Project.toml", "Manifest.toml",

    # Lua
    "*.rockspec",

    # Zig
    "build.zig",

    # Nix
    "flake.nix", "default.nix", "shell.nix",

    # Containers / Infra
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Vagrantfile", "Procfile",

    # Terraform / IaC
    "main.tf", "terraform.tfvars",

    # General / monorepo tooling
    "lerna.json", "nx.json", "turbo.json", "rush.json",
    "Makefile.toml", "justfile", "Justfile",

    # Version control / project markers
    ".gitmodules",
}
        manifest_files = []
        for f in classification["critical"]:
            filename = f.split("/")[-1]
            if any(fnmatch.fnmatch(filename, pat) for pat in manifest_names):
                manifest_files.append(f)
                
        files_to_read.extend(manifest_files[:5]) # limit to 5 manifests

        print(f"  [Quick Summary] Reading {len(files_to_read)} files for fast analysis...")
        full_file_contents = await self.github.get_files_content_batch(
            repo_url,
            files_to_read,
            branch=branch,
            max_chars_per_file=8000,
        )

        context_blocks = []
        for path, content in full_file_contents.items():
            context_blocks.append(f"--- {path} ---\n{content[:4000]}")
            
        context_str = "\n\n".join(context_blocks)
        
        # Only include meaningful metadata for the LLM
        meta_parts = [f"Repository: {meta.get('full_name')}"]
        if meta.get('description'):
            meta_parts.append(f"Description: {meta.get('description')}")
        if meta.get('language'):
            meta_parts.append(f"Primary Language: {meta.get('language')}")
            
        meta_str = "\n".join(meta_parts)

        messages = [
            SystemMessage(content=(
                "You are an expert developer assistant. Provide a rapid, high-level summary of this repository. "
                "Based on the repository metadata, README, and package dependencies provided, describe: "
                "1) What the project is and its primary purpose. "
                "2) The core technology stack and frameworks used. "
                "Keep the summary concise (max 3-4 paragraphs) and highly technical."
            )),
            HumanMessage(content=f"{meta_str}\n\nFiles Content:\n{context_str}"),
        ]

        print("  [Quick Summary] Calling LLM...")
        result = await self.fast_llm.ainvoke(messages)
        print("  [Quick Summary] Done.")
        return result.content

    async def generate_brief(self, repo_url: str, job_id: Optional[str] = None) -> RepoBrief:
        """
        Main entry: 4-pass pipeline to generate comprehensive, accurate repo brief.
        job_id is optional — used to update phase progress in DB.
        """
        from app.services.job_manager import job_manager

        async def set_phase(phase: str) -> None:
            if job_id:
                await job_manager.update_phase(job_id, phase)

        print(f"\n{'='*60}")
        print(f"Starting multi-pass analysis for: {repo_url}")
        print(f"{'='*60}")

        # ── Pass 0: Ensure repo is embedded ──────────────────────────
        await set_phase("embedding")
        t0 = time.time()
        print("\n[Pass 0] Embedding repository chunks...")
        process_stats = await self.pipeline.process_repo(repo_url)
        commit_sha = process_stats["commit_sha"]
        # tree_data from process_repo (avoids a second get_file_tree_structure() call)
        cached_tree_data = process_stats.get("tree_data")  # None on cache hit
        print(f"  Status: {process_stats['status']}, chunks: {process_stats.get('chunks_count', '?')}, "
              f"duration: {time.time()-t0:.1f}s")

        # ── Pass 1: Structural map + full reads of key files ──────────
        await set_phase("structural_map")
        t1 = time.time()
        print("\n[Pass 1] Building structural map and reading key files...")
        structural_context = await self._build_structural_context(repo_url, tree_data=cached_tree_data)
        print(f"  Done in {time.time()-t1:.1f}s")

        # ── Pass 2: Targeted similarity search ────────────────────────
        await set_phase("search")
        t2 = time.time()
        print("\n[Pass 2] Targeted similarity search across architecture topics...")
        search_context = await self._build_search_context(
            repo_url,
            structural_context["detected_frameworks"],
            structural_context["classification"],
        )
        print(f"  Done in {time.time()-t2:.1f}s")

        print("\n[Pass 2b] Building full selected-chunk evidence ledger...")
        all_chunks = await self.pipeline.get_repository_chunks(repo_url)
        print(f"  Evidence ledger covers {len(all_chunks)} selected chunks")

        # ── Pass 3: Parallel file-group summaries (with caching) ──────
        await set_phase("group_summaries")
        t3 = time.time()
        print("\n[Pass 3] Generating file-group summaries (NVIDIA NIM, map-reduce)...")
        group_summaries = await self._generate_group_summaries(
            structural_context, search_context, repo_url, commit_sha
        )
        print(f"  Done in {time.time()-t3:.1f}s, {len(group_summaries)} summaries")

        # ── Static analysis (inject as ground truth for Pass 4) ──────
        dep_usage = _analyze_dependency_usage(
            structural_context["parsed_manifests"],
            structural_context["full_file_contents"],
            search_context,
            all_chunks,
        )
        env_vars = _analyze_env_vars(
            structural_context["full_file_contents"],
            search_context,
            all_chunks,
        )
        tech_evidence = _build_positive_tech_evidence(
            all_chunks,
            structural_context["full_file_contents"],
        )
        structural_context["tech_evidence"] = tech_evidence
        
        # ── Pass 3.5: Graph Query ─────────────────────────────────────
        await set_phase("graph_query")
        t35 = time.time()
        print("\n[Pass 3.5] Querying structural graph data...")
        graph_data = await self._query_graph_data(
            repo_url, 
            structural_context["classification"]["entry_points"]
        )
        print(f"  Done in {time.time()-t35:.1f}s")

        # ── Pass 4: Final synthesis → structured RepoBrief ────────────
        await set_phase("synthesis")
        t4 = time.time()
        print("\n[Pass 4] Synthesizing final RepoBrief (NVIDIA, thinking ON)...")
        try:
            brief = await self._synthesize_brief(
                repo_url=repo_url,
                commit_sha=commit_sha,
                structural_context=structural_context,
                search_context=search_context,
                group_summaries=group_summaries,
                dep_usage=dep_usage,
                env_vars=env_vars,
                tech_evidence=tech_evidence,
                graph_data=graph_data,
                selected_files=process_stats.get("selected_files", []),
                project_types=structural_context.get("project_types", []),
            )
            print(f"  Done in {time.time()-t4:.1f}s")
        except Exception as e:
            print(f"  [Pass 4] FAILED: {e}. Serving lite brief fallback.")
            brief = self._build_lite_brief(
                structural_context, 
                group_summaries, 
                repo_url, 
                commit_sha, 
                process_stats.get("selected_files", []),
                structural_context.get("project_types", []),
                error=str(e)
            )

        await self._cache_final_brief(repo_url, commit_sha, brief)

        print(f"\n{'='*60}")
        print("Analysis complete.")
        print(f"{'='*60}\n")
        return brief

    async def _cache_final_brief(self, repo_url: str, commit_sha: str, brief: RepoBrief) -> None:
        """Persist the final brief so future requests for the same commit are instant."""
        async with async_session() as session:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
            if repo_obj and repo_obj.commit_sha == commit_sha:
                repo_obj.brief = brief.model_dump()
                await session.commit()

    # ─────────────────────────────────────────────────────────
    # Pass 1: Structural map + full critical file reads
    # ─────────────────────────────────────────────────────────

    async def _build_structural_context(
        self,
        repo_url: str,
        tree_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch file tree, classify files, read critical/entry/config files in full. No LLM.
        Accepts optional pre-fetched tree_data from ProcessingPipeline to avoid duplicate API call.
        """
        enriched_data = await self.github.get_repo_enriched_metadata(repo_url)
        meta = enriched_data["meta"]
        branch = meta["default_branch"]

        # Reuse tree_data if provided (already fetched by ProcessingPipeline)
        if tree_data is None:
            print("  [Pass 1] Fetching file tree (not cached from pipeline)...")
            tree_data = await self.github.get_file_tree_structure(repo_url, branch)
        else:
            print("  [Pass 1] Reusing file tree from pipeline (no duplicate fetch).")

        classification = tree_data["classification"]
        dir_languages = tree_data["directory_languages"]
        nested_tree = tree_data["nested_tree"]
        project_types = tree_data.get("detected_project_types", [])

        print(f"  Files classified: "
              f"critical={len(classification['critical'])}, "
              f"entry_points={len(classification['entry_points'])}, "
              f"config={len(classification['config'])}, "
              f"source={len(classification['source'])}, "
              f"docs={len(classification['docs'])}")
        if project_types:
            print(f"  Detected project types: {project_types}")

        files_to_read = (
            classification["critical"][:10] +
            classification["entry_points"][:15] +
            classification["config"][:10] +
            classification["ci_cd"][:3] +
            classification["docs"][:3]
        )

        print(f"  Reading {len(files_to_read)} key files in full...")
        full_file_contents = await self.github.get_files_content_batch(
            repo_url,
            files_to_read,
            branch=branch,
            max_chars_per_file=MAX_CRITICAL_FILE_CHARS,
        )
        print(f"  Successfully read {len(full_file_contents)} files")

        # Parse manifests
        parsed_manifests = {}
        for filename in ["package.json", "requirements.txt", "pyproject.toml", "go.mod", "pom.xml"]:
            matching = [p for p in classification["critical"] if p.endswith(filename)]
            for path in matching[:1]:
                content = full_file_contents.get(path, "")
                if content:
                    if filename == "package.json":
                        parsed_manifests[path] = self.repo_map_builder.parse_package_json(content)
                    elif filename == "requirements.txt":
                        parsed_manifests[path] = self.repo_map_builder.parse_requirements_txt(content)
                    elif filename == "pyproject.toml":
                        parsed_manifests[path] = self.repo_map_builder.parse_pyproject_toml(content)
                    elif filename == "go.mod":
                        parsed_manifests[path] = self.repo_map_builder.parse_go_mod(content)
                    elif filename == "pom.xml":
                        parsed_manifests[path] = self.repo_map_builder.parse_pom_xml(content)

        detected_frameworks = self.repo_map_builder.detect_frameworks(classification, parsed_manifests)
        print(f"  Detected frameworks: {list(detected_frameworks.keys())[:8]}")

        ascii_tree = self.repo_map_builder.build_ascii_tree(nested_tree, max_depth=4)
        inferred_components = self.repo_map_builder.build_component_names(classification, dir_languages)
        manifest_text = self.repo_map_builder.format_dependencies_for_llm(parsed_manifests)

        # Full repo_metadata — all GitHub fields surfaced to the response
        repo_metadata = {
            # Core identity
            "full_name": meta.get("full_name", ""),
            "description": meta.get("description", ""),
            "homepage": meta.get("homepage", ""),
            "topics": meta.get("topics", []),
            # Popularity
            "stars": meta.get("stars", 0),
            "watchers": meta.get("watchers", 0),
            "forks": meta.get("forks", 0),
            "open_issues": meta.get("open_issues", 0),
            "network_count": meta.get("network_count", 0),
            # Provenance
            "license": meta.get("license", "None"),
            "visibility": meta.get("visibility", "public"),
            "archived": meta.get("archived", False),
            "is_fork": meta.get("is_fork", False),
            "parent_repo": meta.get("parent_repo"),
            "has_wiki": meta.get("has_wiki", False),
            "has_discussions": meta.get("has_discussions", False),
            # Dates
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "pushed_at": meta.get("pushed_at", ""),
            # Size
            "size_kb": meta.get("size_kb", 0),
            "primary_language": meta.get("language", ""),
            # Enriched
            "contributors_count": enriched_data.get("contributors_count", 0),
            "latest_release": enriched_data.get("latest_release", "None"),
        }

        return {
            "meta": meta,
            "language_breakdown": enriched_data["language_breakdown"],
            "repo_metadata": repo_metadata,
            "project_types": project_types,
            "branch": branch,
            "classification": classification,
            "dir_languages": dir_languages,
            "ascii_tree": ascii_tree,
            "full_file_contents": full_file_contents,
            "parsed_manifests": parsed_manifests,
            "detected_frameworks": detected_frameworks,
            "manifest_text": manifest_text,
            "inferred_components": inferred_components,
        }

    # ─────────────────────────────────────────────────────────
    # Pass 2: Targeted similarity search
    # ─────────────────────────────────────────────────────────

    async def _build_search_context(
        self,
        repo_url: str,
        detected_frameworks: Dict[str, str],
        classification: Dict[str, List[str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run 10+ targeted similarity searches to surface architecture patterns."""
        base_queries = [
            "API route handlers and HTTP endpoints",
            "database models, schemas, and ORM definitions",
            "authentication and authorization middleware",
            "main application setup and initialization",
            "error handling and input validation",
            "external API integrations and third-party services",
            "state management and data store configuration",
            "test configuration and test patterns",
            "environment configuration and secrets handling",
            "data processing pipeline and business logic",
        ]

        if "Next.js" in detected_frameworks:
            base_queries += ["Next.js pages and App Router layouts", "Next.js API routes and server actions", "Next.js middleware configuration"]
        if "React" in detected_frameworks or "React DOM" in detected_frameworks:
            base_queries += ["React components and hooks"]
        if "Redux" in detected_frameworks or "React Redux" in detected_frameworks:
            base_queries += ["Redux store, reducers and actions"]
        if any(k in detected_frameworks for k in ["Mongoose ODM", "Prisma ORM", "Drizzle ORM", "SQLAlchemy"]):
            base_queries += ["database schema models and migrations"]
        if any(k in detected_frameworks for k in ["Mongoose ODM", "PyMongo", "Motor (async MongoDB)"]):
            base_queries += [
                "MongoDB Mongoose connection setup database client models schemas",
                "MongoClient mongoose.connect pymongo motor database usage",
            ]
        if any(k in detected_frameworks for k in ["Redis"]):
            base_queries += ["Redis client cache queue session usage"]
        if any(k in detected_frameworks for k in ["psycopg2", "asyncpg", "pgvector"]):
            base_queries += ["PostgreSQL database client queries migrations connection"]
        if "FastAPI" in detected_frameworks:
            base_queries += ["FastAPI routers and dependency injection", "Pydantic request and response models"]
        if "Django" in detected_frameworks:
            base_queries += ["Django views, models and URL patterns"]
        if "Docker" in detected_frameworks:
            base_queries += ["Docker configuration and service orchestration"]
        if "GraphQL" in detected_frameworks:
            base_queries += ["GraphQL schema types and resolvers"]

        print(f"  Running {len(base_queries)} targeted queries...")

        try:
            batch_results = await self.pipeline.search_chunks_by_queries(
                base_queries,
                repo_url,
                top_k=CHUNKS_PER_QUERY,
            )
            search_results = list(batch_results.items())
        except Exception as e:
            print(f"  Batched search failed: {e}")
            search_results = [(query, []) for query in base_queries]

        seen_ids = set()
        context_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for query, results in search_results:
            unique_results = [r for r in results if r["id"] not in seen_ids]
            for r in unique_results:
                seen_ids.add(r["id"])
            if unique_results:
                context_by_topic[query] = unique_results

        total = sum(len(v) for v in context_by_topic.values())
        print(f"  Retrieved {total} unique relevant chunks across all queries")
        return context_by_topic

    # ─────────────────────────────────────────────────────────
    # Pass 3: Parallel file-group summaries (map-reduce) + caching
    # ─────────────────────────────────────────────────────────

    async def _generate_group_summaries(
        self,
        structural_context: Dict[str, Any],
        search_context: Dict[str, List[Dict[str, Any]]],
        repo_url: str,
        commit_sha: str,
    ) -> List[str]:
        """
        Generate focused summaries per architectural area in parallel (Groq).
        Caches results in DB keyed by commit_sha — skips LLM on subsequent runs.
        """
        # Check Phase 3 cache
        async with async_session() as session:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
            if (repo_obj
                    and repo_obj.group_summaries_sha == commit_sha
                    and repo_obj.group_summaries):
                cached = repo_obj.group_summaries
                if isinstance(cached, list) and cached:
                    print(f"  [Phase 3] Cache hit — skipping Groq calls ({len(cached)} summaries)")
                    return cached

        # Build groups from file contents + search results
        full_file_contents = structural_context["full_file_contents"]
        groups: Dict[str, List[str]] = {}

        for path, content in full_file_contents.items():
            parts = path.split("/")
            group_key = parts[0] if len(parts) > 1 else "_root"
            groups.setdefault(group_key, [])
            groups[group_key].append(f"--- {path} ---\n{content[:3000]}")

        for topic, chunks in search_context.items():
            for chunk in chunks:
                path = chunk["metadata"].get("path", "unknown")
                parts = path.split("/")
                group_key = f"_search_{parts[0]}" if len(parts) > 1 else "_search_root"
                groups.setdefault(group_key, [])
                groups[group_key].append(
                    f"--- {path} ({chunk['metadata'].get('type', 'snippet')}) ---\n{chunk['text'][:1500]}"
                )

        sorted_groups = sorted(groups.items(), key=lambda x: len("\n".join(x[1])), reverse=True)
        top_groups = sorted_groups[:MAX_SOURCE_GROUPS]

        print(f"  Summarizing {len(top_groups)} file groups "
              f"(NVIDIA NIM, concurrency:{settings.phase3_concurrency})...")
        sem = asyncio.Semaphore(settings.phase3_concurrency)

        async def summarize_group(group_key: str, file_texts: List[str], index: int) -> Tuple[str, Optional[int], Optional[int]]:
            await asyncio.sleep(index * 0.3)  # stagger startup
            context = "\n\n".join(file_texts)[:MAX_GROUP_CONTEXT_CHARS]

            messages = [
                SystemMessage(content=(
                    "You are a senior software architect. Analyze this code from a repository. "
                    "Produce a tight technical summary (2-3 sentences, max 150 words) covering: "
                    "1) What this code/directory does "
                    "2) Key exports, APIs, or interfaces exposed (name actual functions/routes) "
                    "3) Critical dependencies or integrations. "
                    "Be specific — name actual files and functions. Skip obvious observations."
                )),
                HumanMessage(content=f"Directory/Group: {group_key}\n\n{context}"),
            ]

            async with sem:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = await self.fast_llm.ainvoke(messages)
                        # Log token usage if available
                        usage = getattr(result, "response_metadata", {}).get("token_usage", {})
                        prompt_tok = usage.get("prompt_tokens", "?")
                        comp_tok = usage.get("completion_tokens", "?")
                        print(f"    [Pass 3 - Sub-step] Group '{group_key}' completed ({prompt_tok}p/{comp_tok}c tokens)")
                        return f"[{group_key}]\n{result.content}", prompt_tok, comp_tok
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg and attempt < max_retries - 1:
                            wait_time = 3 ** (attempt + 1)  # 3s, 9s, 27s
                            print(f"  [429 Rate Limit] Group '{group_key}' (NVIDIA NIM) retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"  Group summary failed for '{group_key}': {e}")
                            return f"[{group_key}] (summary failed: {e})", None, None
                return f"[{group_key}] (summary failed: max retries)", None, None

        raw_results = await asyncio.gather(
            *(summarize_group(key, texts, idx) for idx, (key, texts) in enumerate(top_groups))
        )
        summaries = [r[0] for r in raw_results]
        total_prompt = sum(r[1] for r in raw_results if r[1] is not None)
        total_comp = sum(r[2] for r in raw_results if r[2] is not None)
        print(f"  Generated {len(summaries)} group summaries | "
              f"NVIDIA NIM tokens: ~{total_prompt}p / ~{total_comp}c")

        # Persist to DB cache
        async with async_session() as session:
            stmt = select(Repository).where(Repository.repo_url == repo_url)
            result = await session.execute(stmt)
            repo_obj = result.scalar_one_or_none()
            if repo_obj:
                repo_obj.group_summaries = summaries
                repo_obj.group_summaries_sha = commit_sha
                await session.commit()
                print(f"  [Phase 3] Cached group summaries for commit {commit_sha[:8]}")

        return summaries

    async def _query_graph_data(self, repo_url: str, entry_points: List[str]) -> Dict[str, str]:
        """Pass 3.5: Query structural graph for data flow, unused files, and architecture verdict."""
        from app.models.schema import GraphNode, GraphEdge, Repository
        from collections import defaultdict
        
        async with async_session() as session:
            stmt_repo = select(Repository).where(Repository.repo_url == repo_url)
            repo = (await session.execute(stmt_repo)).scalar_one_or_none()
            if not repo:
                return {"data_flow": "", "unused_deps": "", "arch_verdict": ""}
                
            nodes_res = await session.execute(select(GraphNode).where(GraphNode.repository_id == repo.id))
            edges_res = await session.execute(select(GraphEdge).where(GraphEdge.repository_id == repo.id))
            
            nodes = nodes_res.scalars().all()
            edges = edges_res.scalars().all()
            
        if not nodes:
            return {"data_flow": "No graph data available.", "unused_deps": "No graph data available.", "arch_verdict": "No graph data available."}
            
        node_by_id = {n.id: n for n in nodes}
        node_by_path = {n.path: n for n in nodes}
        
        # Build adjacency lists for 'imports'
        adj_imports = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Build undirected adjacency for connected components
        adj_undirected = defaultdict(list)
        
        for e in edges:
            if e.type == "imports":
                adj_imports[e.source_node_id].append(e.target_node_id)
                in_degree[e.target_node_id] += 1
                adj_undirected[e.source_node_id].append(e.target_node_id)
                adj_undirected[e.target_node_id].append(e.source_node_id)
                
        # 1. TRACED DATA FLOW
        data_flow_lines = []
        for ep in entry_points:
            ep_node = node_by_path.get(ep)
            if ep_node:
                visited = {ep_node.id}
                queue = [(ep_node.id, [ep_node.path])]
                traces = []
                while queue and len(traces) < 10:
                    curr_id, path_chain = queue.pop(0)
                    if len(path_chain) >= 4:
                        traces.append(" -> ".join(path_chain))
                        continue
                        
                    children = adj_imports.get(curr_id, [])
                    if not children:
                        traces.append(" -> ".join(path_chain))
                    else:
                        for child_id in children[:3]:
                            if child_id not in visited:
                                visited.add(child_id)
                                child_path = node_by_id[child_id].path
                                queue.append((child_id, path_chain + [child_path]))
                if traces:
                    data_flow_lines.append(f"Entry: {ep}\n  " + "\n  ".join(traces))
                    
        data_flow_text = "\n\n".join(data_flow_lines) if data_flow_lines else "No import traces found from entry points."
        
        # 2. DEPENDENCY USAGE (Unused local files)
        unused_files = []
        entry_paths_set = set(entry_points)
        for n in nodes:
            if n.type == "file" and n.path not in entry_paths_set:
                if in_degree[n.id] == 0:
                    unused_files.append(n.path)
                    
        if unused_files:
            unused_deps_text = "The following local files have NO incoming import edges (they might be unused or dynamically imported):\n"
            unused_deps_text += ", ".join(sorted(unused_files)[:50])
            if len(unused_files) > 50:
                unused_deps_text += f" (+{len(unused_files)-50} more)"
        else:
            unused_deps_text = "All local files have incoming import edges."
            
        # 3. ARCHITECTURE VERDICT (Connected Components)
        visited_cc = set()
        components_count = 0
        file_nodes = [n for n in nodes if n.type == "file"]
        
        for n in file_nodes:
            if n.id not in visited_cc:
                components_count += 1
                q = [n.id]
                visited_cc.add(n.id)
                while q:
                    curr = q.pop(0)
                    for neighbor in adj_undirected.get(curr, []):
                        if neighbor not in visited_cc:
                            visited_cc.add(neighbor)
                            q.append(neighbor)
                            
        arch_verdict = f"The import graph contains {components_count} isolated connected component(s). "
        if components_count == 1:
            arch_verdict += "This strongly implies a monolithic or single tightly-coupled architecture."
        else:
            arch_verdict += "This implies multiple disconnected modules or microservices."
            
        return {
            "data_flow": data_flow_text,
            "unused_deps": unused_deps_text,
            "arch_verdict": arch_verdict
        }

    # ─────────────────────────────────────────────────────────
    # Pass 4: Final synthesis → RepoBrief
    # ─────────────────────────────────────────────────────────

    async def _synthesize_brief(
        self,
        repo_url: str,
        commit_sha: str,
        structural_context: Dict[str, Any],
        search_context: Dict[str, List[Dict[str, Any]]],
        group_summaries: List[str],
        dep_usage: Dict[str, Dict[str, Any]],
        env_vars: Dict[str, List[str]],
        tech_evidence: Dict[str, List[str]],
        graph_data: Dict[str, str],
        selected_files: List[str] = None,
        project_types: List[str] = None,
    ) -> RepoBrief:
        """
        Combine all context and produce the final RepoBrief.
        Injects: metadata, ASCII tree, manifests, frameworks, group summaries,
        similarity search (with real line refs), dep usage, env vars (all ground truth).
        """
        meta = structural_context["meta"]
        language_breakdown = structural_context["language_breakdown"]
        repo_metadata = structural_context["repo_metadata"]
        ascii_tree = structural_context["ascii_tree"]
        manifest_text = structural_context["manifest_text"]
        detected_frameworks = structural_context["detected_frameworks"]
        full_file_contents = structural_context["full_file_contents"]
        inferred_components = structural_context["inferred_components"]
        project_types = project_types or structural_context.get("project_types", [])

        critical_files_text = self._format_critical_files(
            structural_context["classification"],
            full_file_contents,
        )
        search_text = self._format_search_context(search_context)
        summaries_text = "\n\n".join(group_summaries)
        frameworks_text = "\n".join(f"  - {fw}: {ver}" for fw, ver in detected_frameworks.items())
        components_text = "\n".join(
            f"  - /{c['name']}: {c['estimated_role']} [{c['primary_language']}]"
            for c in inferred_components
        )
        
        # Format language breakdown
        lang_text = "\n".join(f"  - {lang}: {pct}%" for lang, pct in language_breakdown.items())
        
        # Format repo metadata
        meta_text = "\n".join(f"  - {k}: {v}" for k, v in repo_metadata.items())
        
        dep_usage_text = _format_dep_usage(dep_usage)
        env_vars_text = _format_env_vars(env_vars)
        tech_evidence_text = _format_tech_evidence(tech_evidence)
        project_types_text = ", ".join(project_types) if project_types else "Not detected"

        system_prompt = """You are an expert software architect producing a comprehensive repository brief.
You have been given:
1. Repository metadata and directory structure
2. Full contents of key files (README, manifests, configs, entry points)
3. Auto-detected frameworks and dependencies
4. Per-component summaries (from actual code analysis)
5. Targeted code search results with REAL file:line references
6. STATIC ANALYSIS: dependency usage (declared vs. actually imported in code)
7. STATIC ANALYSIS: positive technology evidence found across all selected chunks
8. STATIC ANALYSIS: env vars actually read from code (not docs)

Your task: produce a HIGHLY ACCURATE RepoBrief that could let someone draw an architecture diagram.

STRICT RULES:
- API endpoints: report HTTP methods ONLY from actual route handler code (@app.get, router.post, app.use). NEVER infer from folder names.
- Architecture type: Treat the ARCHITECTURE VERDICT as verified ground truth. Do not label as 'microservices' if the verdict says there is only 1 connected component.
- File citations: for every claim in architecture_summary and data_flow, cite at least one real file path using the references provided. Use format [filename.ext:Lstart-Lend]. Do NOT invent line numbers — use only what's in the context.
- Code vs docs: flag a README/comment contradiction ONLY when there is positive code evidence that directly contradicts it. Missing evidence, sparse retrieval, or "not observed in selected code" is not a contradiction.
- Data flow: Treat the TRACED DATA FLOW as verified ground truth. Trace data flow using the exact chain provided. Do not narrate from memory.
- Dependencies: Treat the DEPENDENCY USAGE and STATIC ANALYSIS sections as verified ground truth. "NOT OBSERVED in selected analyzed code" is uncertainty, not an issue. Never put a dependency into flagged_issues merely because it was not observed.
- Env vars: use the STATIC ANALYSIS section. Only report vars the CODE actually reads.
- ALL fields REQUIRED with substantive content when evidence exists. Empty lists are acceptable only when the context has no supporting evidence.
- EVIDENCE FIRST: Do not infer specific technologies, endpoints, databases, services, deployment topology, auth, or data stores from framework conventions alone. If a detail is not present in metadata, manifests, file contents, summaries, or search results, write "Not evidenced in the analyzed files" instead of inventing it.
- FLAGGED ISSUES: Use a high-precision policy. Include an issue only if at least one analyzed file provides positive evidence of the problem and no positive evidence section contradicts it. Do not flag "README says MongoDB but code does not use MongoDB" when MongoDB/Mongoose/PyMongo/Motor evidence appears anywhere in positive technology evidence, dependency usage, full files, summaries, or search results. If evidence is incomplete, leave flagged_issues empty.
- JSON SCHEMA COMPLIANCE: You must strictly adhere to the provided schema. Do NOT output a blank or empty JSON skeleton. EVERY string field must contain a detailed description (minimum 2 sentences). EVERY list field must contain all relevant items found in the context.
- NO MARKDOWN REPORTS: You must return the results SOLELY by populating the provided JSON/function schema. Do NOT write a markdown report or table. ALL information must be placed inside the schema fields.
- LIST POPULATION MANDATE: You MUST populate arrays correctly. Do NOT cram component or tech stack lists into the `summary` or `architecture_summary` strings. Use the proper `components` and `tech_stack` arrays.
- JSON TEMPLATE EXPECTATION: Your JSON must strictly mirror this structure:
```json
{
  "tech_stack": [{"name": "React", "version": "18", "category": "framework", "purpose": "UI"}],
  "architecture_summary": "Detailed string here...",
  "entry_points": [{"file_path": "main.py", "description": "Entry"}],
  "flagged_issues": ["Issue 1"],
  "summary": "High-level summary here...",
  "components": [{"name": "backend", "directory": "api/", "description": "...", "key_files": ["api/routes.py"], "depends_on": [], "technologies": ["FastAPI"]}],
  "data_flow": "String describing flow...",
  "detected_patterns": ["MVC"],
  "dependency_graph": {"backend": ["database"]}
}
```
- LANGUAGE BREAKDOWN is from GitHub's byte-count analysis — treat as GROUND TRUTH for tech_stack. Every language listed MUST appear in tech_stack. Do not omit languages visible in the breakdown.
- DIRECTORY STRUCTURE is complete — use it to identify ALL components. Every top-level directory with source code is a potential component.
- key_files: for each component, list files from the directory structure that appear architecturally significant. If fewer than 2 files are evidenced, include only the evidenced files.
- dependency_graph: include edges supported by imports, route wiring, service calls, configs, or explicit summaries. Do not add dependency edges from directory names alone."""

        user_prompt = f"""Repository: {repo_url}
Commit: {commit_sha}
GitHub metadata:
  - Primary language: {meta.get('language', 'Unknown')}
  - Stars: {meta.get('stars', 0)}
  - Description: {meta.get('description', 'No description')}
  - Topics: {', '.join(meta.get('topics', []))}

=== DETECTED PROJECT TYPES (from file tree markers — treat as ground truth) ===
{project_types_text}

=== GITHUB REPOSITORY METADATA (verified) ===
{meta_text}

=== LANGUAGE BREAKDOWN BY DIRECTORY (ground truth from GitHub API) ===
{lang_text}

=== DIRECTORY STRUCTURE ===
{ascii_tree[:12000]}

=== AUTO-DETECTED FRAMEWORKS & DEPENDENCIES ===
{frameworks_text}

=== INFERRED COMPONENTS (from directory structure) ===
{components_text}

=== MANIFEST / DEPENDENCY DETAILS ===
{manifest_text[:6000]}

=== STATIC ANALYSIS: DEPENDENCY USAGE (declared vs. actually imported in code) ===
{dep_usage_text}

=== STATIC ANALYSIS: POSITIVE TECHNOLOGY EVIDENCE ACROSS SELECTED CHUNKS ===
{tech_evidence_text}

=== STATIC ANALYSIS: ENVIRONMENT VARIABLES (read from code, not docs) ===
{env_vars_text}

=== TRACED DATA FLOW (from graph) ===
{graph_data.get('data_flow', '')}

=== DEPENDENCY USAGE (from graph) ===
{graph_data.get('unused_deps', '')}

=== ARCHITECTURE VERDICT (from graph) ===
{graph_data.get('arch_verdict', '')}

=== FULL CRITICAL FILES (README, configs, entry points) ===
{critical_files_text[:SYNTHESIS_CRITICAL_FILES_CHARS]}

=== PER-COMPONENT SUMMARIES ===
{summaries_text[:SYNTHESIS_SUMMARIES_CHARS]}

=== TARGETED CODE SEARCH RESULTS (with real file:line references) ===
{search_text[:SYNTHESIS_SEARCH_CHARS]}

Now produce the complete RepoBrief with all evidenced fields populated accurately. Prefer "Not evidenced in the analyzed files" over unsupported guesses."""

        parser = PydanticOutputParser(pydantic_object=LLMRepoBrief)
        format_instructions = parser.get_format_instructions()
        system_prompt += f"\n\n{format_instructions}\n\nCRITICAL FORMAT REQUIREMENT:\nYou MUST enclose your final JSON output inside a single markdown code block:\n```json\n{{\n  ...\n}}\n```\nDo NOT add any text outside the code block."

        print(f"  Calling NVIDIA reasoning LLM ({settings.repo_analizer_model}, thinking={settings.llm_enable_thinking})...")
        t_start = time.time()

        res = await self.reasoning_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        
        try:
            brief = parser.invoke(res)
        except Exception as e:
            print(f"  [Pass 4] Parser warning: {e}. Executing robust multi-strategy JSON extraction...")
            try:
                json_data = _extract_json_payload(res.content)
                brief = LLMRepoBrief.model_validate(json_data)
            except Exception as inner_e:
                logger.error(f"  [Pass 4] Robust JSON extraction failed: {inner_e}")
                raise ValueError(f"Failed to parse LLM JSON: {inner_e}")

        duration = time.time() - t_start
        print(f"  [Pass 4] NVIDIA synthesis: {duration:.1f}s")

        # Convert LLMRepoBrief to full RepoBrief, injecting static analysis
        final_brief = RepoBrief(
            **brief.model_dump(),
            directory_structure=ascii_tree,
            framework_details=detected_frameworks,
            language_breakdown=language_breakdown,
            repo_metadata=repo_metadata,
            project_types=project_types,
            selected_files=selected_files or [],
        )

        return self._ground_brief_in_static_context(final_brief, structural_context, dep_usage, tech_evidence)

    def _ground_brief_in_static_context(
        self,
        brief: RepoBrief,
        structural_context: Dict[str, Any],
        dep_usage: Optional[Dict[str, Dict[str, Any]]] = None,
        tech_evidence: Optional[Dict[str, List[str]]] = None,
    ) -> RepoBrief:
        """
        Deterministic guardrails after LLM synthesis.
        Ensures static facts are represented and leaves unknowns explicit instead of
        letting schema pressure become made-up detail.
        """
        language_breakdown = structural_context.get("language_breakdown", {})
        existing_tech = {item.name.lower() for item in brief.tech_stack}
        for language, pct in language_breakdown.items():
            if language.lower() not in existing_tech:
                brief.tech_stack.append(
                    TechItem(
                        name=language,
                        version=None,
                        category="language",
                        purpose=f"GitHub language analysis reports {pct}% of repository bytes.",
                    )
                )

        if not brief.components:
            brief.components = [
                ComponentInfo(
                    name=c["name"],
                    directory=c["directory"],
                    description=c["estimated_role"],
                    key_files=[],
                    depends_on=[],
                    technologies=[c["primary_language"]],
                )
                for c in structural_context.get("inferred_components", [])
            ]

        if not brief.entry_points:
            entry_paths = structural_context.get("classification", {}).get("entry_points", [])
            brief.entry_points = [
                FileReference(
                    file_path=path,
                    description="Detected by file-tree heuristics as a likely entry point; no deeper route evidence was available.",
                )
                for path in entry_paths[:10]
            ]

        if not brief.dependency_graph and brief.components:
            brief.dependency_graph = {component.name: component.depends_on for component in brief.components}

        brief.flagged_issues = self._filter_unsupported_flagged_issues(
            brief.flagged_issues,
            dep_usage or {},
            tech_evidence or structural_context.get("tech_evidence", {}),
        )

        return brief

    def _filter_unsupported_flagged_issues(
        self,
        issues: List[str],
        dep_usage: Dict[str, Dict[str, Any]],
        tech_evidence: Dict[str, List[str]],
    ) -> List[str]:
        """
        High-precision issue gate.
        The model may still over-read "not observed" as "unused"; remove issues
        that are based on absence of evidence or contradicted by positive usage.
        """
        if not issues:
            return []

        observed_deps = {
            dep.lower()
            for dep, data in dep_usage.items()
            if data.get("status") == "observed_used"
        }
        observed_tech = {tech.lower() for tech, refs in tech_evidence.items() if refs}

        protected_terms = {
            "mongodb": {"mongodb", "mongo", "mongoose", "pymongo", "motor"},
            "redis": {"redis"},
            "postgres": {"postgres", "postgresql", "psycopg", "asyncpg", "pgvector"},
            "mysql": {"mysql", "mariadb"},
            "sqlite": {"sqlite"},
        }

        absence_markers = (
            "not used",
            "never used",
            "unused",
            "declared only",
            "not observed",
            "not found",
            "no code",
            "does not use",
            "doesn't use",
            "readme",
            "documentation",
            "docs",
        )

        filtered: List[str] = []
        for issue in issues:
            issue_lower = issue.lower()

            if any(marker in issue_lower for marker in absence_markers):
                contradicted_by_positive_evidence = False
                for canonical, terms in protected_terms.items():
                    if not any(term in issue_lower for term in terms):
                        continue
                    has_tech_evidence = canonical in observed_tech
                    has_dep_evidence = any(
                        any(term in dep for term in terms)
                        for dep in observed_deps
                    )
                    if has_tech_evidence or has_dep_evidence:
                        contradicted_by_positive_evidence = True
                        break

                if contradicted_by_positive_evidence:
                    logger.info("Removed hallucinated flagged_issue contradicted by positive evidence: %s", issue)
                    continue

                # "Not found/unused" claims are too low-confidence for flagged_issues.
                # Keep this field for positive, actionable defects only.
                if any(marker in issue_lower for marker in ("not used", "never used", "unused", "declared only", "not observed", "not found", "no code", "does not use", "doesn't use")):
                    logger.info("Removed low-confidence absence-based flagged_issue: %s", issue)
                    continue

            filtered.append(issue)

        return filtered

    # ─────────────────────────────────────────────────────────
    # Lite brief fallback (Phase 4 failed)
    # ─────────────────────────────────────────────────────────

    def _build_lite_brief(
        self,
        structural_context: Dict[str, Any],
        group_summaries: List[str],
        repo_url: str,
        commit_sha: str,
        selected_files: List[str] = None,
        project_types: List[str] = None,
        error: str = "",
    ) -> RepoBrief:
        """
        Degraded brief assembled from Phase 1 + Phase 3 without any LLM.
        Flagged so consumers know depth is reduced.
        """
        meta = structural_context["meta"]
        detected_frameworks = structural_context["detected_frameworks"]
        inferred_components = structural_context["inferred_components"]
        ascii_tree = structural_context["ascii_tree"]

        components = [
            ComponentInfo(
                name=c["name"],
                directory=c["directory"],
                description=c["estimated_role"],
                key_files=[],
                depends_on=[],
                technologies=[c["primary_language"]],
            )
            for c in inferred_components
        ]

        tech_stack = [
            TechItem(name=fw, version=ver or "detected", category="framework", purpose="detected from config files")
            for fw, ver in detected_frameworks.items()
        ]

        summaries_combined = "\n\n".join(group_summaries) if group_summaries else "No summaries available."

        warning = (
            "⚠️ LITE BRIEF: Phase 4 LLM synthesis was unavailable. "
            "This brief is assembled from static analysis only (Phase 1 + Phase 3). "
            "Architecture details and data flow descriptions may lack depth."
        )
        if error:
            warning += f" Failure reason: {error[:200]}"

        return RepoBrief(
            tech_stack=tech_stack,
            architecture_summary=f"[Lite brief] {meta.get('description', '')} | Component summaries:\n{summaries_combined[:3000]}",
            entry_points=[],
            flagged_issues=[warning],
            summary=f"{meta.get('description', 'No description.')} (Lite brief — reduced detail)",
            components=components,
            data_flow="Phase 4 synthesis unavailable — data flow not traced.",
            detected_patterns=[],
            framework_details=detected_frameworks,
            directory_structure=ascii_tree[:2000],
            dependency_graph={c["name"]: [] for c in inferred_components},
            selected_files=selected_files or [],
            project_types=project_types or [],
        )

    # ─────────────────────────────────────────────────────────
    # Formatting helpers
    # ─────────────────────────────────────────────────────────

    def _format_critical_files(
        self,
        classification: Dict[str, List[str]],
        full_file_contents: Dict[str, str],
    ) -> str:
        """Format full-read files into LLM-readable text, priority files first."""
        sections = []
        priority_order = (
            classification.get("critical", []) +
            classification.get("entry_points", []) +
            classification.get("config", []) +
            classification.get("ci_cd", []) +
            classification.get("docs", [])
        )
        seen = set()
        for path in priority_order:
            if path in seen or path not in full_file_contents:
                continue
            seen.add(path)
            content = full_file_contents[path]
            sections.append(f"=== {path} ({len(content)} chars) ===\n{content}")
        return "\n\n".join(sections)

    def _format_search_context(
        self,
        search_context: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        Format similarity search results grouped by topic.
        Injects REAL file:line references from chunk metadata so the LLM
        copies actual references instead of fabricating line numbers.
        """
        sections = []
        for topic, chunks in search_context.items():
            if not chunks:
                continue
            sections.append(f"[Topic: {topic}]")
            for chunk in chunks:
                path = chunk["metadata"].get("path", "unknown")
                chunk_type = chunk["metadata"].get("type", "snippet")
                node_name = chunk["metadata"].get("node_name", "")
                start_line = chunk["metadata"].get("start_line", "")
                end_line = chunk["metadata"].get("end_line", "")

                # Build real file:line reference from chunk metadata
                if start_line and end_line:
                    line_ref = f":L{start_line}-L{end_line}"
                elif start_line:
                    line_ref = f":L{start_line}"
                else:
                    line_ref = ""

                label = f"{path}{line_ref} > {node_name}" if node_name else f"{path}{line_ref}"
                sections.append(f"  -- {label} ({chunk_type}) --")
                sections.append(f"  {chunk['text'][:800]}")
        return "\n".join(sections)
