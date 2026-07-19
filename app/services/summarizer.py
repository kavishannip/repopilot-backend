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


# ─────────────────────────────────────────────────────────────────────────────
# Static analysis helpers (deterministic — injected as ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_dependency_usage(
    parsed_manifests: Dict[str, Any],
    full_file_contents: Dict[str, str],
    search_context: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, str]:
    """
    Cross-reference manifest deps against actual imports in code.
    Returns {dep_name: "used" | "declared_only"}.

    Import data sources:
    1. full_file_contents — regex scan for import/require/from
    2. search_context chunk metadata — chunker.py already extracts imports
    """
    # Collect all declared deps
    declared: Dict[str, str] = {}  # dep_name -> version
    for manifest_data in parsed_manifests.values():
        declared.update(manifest_data.get("all_dependencies", {}))

    if not declared:
        return {}

    # Collect all imported/required module names from code
    imported_raw: set = set()

    # Regex patterns: Python, JS/TS
    import_patterns = [
        r"(?:from\s+([\w.\-]+)\s+import|import\s+([\w.\-]+))",           # Python
        r"(?:import|from)\s+['\"]([^'\"]+)['\"]",                        # JS/TS static
        r"require\(['\"]([^'\"]+)['\"]\)",                               # CommonJS
    ]

    def extract_module_names(text: str) -> set:
        found = set()
        for pat in import_patterns:
            for m in re.finditer(pat, text):
                for g in m.groups():
                    if g:
                        # Normalize: take first path segment, lowercase
                        root = g.split("/")[0].split(".")[0].lower().lstrip("@")
                        if root and len(root) > 1:
                            found.add(root)
                            # Also add with @ prefix for scoped packages
                            if "/" in g and g.startswith("@"):
                                found.add(g.split("/")[0].lower())
        return found

    for content in full_file_contents.values():
        imported_raw.update(extract_module_names(content))

    # Also pull from chunk metadata (chunker already extracted imports)
    for chunks in search_context.values():
        for chunk in chunks:
            for imp_line in chunk.get("metadata", {}).get("imports", []):
                imported_raw.update(extract_module_names(imp_line))

    # Match declared deps against imported
    result = {}
    for dep, version in declared.items():
        dep_normalized = dep.lower().lstrip("@").split("/")[0].replace("-", "").replace("_", "")
        is_used = any(
            dep_normalized in imp.replace("-", "").replace("_", "")
            or imp in dep.lower()
            for imp in imported_raw
        )
        result[dep] = "used" if is_used else "declared_only"

    return result


def _analyze_env_vars(
    full_file_contents: Dict[str, str],
    search_context: Dict[str, List[Dict[str, Any]]],
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

    return env_vars


def _format_dep_usage(dep_usage: Dict[str, str]) -> str:
    """Format dependency usage diff for LLM prompt injection."""
    if not dep_usage:
        return "No manifest found or no dependencies declared."

    used = sorted(k for k, v in dep_usage.items() if v == "used")
    declared_only = sorted(k for k, v in dep_usage.items() if v == "declared_only")

    lines = []
    if used:
        lines.append(f"ACTUALLY IMPORTED in code ({len(used)}): {', '.join(used[:40])}")
    if declared_only:
        lines.append(f"DECLARED IN MANIFEST only — not seen in imports ({len(declared_only)}): {', '.join(declared_only[:30])}")
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

        # ── Pass 3: Parallel file-group summaries (with caching) ──────
        await set_phase("group_summaries")
        t3 = time.time()
        print("\n[Pass 3] Generating file-group summaries (Groq, map-reduce)...")
        group_summaries = await self._generate_group_summaries(
            structural_context, search_context, repo_url, commit_sha
        )
        print(f"  Done in {time.time()-t3:.1f}s, {len(group_summaries)} summaries")

        # ── Static analysis (inject as ground truth for Pass 4) ──────
        dep_usage = _analyze_dependency_usage(
            structural_context["parsed_manifests"],
            structural_context["full_file_contents"],
            search_context,
        )
        env_vars = _analyze_env_vars(
            structural_context["full_file_contents"],
            search_context,
        )
        logger.info(f"Static analysis: {len(dep_usage)} deps classified, {len(env_vars)} env vars found")

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
                selected_files=process_stats.get("selected_files", []),
                project_types=structural_context.get("project_types", []),
            )
            print(f"  Done in {time.time()-t4:.1f}s")
        except Exception as e:
            print(f"  [Pass 4] FAILED: {e}. Serving lite brief fallback.")
            brief = self._build_lite_brief(
                structural_context, group_summaries, repo_url, commit_sha, error=str(e)
            )

        print(f"\n{'='*60}")
        print("Analysis complete.")
        print(f"{'='*60}\n")
        return brief

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
        if "FastAPI" in detected_frameworks:
            base_queries += ["FastAPI routers and dependency injection", "Pydantic request and response models"]
        if "Django" in detected_frameworks:
            base_queries += ["Django views, models and URL patterns"]
        if "Docker" in detected_frameworks:
            base_queries += ["Docker configuration and service orchestration"]
        if "GraphQL" in detected_frameworks:
            base_queries += ["GraphQL schema types and resolvers"]

        print(f"  Running {len(base_queries)} targeted queries...")

        async def search(query: str):
            try:
                results = await self.pipeline.search_chunks(query, repo_url, top_k=CHUNKS_PER_QUERY)
                return query, results
            except Exception as e:
                print(f"  Search failed for '{query}': {e}")
                return query, []

        search_results = await asyncio.gather(*(search(q) for q in base_queries))

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
              f"(Groq:{settings.groq_model}, concurrency:{settings.phase3_concurrency})...")
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
                        logger.debug(f"  [Phase 3] Group '{group_key}': {prompt_tok}p/{comp_tok}c tokens")
                        return f"[{group_key}]\n{result.content}", prompt_tok, comp_tok
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg and attempt < max_retries - 1:
                            wait_time = 3 ** (attempt + 1)  # 3s, 9s, 27s
                            print(f"  [429 Rate Limit] Group '{group_key}' (Groq) retrying in {wait_time}s...")
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
              f"Groq tokens: ~{total_prompt}p / ~{total_comp}c")

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
        dep_usage: Dict[str, str],
        env_vars: Dict[str, List[str]],
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
        project_types_text = ", ".join(project_types) if project_types else "Not detected"

        system_prompt = """You are an expert software architect producing a comprehensive repository brief.
You have been given:
1. Repository metadata and directory structure
2. Full contents of key files (README, manifests, configs, entry points)
3. Auto-detected frameworks and dependencies
4. Per-component summaries (from actual code analysis)
5. Targeted code search results with REAL file:line references
6. STATIC ANALYSIS: dependency usage (declared vs. actually imported in code)
7. STATIC ANALYSIS: env vars actually read from code (not docs)

Your task: produce a HIGHLY ACCURATE RepoBrief that could let someone draw an architecture diagram.

STRICT RULES:
- API endpoints: report HTTP methods ONLY from actual route handler code (@app.get, router.post, app.use). NEVER infer from folder names.
- Architecture type: label as 'microservices' ONLY if multiple separately deployable services with distinct entry points/Dockerfiles/ports. Multiple modules = 'modular monolith'.
- File citations: for every claim in architecture_summary and data_flow, cite at least one real file path using the references provided. Use format [filename.ext:Lstart-Lend]. Do NOT invent line numbers — use only what's in the context.
- Code vs docs: when README/comments contradict code, treat CODE as ground truth. Flag discrepancy in flagged_issues.
- Data flow: trace through ACTUAL function names and imports you see. Example: "Request hits router.post('/analyze') in endpoints.py → calls summarizer.generate_brief() → ..."
- Dependencies: use the STATIC ANALYSIS section for declared vs. used. Do not guess from manifest alone.
- Env vars: use the STATIC ANALYSIS section. Only report vars the CODE actually reads.
- ALL fields REQUIRED with substantive content. Empty lists unacceptable when evidence exists.
- SYNTHESIZE CONFIDENTLY: The provided context contains everything you need. Synthesize the snippets, summaries, and structural maps to comprehensively fill out EVERY field. NEVER leave a field empty or output an empty list/object. If a specific detail is missing, infer the most likely architectural pattern based on standard practices for the detected frameworks.
- If highly uncertain about a specific detail, state your assumption in flagged_issues rather than leaving the field blank.
- JSON SCHEMA COMPLIANCE: You must strictly adhere to the provided schema. Do NOT output a blank or empty JSON skeleton. EVERY string field must contain a detailed description (minimum 2 sentences). EVERY list field must contain all relevant items found in the context.
- NO MARKDOWN REPORTS: You must return the results SOLELY by populating the provided JSON/function schema. Do NOT write a markdown report or table. ALL information must be placed inside the schema fields.
- LANGUAGE BREAKDOWN is from GitHub's byte-count analysis — treat as GROUND TRUTH for tech_stack. Every language listed MUST appear in tech_stack. Do not omit languages visible in the breakdown.
- DIRECTORY STRUCTURE is complete — use it to identify ALL components. Every top-level directory with source code is a potential component.
- key_files: for each component, list at least 2-3 files from the directory structure that appear architecturally significant (entry points, configs, main files).
- dependency_graph: infer edges from import analysis AND from architectural relationships visible in the directory structure and component descriptions."""

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

=== STATIC ANALYSIS: ENVIRONMENT VARIABLES (read from code, not docs) ===
{env_vars_text}

=== FULL CRITICAL FILES (README, configs, entry points) ===
{critical_files_text[:SYNTHESIS_CRITICAL_FILES_CHARS]}

=== PER-COMPONENT SUMMARIES ===
{summaries_text[:SYNTHESIS_SUMMARIES_CHARS]}

=== TARGETED CODE SEARCH RESULTS (with real file:line references) ===
{search_text[:SYNTHESIS_SEARCH_CHARS]}

Now produce the complete RepoBrief with ALL fields populated accurately."""

        parser = PydanticOutputParser(pydantic_object=LLMRepoBrief)
        format_instructions = parser.get_format_instructions()
        
        system_prompt += f"\n\n{format_instructions}"

        print(f"  Calling NVIDIA reasoning LLM ({settings.repo_analizer_model}, thinking={settings.llm_enable_thinking})...")
        t_start = time.time()

        res = await self.reasoning_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        
        try:
            brief = parser.invoke(res)
        except Exception as e:
            # Fallback if parser fails due to cut-off or weird format
            print(f"  [Pass 4] Parse warning: {e}. Trying raw json extraction...")
            import json
            import re
            match = re.search(r'\{.*\}', res.content, re.DOTALL)
            if match:
                try:
                    brief = LLMRepoBrief.model_validate_json(match.group(0))
                except Exception as inner_e:
                    raise ValueError(f"Failed to parse LLM JSON: {inner_e}")
            else:
                raise ValueError("No JSON object found in LLM output")

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

        return final_brief

    # ─────────────────────────────────────────────────────────
    # Lite brief fallback (Phase 4 failed)
    # ─────────────────────────────────────────────────────────

    def _build_lite_brief(
        self,
        structural_context: Dict[str, Any],
        group_summaries: List[str],
        repo_url: str,
        commit_sha: str,
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
