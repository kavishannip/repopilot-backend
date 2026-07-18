"""
Multi-pass repository summarizer.

Pass 1: Read critical files in full + build structural map (no LLM)
Pass 2: Targeted similarity search (no LLM)
Pass 3: Parallel file-group summaries — fast LLM (thinking OFF, 2048 tokens)
Pass 4: Final synthesis — reasoning LLM (thinking ON, 16384 tokens, 1M context window)

nvidia/nemotron-3-super-120b-a12b advantages exploited:
  - 1M token context window: feed full critical files + all group summaries without truncation
  - Reasoning mode (CoT): enabled on final pass for architecture-level reasoning
  - RAG-optimized: model tuned for exactly this retrieval-augmented generation pattern
  - LatentMoE: 12B active params — faster than 120B dense at inference
"""

import asyncio
import json
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from app.models.brief import RepoBrief, ComponentInfo, TechItem, FileReference
from app.services.llm import get_fast_llm, get_reasoning_llm
from app.services.pipeline import ProcessingPipeline
from app.services.repo_map import RepoMap
from app.services.github_client import GitHubClient
from app.config import settings


# Max chars to read per individual critical/entry-point file
MAX_CRITICAL_FILE_CHARS = settings.max_file_content_chars

# How many source chunks to retrieve per targeted query
CHUNKS_PER_QUERY = 5

# Max total context chars fed to each group summary call (fast LLM)
MAX_GROUP_CONTEXT_CHARS = 15_000

# Parallel group summary buckets
MAX_SOURCE_GROUPS = 10

# Context caps for final synthesis — generous thanks to 1M context window
# (chars, not tokens; rough 1 token ≈ 4 chars)
SYNTHESIS_CRITICAL_FILES_CHARS = 30_000   # was 10k
SYNTHESIS_SUMMARIES_CHARS      = 15_000   # was 4k
SYNTHESIS_SEARCH_CHARS         = 15_000   # was 5k


class RepoSummarizer:
    def __init__(self):
        # Fast LLM: thinking OFF, 2048 output tokens — for parallel group summaries
        self.fast_llm = get_fast_llm()
        # Reasoning LLM: thinking ON (if enabled), 16384 output tokens — for final synthesis
        self.reasoning_llm = get_reasoning_llm()
        self.reasoning_llm_structured = get_reasoning_llm().with_structured_output(RepoBrief)
        self.pipeline = ProcessingPipeline()
        self.repo_map_builder = RepoMap()
        self.github = GitHubClient()

    async def generate_brief(self, repo_url: str) -> RepoBrief:
        """
        Main entry: 4-pass pipeline to generate comprehensive, accurate repo brief.
        """
        print(f"\n{'='*60}")
        print(f"Starting multi-pass analysis for: {repo_url}")
        print(f"{'='*60}")

        # ── Pass 0: Ensure repo is embedded (for similarity search) ──
        print("\n[Pass 0] Embedding repository chunks...")
        process_stats = await self.pipeline.process_repo(repo_url)
        commit_sha = process_stats["commit_sha"]
        print(f"  Status: {process_stats['status']}, chunks: {process_stats.get('chunks_count', '?')}")

        # ── Pass 1: Structural map + full reads of key files ──
        print("\n[Pass 1] Building structural map and reading key files...")
        structural_context = await self._build_structural_context(repo_url)

        # ── Pass 2: Targeted similarity search for architecture patterns ──
        print("\n[Pass 2] Targeted similarity search across architecture topics...")
        search_context = await self._build_search_context(
            repo_url,
            structural_context["detected_frameworks"],
            structural_context["classification"],
        )

        # ── Pass 3: Parallel file-group summaries ──
        print("\n[Pass 3] Generating file-group summaries (map-reduce)...")
        group_summaries = await self._generate_group_summaries(
            structural_context, search_context
        )

        # ── Pass 4: Final synthesis → structured RepoBrief ──
        print("\n[Pass 4] Synthesizing final RepoBrief...")
        brief = await self._synthesize_brief(
            repo_url=repo_url,
            commit_sha=commit_sha,
            structural_context=structural_context,
            search_context=search_context,
            group_summaries=group_summaries,
        )

        print(f"\n{'='*60}")
        print("Analysis complete.")
        print(f"{'='*60}\n")
        return brief

    # ─────────────────────────────────────────────────────────
    # Pass 1: Structural map + full critical file reads
    # ─────────────────────────────────────────────────────────

    async def _build_structural_context(self, repo_url: str) -> Dict[str, Any]:
        """
        Fetch the file tree, classify files, and read critical/entry/config files in full.
        No LLM involved — pure structured data extraction.
        """
        # Get repo metadata
        meta = await self.github.get_repo_metadata(repo_url)
        branch = meta["default_branch"]

        # Build file tree structure
        tree_data = await self.github.get_file_tree_structure(repo_url, branch)
        classification = tree_data["classification"]
        dir_languages = tree_data["directory_languages"]
        nested_tree = tree_data["nested_tree"]

        print(f"  Files classified: "
              f"critical={len(classification['critical'])}, "
              f"entry_points={len(classification['entry_points'])}, "
              f"config={len(classification['config'])}, "
              f"source={len(classification['source'])}, "
              f"docs={len(classification['docs'])}")

        # Files to read in full (ordered by priority)
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
        for filename in ["package.json", "requirements.txt", "pyproject.toml"]:
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

        # Detect frameworks
        detected_frameworks = self.repo_map_builder.detect_frameworks(classification, parsed_manifests)
        print(f"  Detected frameworks: {list(detected_frameworks.keys())[:8]}")

        # Build ASCII tree (top 4 levels)
        ascii_tree = self.repo_map_builder.build_ascii_tree(nested_tree, max_depth=4)

        # Infer components
        inferred_components = self.repo_map_builder.build_component_names(classification, dir_languages)

        # Format manifest info for LLM
        manifest_text = self.repo_map_builder.format_dependencies_for_llm(parsed_manifests)

        return {
            "meta": meta,
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
        """
        Run 10+ targeted similarity searches to surface architecture patterns.
        Queries are adapted based on detected tech stack.
        """
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

        # Adaptive queries based on detected frameworks
        if "Next.js" in detected_frameworks:
            base_queries += [
                "Next.js pages and App Router layouts",
                "Next.js API routes and server actions",
                "Next.js middleware configuration",
            ]
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

        # Run all queries concurrently
        async def search(query: str):
            try:
                results = await self.pipeline.search_chunks(query, repo_url, top_k=CHUNKS_PER_QUERY)
                return query, results
            except Exception as e:
                print(f"  Search failed for '{query}': {e}")
                return query, []

        search_results = await asyncio.gather(*(search(q) for q in base_queries))

        # Deduplicate by chunk ID
        seen_ids = set()
        # Track which already-read files we have in full_file_contents so we skip redundant chunks
        context_by_topic: Dict[str, List[Dict[str, Any]]] = {}

        for query, results in search_results:
            unique_results = []
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)
            if unique_results:
                context_by_topic[query] = unique_results

        total = sum(len(v) for v in context_by_topic.values())
        print(f"  Retrieved {total} unique relevant chunks across all queries")

        return context_by_topic

    # ─────────────────────────────────────────────────────────
    # Pass 3: Parallel file-group summaries (map-reduce)
    # ─────────────────────────────────────────────────────────

    async def _generate_group_summaries(
        self,
        structural_context: Dict[str, Any],
        search_context: Dict[str, List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate focused summaries per architectural area in parallel.
        Each summary captures: what this area does, key patterns, key exports/APIs.
        """
        full_file_contents = structural_context["full_file_contents"]
        inferred_components = structural_context["inferred_components"]

        # Group files by top-level directory / component
        groups: Dict[str, List[str]] = {}

        # Group the directly-read files
        for path, content in full_file_contents.items():
            parts = path.split("/")
            group_key = parts[0] if len(parts) > 1 else "_root"
            groups.setdefault(group_key, [])
            groups[group_key].append(f"--- {path} ---\n{content[:3000]}")

        # Also group similarity-search results
        for topic, chunks in search_context.items():
            for chunk in chunks:
                path = chunk["metadata"].get("path", "unknown")
                parts = path.split("/")
                group_key = f"_search_{parts[0]}" if len(parts) > 1 else "_search_root"
                groups.setdefault(group_key, [])
                groups[group_key].append(
                    f"--- {path} ({chunk['metadata'].get('type', 'snippet')}) ---\n{chunk['text'][:1500]}"
                )

        # Limit to top groups by content size, combine small groups
        sorted_groups = sorted(groups.items(), key=lambda x: len("\n".join(x[1])), reverse=True)
        top_groups = sorted_groups[:MAX_SOURCE_GROUPS]

        print(f"  Summarizing {len(top_groups)} file groups in parallel...")

        async def summarize_group(group_key: str, file_texts: List[str]) -> str:
            context = "\n\n".join(file_texts)[:MAX_GROUP_CONTEXT_CHARS]
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=(
                    "You are a senior software architect. Analyze this code and files from a repository. "
                    "Produce a concise technical summary (3-5 sentences) covering: "
                    "1) What this code/directory does "
                    "2) Key patterns/architecture used "
                    "3) Main APIs, exports, or interfaces exposed "
                    "4) Notable dependencies or integrations. "
                    "Be specific — name actual files, functions, routes, models. No fluff."
                )),
                HumanMessage(content=f"Directory/Group: {group_key}\n\n{context}"),
            ])
            try:
                result = await self.fast_llm.ainvoke(prompt.format_messages())
                return f"[{group_key}]\n{result.content}"
            except Exception as e:
                print(f"  Group summary failed for '{group_key}': {e}")
                return f"[{group_key}] (summary failed: {e})"

        summaries = await asyncio.gather(
            *(summarize_group(key, texts) for key, texts in top_groups)
        )
        print(f"  Generated {len(summaries)} group summaries")
        return list(summaries)

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
    ) -> RepoBrief:
        """
        Combine all gathered context and produce the final RepoBrief.
        The LLM now has: repo metadata, ASCII tree, manifest data, framework detection,
        file-group summaries, and targeted similarity search results.
        """
        meta = structural_context["meta"]
        ascii_tree = structural_context["ascii_tree"]
        manifest_text = structural_context["manifest_text"]
        detected_frameworks = structural_context["detected_frameworks"]
        full_file_contents = structural_context["full_file_contents"]
        inferred_components = structural_context["inferred_components"]

        # Format full critical file contents for LLM
        critical_files_text = self._format_critical_files(
            structural_context["classification"],
            full_file_contents,
        )

        # Format search context
        search_text = self._format_search_context(search_context)

        # Format group summaries
        summaries_text = "\n\n".join(group_summaries)

        # Format detected frameworks
        frameworks_text = "\n".join(
            f"  - {fw}: {ver}" for fw, ver in detected_frameworks.items()
        )

        # Format inferred components
        components_text = "\n".join(
            f"  - /{c['name']}: {c['estimated_role']} [{c['primary_language']}]"
            for c in inferred_components
        )

        system_prompt = """You are an expert software architect producing a comprehensive repository brief.
You have been given:
1. Repository metadata and directory structure
2. Full contents of key files (README, manifests, configs, entry points)
3. Auto-detected frameworks and dependencies
4. Per-component summaries
5. Targeted code search results covering architecture, routing, models, auth, etc.

Your task: produce a HIGHLY ACCURATE RepoBrief that could let someone draw an architecture diagram.

Rules:
- Cite ACTUAL file paths, function names, and library versions you see in the context
- If you see package.json or requirements.txt, use the EXACT versions listed
- Do NOT invent components or patterns not evidenced in the provided context, BUT if the directory structure is flat (e.g. all files in root), logically group files into components (e.g. "Main Process", "Renderer Process", "Core Logic") so the components list is NEVER empty.
- If something is uncertain, say so in flagged_issues — don't hallucinate
- For architecture_summary: describe the actual request/response flow through the system
- For data_flow: trace data from user input → processing → storage → response. This field is REQUIRED.
- For dependency_graph: name actual components you can confirm exist. This field is REQUIRED.
- For detected_patterns: only include patterns you have evidence for. This field is REQUIRED.
- ALL fields in the JSON schema are strictly REQUIRED. You MUST provide a value, even if it is an empty list/dict or a string saying "None detected". Do NOT omit any fields."""

        user_prompt = f"""Repository: {repo_url}
Commit: {commit_sha}
GitHub metadata:
  - Primary language: {meta.get('language', 'Unknown')}
  - Stars: {meta.get('stars', 0)}
  - Description: {meta.get('description', 'No description')}
  - Topics: {', '.join(meta.get('topics', []))}

=== DIRECTORY STRUCTURE ===
{ascii_tree[:4000]}

=== AUTO-DETECTED FRAMEWORKS & DEPENDENCIES ===
{frameworks_text}

=== INFERRED COMPONENTS (from directory structure) ===
{components_text}

=== MANIFEST / DEPENDENCY DETAILS ===
{manifest_text[:6000]}

=== FULL CRITICAL FILES (README, configs, entry points) ===
{critical_files_text[:30000]}

=== PER-COMPONENT SUMMARIES ===
{summaries_text[:15000]}

=== TARGETED CODE SEARCH RESULTS ===
{search_text[:15000]}

Now produce the complete RepoBrief with ALL fields populated accurately."""

        print("  Calling reasoning LLM (thinking=ON) for final structured RepoBrief...")
        try:
            brief = await self.reasoning_llm_structured.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
        except Exception as e:
            print(f"  Structured output failed, falling back: {e}")
            brief = await self._synthesize_brief_fallback(
                repo_url, commit_sha, structural_context, detected_frameworks
            )

        # Post-process: inject ASCII tree and framework_details from our own detection
        # (LLM may not always populate these correctly)
        if not brief.directory_structure:
            brief.directory_structure = ascii_tree[:2000]
        if not brief.framework_details:
            brief.framework_details = detected_frameworks

        return brief

    def _format_critical_files(
        self,
        classification: Dict[str, List[str]],
        full_file_contents: Dict[str, str],
    ) -> str:
        """Format the full-read files into LLM-readable text, highest priority files first."""
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
            sections.append(f"=== {path} ===\n{content}")

        return "\n\n".join(sections)

    def _format_search_context(
        self,
        search_context: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """Format similarity search results grouped by topic."""
        sections = []
        for topic, chunks in search_context.items():
            if not chunks:
                continue
            sections.append(f"[Topic: {topic}]")
            for chunk in chunks:
                path = chunk["metadata"].get("path", "unknown")
                chunk_type = chunk["metadata"].get("type", "snippet")
                node_name = chunk["metadata"].get("node_name", "")
                label = f"{path} > {node_name}" if node_name else path
                sections.append(f"  -- {label} ({chunk_type}) --")
                sections.append(f"  {chunk['text'][:800]}")
        return "\n".join(sections)

    async def _synthesize_brief_fallback(
        self,
        repo_url: str,
        commit_sha: str,
        structural_context: Dict[str, Any],
        detected_frameworks: Dict[str, str],
    ) -> RepoBrief:
        """Fallback synthesis with reduced context if main call fails."""
        meta = structural_context["meta"]
        manifest_text = structural_context["manifest_text"]
        ascii_tree = structural_context["ascii_tree"]

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert software architect. Generate a RepoBrief JSON."),
            HumanMessage(content=f"""Repo: {repo_url}
Description: {meta.get('description', '')}
Language: {meta.get('language', '')}
Frameworks: {json.dumps(detected_frameworks)}
Directory tree:
{ascii_tree[:2000]}

Manifests:
{manifest_text[:3000]}

Generate a complete RepoBrief."""),
        ])
        return await self.reasoning_llm_structured.ainvoke(prompt.format_messages())
