"""
SmartFileSelector — index-based LLM file selection with accuracy gate.

Design:
1. Rule-based pre-scoring ranks the optional (source + docs) pool — no LLM cost.
2. If the optional pool fits within budget after must-keep files, skip LLM entirely.
3. Otherwise send a numbered list to the LLM; it returns INDICES only (not full paths).
   - Saves ~350-500 output tokens vs path-based output
   - No hallucination risk: backend maps index -> path
4. LLM has full freedom on count: select 0 to N files, no forced quota.
5. Accuracy gate: if must-keep alone exceeds MAX_FILE_COUNT, raise RepoTooComplexError.
   We NEVER truncate silently — accuracy is non-negotiable.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from app.services.llm import get_reasoning_llm
from app.services.github_client import MAX_FILE_COUNT, RepoTooComplexError
from app.config import settings

logger = logging.getLogger(__name__)

# ── Rule-based scoring constants ─────────────────────────────────────────────

# Directory segments that signal architectural importance
ARCH_DIRS = {
    "services", "core", "domain", "lib", "store", "context", "middleware",
    "infrastructure", "repositories", "usecases", "interactors", "adapters",
    "controllers", "handlers", "providers", "resolvers", "commands", "queries",
    "api", "internal", "pkg",
}

# Filename substrings that signal architectural importance
ARCH_NAMES = {
    "schema", "model", "store", "context", "pipeline", "handler",
    "router", "registry", "factory", "manager", "dispatcher", "processor",
    "config", "settings", "constants", "types", "interfaces", "base",
}

# Filename substrings that deprioritize (test / mock / generated)
TEST_NAMES = {"test", "spec", "mock", "fixture", "stub", "fake", "seed", "snapshot"}

# Extensions that are usually low-value source files (override if in global/theme dir)
STYLE_EXTS = {".css", ".scss", ".sass", ".less", ".styl"}

SIZE_BOOST_LARGE = 5_000   # bytes — likely non-trivial file
SIZE_BOOST_MEDIUM = 2_000  # bytes


def _rule_score(path: str, size_bytes: int) -> int:
    """
    Returns a score for an optional file. Higher = more important.
    Used to rank candidates before (and potentially instead of) the LLM call.
    """
    score = 0
    parts = path.lower().split("/")
    filename = parts[-1]
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

    # Directory signals
    for part in parts[:-1]:
        if part in ARCH_DIRS:
            score += 30
            break

    # Filename signals
    for name_hint in ARCH_NAMES:
        if name_hint in filename:
            score += 20
            break

    # Size signals
    if size_bytes >= SIZE_BOOST_LARGE:
        score += 15
    elif size_bytes >= SIZE_BOOST_MEDIUM:
        score += 10

    # Deprioritize tests / mocks
    for test_hint in TEST_NAMES:
        if test_hint in filename:
            score -= 20
            break

    # Deprioritize pure style files
    if ext in STYLE_EXTS:
        score -= 10

    return score


# ── LLM structured output schema ─────────────────────────────────────────────

class FileSelection(BaseModel):
    selected_indices: List[int] = Field(
        description=(
            "Indices of OPTIONAL files to include for analysis. "
            "Return ONLY the integer indices from the OPTIONAL section. "
            "Select as many as needed for accurate analysis — no forced quota. "
            "Omit indices for files that add no architectural value."
        )
    )
    reasoning: str = Field(
        description="1-2 sentence explanation of selection strategy."
    )


# ── Main selector ─────────────────────────────────────────────────────────────

class SmartFileSelector:
    def __init__(self):
        self.llm = get_reasoning_llm().with_structured_output(FileSelection)

    async def select_files(
        self,
        all_files: List[Dict[str, Any]],       # [{path, type, size, ...}] from list_files()
        classification: Dict[str, List[str]],  # from classify_files()
        ascii_tree: str,
        file_sizes: Optional[Dict[str, int]] = None,  # path -> bytes from tree_data
    ) -> List[Dict[str, Any]]:
        """
        Select architecturally significant files for chunking + embedding.

        Returns filtered list of file dicts. Raises RepoTooComplexError if
        must-keep files alone exceed MAX_FILE_COUNT (accuracy cannot be guaranteed).

        If smart_file_select_enabled is False, returns all files (up to MAX_FILE_COUNT).
        """
        file_sizes = file_sizes or {}

        # ── Step 0: Accuracy gate for must-keep overflow ──────────────────
        must_keep_paths = set(
            classification.get("critical", []) +
            classification.get("entry_points", []) +
            classification.get("config", []) +
            classification.get("ci_cd", [])
        )

        if len(must_keep_paths) > MAX_FILE_COUNT:
            logger.warning(
                f"This repository has {len(must_keep_paths)} critical/entry-point/config files, "
                f"which exceeds the {MAX_FILE_COUNT}-file chunking budget. "
                f"Proceeding anyway to maintain accuracy, but this may cause large embedding costs."
            )

        # ── Step 1: If smart selection disabled, return up to MAX_FILE_COUNT ──
        if not settings.smart_file_select_enabled:
            return all_files[:MAX_FILE_COUNT]

        # ── Step 2: Separate must-keep from optional ──────────────────────
        opt_paths = classification.get("source", []) + classification.get("docs", [])

        budget_remaining = MAX_FILE_COUNT - len(must_keep_paths)

        # If all optional files fit within budget, keep everything
        if len(opt_paths) <= budget_remaining:
            logger.info(
                f"[SmartFileSelector] All {len(opt_paths)} optional files fit within budget "
                f"({budget_remaining} slots). No LLM selection needed."
            )
            return all_files[:MAX_FILE_COUNT]

        # ── Step 3: Rule-based pre-scoring of optional pool ──────────────
        scored_opts = sorted(
            opt_paths,
            key=lambda p: _rule_score(p, file_sizes.get(p, 0)),
            reverse=True,   # highest score first
        )

        # If rule-based top-N exactly fills the budget, skip LLM
        if len(scored_opts) <= budget_remaining:
            selected_opt = set(scored_opts)
        elif _rule_score(scored_opts[budget_remaining - 1], file_sizes.get(scored_opts[budget_remaining - 1], 0)) != \
             _rule_score(scored_opts[budget_remaining], file_sizes.get(scored_opts[budget_remaining], 0)):
            # Clear score boundary at the cut-off: rule-based is decisive, skip LLM
            selected_opt = set(scored_opts[:budget_remaining])
            logger.info(
                f"[SmartFileSelector] Rule-based scoring decisive. "
                f"Selected {len(selected_opt)}/{len(opt_paths)} optional files."
            )
        else:
            # ── Step 4: LLM index-based selection ────────────────────────
            selected_opt = await self._llm_select(
                must_keep_paths=must_keep_paths,
                opt_paths=scored_opts,     # pre-scored order helps LLM
                budget_remaining=budget_remaining,
                ascii_tree=ascii_tree,
                file_sizes=file_sizes,
            )

        final_paths = must_keep_paths | selected_opt
        total = len(final_paths)

        if total > MAX_FILE_COUNT:
            logger.warning(
                f"Analysis requires {total} files but the chunking budget is {MAX_FILE_COUNT}. "
                "Proceeding with all required files to ensure accuracy."
            )

        logger.info(
            f"[SmartFileSelector] Final selection: {total} files "
            f"({len(must_keep_paths)} auto-included + {len(selected_opt)} optional). "
            f"Reduced from {len(all_files)} total."
        )

        return [f for f in all_files if f["path"] in final_paths]

    async def _llm_select(
        self,
        must_keep_paths: set,
        opt_paths: List[str],
        budget_remaining: int,
        ascii_tree: str,
        file_sizes: Dict[str, int],
    ) -> set:
        """
        Ask the LLM to select important files by INDEX (not full path).
        Saves ~350-500 output tokens vs path-based output.
        LLM has full freedom on count — no forced quota.
        """
        logger.info(
            f"[SmartFileSelector] Calling NVIDIA LLM to select from "
            f"{len(opt_paths)} optional files (budget: {budget_remaining} slots)..."
        )

        # Build numbered file list
        auto_lines = []
        for i, path in enumerate(sorted(must_keep_paths)):
            auto_lines.append(f"  {i}. [AUTO] {path}")

        opt_lines = []
        opt_index_map: Dict[int, str] = {}  # index -> path
        base_idx = len(must_keep_paths)
        for i, path in enumerate(opt_paths):
            idx = base_idx + i
            size_kb = file_sizes.get(path, 0) / 1024
            size_str = f" [{size_kb:.1f}KB]" if size_kb > 0 else ""
            opt_lines.append(f"  {idx}. [OPT]{size_str} {path}")
            opt_index_map[idx] = path

        system_prompt = (
            "You are an expert software architect selecting files for deep analysis.\n"
            "Files marked [AUTO] are always included — do NOT return their indices.\n"
            "Files marked [OPT] are candidates — return indices of those worth analyzing.\n\n"
            "Selection criteria (include if any apply):\n"
            "  - Core domain models, business logic, data schemas\n"
            "  - Service/handler implementations with non-trivial logic\n"
            "  - State management, pipeline orchestration\n"
            "  - Key algorithmic or data-processing components\n"
            "  - Central utilities used across multiple modules\n\n"
            "Exclude:\n"
            "  - Simple wrapper files with no logic\n"
            "  - Pure styling (CSS/SCSS) unless it's a global design system\n"
            "  - Generated or boilerplate code\n"
            "  - Test files (unless they demonstrate critical patterns not visible elsewhere)\n\n"
            "ACCURACY RULE: Select ONLY the files that actually matter to understand the project architecture and logic. "
            "Return the indices of all such files. Do not artificially cap your selection — accuracy is the priority. "
            "However, do not select files that are redundant or unnecessary for a high-level understanding."
        )

        user_prompt = (
            f"Directory structure:\n{ascii_tree[:3000]}\n\n"
            "AUTO-INCLUDED (always analyzed — do NOT select these):\n"
            + "\n".join(auto_lines) +
            "\n\nOPTIONAL (select indices of files needed for accurate analysis):\n"
            + "\n".join(opt_lines) +
            f"\n\nReturn selected_indices as a list of integers from the OPTIONAL section only."
        )

        try:
            result = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            # Map valid indices back to paths
            selected_paths = set()
            for idx in result.selected_indices:
                if idx in opt_index_map:
                    selected_paths.add(opt_index_map[idx])
                else:
                    logger.warning(f"[SmartFileSelector] LLM returned invalid index {idx}, skipping.")

            logger.info(
                f"[SmartFileSelector] LLM selected {len(selected_paths)} optional files. "
                f"Reasoning: {result.reasoning[:120]}"
            )
            return selected_paths

        except Exception as e:
            logger.error(
                f"[SmartFileSelector] LLM selection failed: {e}. "
                "Falling back to rule-based top-N."
            )
            # Fallback: rule-based top-N (already pre-scored order)
            return set(opt_paths[:budget_remaining])
