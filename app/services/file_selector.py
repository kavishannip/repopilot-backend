import json
import logging
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm import get_reasoning_llm
from app.config import settings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SelectedFiles(BaseModel):
    files: List[str] = Field(description="List of exact file paths selected for analysis")
    reasoning: str = Field(description="Brief explanation of selection strategy")

class SmartFileSelector:
    def __init__(self):
        # User requested NVIDIA model for this critical task
        self.llm = get_reasoning_llm().with_structured_output(SelectedFiles)

    async def select_files(self, all_files: List[Dict[str, Any]], classification: Dict[str, List[str]], ascii_tree: str) -> List[Dict[str, Any]]:
        """
        Uses LLM to pick the most architecturally significant files.
        Always includes critical, config, and entry point files.
        Filters down 'source' and 'docs' files.
        """
        if not settings.smart_file_select_enabled or len(all_files) <= settings.smart_file_select_max_files:
            return all_files

        # Files we MUST keep
        must_keep_paths = set(
            classification.get("critical", []) +
            classification.get("entry_points", []) +
            classification.get("config", []) +
            classification.get("ci_cd", [])
        )

        # Files LLM gets to choose from (source + docs)
        choosable_paths = classification.get("source", []) + classification.get("docs", [])
        
        # If the mandatory files already exceed max (unlikely), just return top N
        if len(must_keep_paths) >= settings.smart_file_select_max_files:
            # We'll just return the must-keep files, maybe truncated
            sorted_must_keep = [f for f in all_files if f["path"] in must_keep_paths]
            return sorted_must_keep[:settings.smart_file_select_max_files]

        slots_available = settings.smart_file_select_max_files - len(must_keep_paths)
        if len(choosable_paths) <= slots_available:
            return all_files # We can keep everything

        logger.info(f"[SmartFileSelector] Filtering {len(choosable_paths)} source files down to ~{slots_available} using NVIDIA LLM...")

        system_prompt = f"""You are an expert software architect analyzing a repository.
Your task is to select up to {slots_available} files from a given list that are most critical for understanding the architecture, business logic, and core data flow.

Focus on:
1. Core domain models / database schemas
2. Main service / business logic implementations
3. Central state management
4. Key API route definitions (if not already included)
5. Complex algorithmic components

Avoid:
1. Simple UI components (unless central to the app)
2. Test files (unless demonstrating key patterns)
3. Auto-generated code
4. Purely decorative CSS/styling

Return ONLY exact file paths from the provided list."""

        user_prompt = f"""Directory Structure:
{ascii_tree[:4000]}

Files already included (DO NOT select these):
{json.dumps(list(must_keep_paths)[:50], indent=2)}

Available files to choose from (SELECT UP TO {slots_available}):
{json.dumps(choosable_paths, indent=2)}"""

        try:
            result = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            selected_paths = set(result.files)
            # Ensure they actually exist in choosable_paths to avoid hallucinations
            valid_selected = selected_paths.intersection(set(choosable_paths))
            
            final_keep_paths = must_keep_paths.union(valid_selected)
            
            logger.info(f"[SmartFileSelector] Selected {len(valid_selected)} source files. Total files for analysis: {len(final_keep_paths)} (down from {len(all_files)})")
            
            # Reconstruct list of dicts maintaining original order/score
            filtered_files = [f for f in all_files if f["path"] in final_keep_paths]
            return filtered_files
            
        except Exception as e:
            logger.error(f"[SmartFileSelector] LLM selection failed: {e}. Falling back to keeping all files.")
            return all_files[:settings.smart_file_select_max_files] # fallback truncation
