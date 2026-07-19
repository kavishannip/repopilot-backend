from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class FileReference(BaseModel):
    file_path: str = Field(description="The path to the file being referenced")
    description: str = Field(description="Why this file is important or what it contains")


class ComponentInfo(BaseModel):
    name: str = Field(description="Component name (usually directory or module name)")
    directory: str = Field(description="Primary directory path of this component")
    description: str = Field(description="What this component does and its role in the system")
    key_files: List[str] = Field(default_factory=list, description="Key files in this component")
    depends_on: List[str] = Field(default_factory=list, description="Other components this depends on")
    technologies: List[str] = Field(default_factory=list, description="Technologies used in this component")


class TechItem(BaseModel):
    name: str = Field(description="Technology name, e.g. 'React', 'FastAPI', 'PostgreSQL'")
    version: Optional[str] = Field(default=None, description="Version if detected, e.g. '18.x', '0.115.0'")
    category: str = Field(description="Category: 'framework', 'library', 'database', 'language', 'tool', 'platform'")
    purpose: str = Field(description="What role this technology plays in the project")


class LLMRepoBrief(BaseModel):
    # Core fields
    tech_stack: List[TechItem] = Field(
        description="List of main technologies with versions and categories"
    )
    architecture_summary: str = Field(
        description="Detailed description of overall architecture, how components interact, request/data flow"
    )
    entry_points: List[FileReference] = Field(
        description="Key files that serve as entry points: main files, route files, API handlers, core configs"
    )
    flagged_issues: List[str] = Field(
        description="Potential issues, missing documentation, technical debt, or security concerns"
    )
    summary: str = Field(
        description="High-level 2-3 paragraph summary: what the repo does, who it's for, key capabilities"
    )

    # Architectural detail
    components: List[ComponentInfo] = Field(
        description="Architectural components breakdown (frontend, backend, DB layer, etc.)"
    )
    data_flow: str = Field(
        description="How data flows through the system: user request → ... → response. Include key transformation steps."
    )
    detected_patterns: List[str] = Field(
        description="Architectural/design patterns detected: e.g. 'MVC', 'REST API', 'PWA', 'Server-Side Rendering', 'Event-driven', 'Monorepo'"
    )

    dependency_graph: Dict[str, List[str]] = Field(
        description="Component dependency map: {'frontend': ['backend_api', 'auth'], 'backend_api': ['database', 'cache']}"
    )


class RepoBrief(LLMRepoBrief):
    """
    Final output schema including statically generated fields.
    """
    framework_details: Dict[str, str] = Field(
        description="Detected frameworks and tools with versions: {'Next.js': '15.x', 'React': '18.x'}"
    )
    directory_structure: str = Field(
        description="ASCII tree showing project directory structure (full tree)"
    )
    language_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of code in each language (e.g. {'Go': 45.2, 'Java': 34.8})"
    )
    repo_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "All GitHub API metadata: stars, forks, watchers, open_issues, topics, homepage, "
            "license, created_at, updated_at, pushed_at, visibility, is_fork, archived, "
            "contributors_count, latest_release, size_kb, has_wiki, has_discussions, etc."
        )
    )
    project_types: List[str] = Field(
        default_factory=list,
        description=(
            "Detected project type markers from the file tree, e.g. ['Python', 'Django', 'Docker', 'Monorepo']. "
            "Derived from PROJECT_TYPE_MARKERS heuristics — not LLM-generated."
        )
    )
    selected_files: List[str] = Field(
        default_factory=list,
        description="Files selected as architecturally significant and used for deep analysis."
    )
