from pydantic import BaseModel, Field
from typing import List, Dict, Optional


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


class RepoBrief(BaseModel):
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
        default_factory=list,
        description="Architectural components breakdown (frontend, backend, DB layer, etc.)"
    )
    data_flow: str = Field(
        default="",
        description="How data flows through the system: user request → ... → response. Include key transformation steps."
    )
    detected_patterns: List[str] = Field(
        default_factory=list,
        description="Architectural/design patterns detected: e.g. 'MVC', 'REST API', 'PWA', 'Server-Side Rendering', 'Event-driven', 'Monorepo'"
    )

    # Rich metadata
    framework_details: Dict[str, str] = Field(
        default_factory=dict,
        description="Detected frameworks and tools with versions: {'Next.js': '15.x', 'React': '18.x'}"
    )
    directory_structure: str = Field(
        default="",
        description="ASCII tree showing project directory structure (top 3-4 levels)"
    )
    dependency_graph: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Component dependency map: {'frontend': ['backend_api', 'auth'], 'backend_api': ['database', 'cache']}"
    )
